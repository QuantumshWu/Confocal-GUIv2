from .gui_device import *
from Confocal_GUIv2.helper import log_error, str2python, python2str

import importlib
from dataclasses import dataclass, field
import weakref
import ast
from typing import Any, Callable, Dict, Optional, List, Set, Tuple

def get_task_mod():
    """
    Minimal policy:
    - If DeviceManager exists, treat __main__ as the task module (notebook globals).
    - Else, fall back to Confocal_GUIv2.logic.task.
    """
    from Confocal_GUIv2.device import get_devices
    dm = get_devices()
    if dm is not None:
        return sys.modules.get("__main__", importlib.import_module("Confocal_GUIv2.logic.task"))
    return importlib.import_module("Confocal_GUIv2.logic.task")

def get_logic_mod():
    return importlib.import_module("Confocal_GUIv2.logic")

class TaskCanceled(Exception):
    """Internal-only cancellation used to unwind the task stack cleanly."""
    pass


# --------------------------- 1) Invoker ----------------------------
class GuiInvoker(QtCore.QObject):
    """
    Global, reusable invoker to run any callable on the GUI thread
    synchronously and return its result.

    MUST be created on the GUI (main) thread.
    """
    _sig = QtCore.pyqtSignal(object)  # carries a zero-arg callable

    def __init__(self, parent: Any = None):
        super().__init__(parent)
        # Slot executes on this object's thread (GUI if created there).
        self._sig.connect(self._on_invoke, QtCore.Qt.QueuedConnection)

    @QtCore.pyqtSlot(object)
    def _on_invoke(self, runner: Callable[[], None]) -> None:
        runner()

    def call(self, fn: Callable[[], Any]) -> Any:
        """
        Run `fn` on the invoker's thread (GUI) and return its result.
        - Fast path when already on GUI thread.
        - Uses a small local QEventLoop on the caller thread to wait safely.
        """
        if QtCore.QThread.currentThread() is self.thread():
            return fn()

        box = {"ret": None, "exc": None}
        loop = QtCore.QEventLoop()  # runs on the caller's thread

        def runner():
            try:
                box["ret"] = fn()
            except Exception as e:
                box["exc"] = e
            # Quit the caller's loop safely (post back to that thread).
            QtCore.QMetaObject.invokeMethod(loop, "quit", QtCore.Qt.QueuedConnection)

        self._sig.emit(runner)
        loop.exec_()

        if box["exc"] is not None:
            raise box["exc"]
        return box["ret"]


# --------------------------- 2) Proxy ------------------------------
class GuiProxy:
    """GET-only, lazy, thread-safe proxy.
    - Attribute reads resolved on the GUI thread.
    - Callables return a wrapper that executes on the GUI thread
      (re-resolves the bound method on every call).
    - Plain immutable values are returned as-is.
    - Everything else returns another GuiProxy (lazy chained access).
    """
    __slots__ = ("_invoker", "_getter")
    _PLAIN = (int, float, str, bool, type(None), bytes, tuple, frozenset)

    def __init__(self, invoker, getter):
        object.__setattr__(self, "_invoker", invoker)
        object.__setattr__(self, "_getter", getter)

    def __getattr__(self, name: str):
        inv, get = self._invoker, self._getter

        # Peek once on the GUI thread to decide wrapping strategy
        val = inv.call(lambda: getattr(get(), name))

        # Callable → wrapper that re-resolves each call (stale-proof) on GUI thread
        if callable(val):
            def _wrapped(*args, **kwargs):
                return inv.call(lambda: getattr(get(), name)(*args, **kwargs))
            _wrapped.__name__ = getattr(val, "__name__", name)
            _wrapped.__doc__  = getattr(val, "__doc__", None)
            return _wrapped

        # Plain immutables → return by value
        if isinstance(val, self._PLAIN):
            return val

        # Other objects (matplotlib Figure/Axes, DataFigure, lists, dicts, np arrays, …)
        # → keep chaining lazily to ensure any later call still runs on the GUI thread
        return GuiProxy(inv, lambda: getattr(get(), name))


# -------------------------- 3) Hooks (func) ------------------------
def caller_in_gui(invoker: GuiInvoker, plot_widget: Any) -> GuiProxy:
    """
    Start the plot tab, wait for RUNSTATE==STOP (event-driven), then return
    a GET-ONLY GuiProxy for plot_widget.live_plot.

    This function touches GUI objects only on the GUI thread; waiting happens
    on the caller thread via a temporary QEventLoop.

    Example:
        inv = GuiInvoker(parent=main_gui)   # create on GUI thread
        live_plot = caller_in_gui(inv, plot_widget)
        # Now from a worker thread:
        live_plot.data_figure.save("x.png") # runs on GUI thread
    """
    # 1) Start on GUI thread (must be quick/non-blocking)
    invoker.call(lambda: plot_widget.on_start())

    # 2) Event-driven wait for STOP in caller thread
    loop = QtCore.QEventLoop()

    def connect_on_gui():
        sm = getattr(plot_widget, "stateui_manager", None)

        def on_changed(new_state):
            if getattr(new_state, "name", None) == "STOP":
                QtCore.QMetaObject.invokeMethod(loop, "quit", QtCore.Qt.QueuedConnection)

        sm.runstateChanged.connect(on_changed, QtCore.Qt.QueuedConnection)

        # Fast path: already STOP
        try:
            if getattr(sm, "runstate", None) and getattr(sm.runstate, "name", None) == "STOP":
                QtCore.QMetaObject.invokeMethod(loop, "quit", QtCore.Qt.QueuedConnection)
        except Exception:
            pass

    invoker.call(connect_on_gui)
    loop.exec_()

    # 3) Return a ready-to-use proxy for live_plot (GET-ONLY)
    return GuiProxy(invoker, lambda: getattr(plot_widget, "live_plot", None))



# --------------------------------------------------------------------------
# 4) Targeted monkey-patch: only the caller names you specify in plan
# --------------------------------------------------------------------------

@dataclass
class TaskPlan:
    task_func: Callable[..., Any]
    task_kwargs: Dict[str, Any]
    bound_a: Dict[str, Any] = field(default_factory=dict)   # caller_name -> plot_widget (Existing tab)
    bound_c: Set[str] = field(default_factory=set)          # caller_name set for Shared Slot mode


class patched_callers:
    """
    Context manager that patches *caller symbols inside the task module's namespace*.

    Now supports two binding modes per caller:
      - Existing Tab (formerly A): use a user-selected existing plot tab.
      - Shared Slot (formerly C): all callers marked C share a single left-side tab slot per runner.

    It also cooperates with TaskRunner to honor stop requests without touching task source.
    """
    def __init__(self, plan: TaskPlan, invoker: GuiInvoker, main_gui: Any, runner: "TaskRunner"):
        self.plan = plan
        self.invoker = invoker
        self.main_gui = main_gui
        self._task_mod = None
        self._originals: Dict[str, Callable] = {}
        self._runner = runner
        # Shared slot for all C-callers within this runner:
        self._c_slot = None  # dict: {"tw":..., "idx":..., "plot":...}

    def __enter__(self):
        self._task_mod = get_task_mod()

        def _wrap(caller_name: str):
            def wrapped(*args, **kwargs):
                # 0) Pre-check stop before doing any GUI work
                if self._runner.is_stop_requested():
                    raise TaskCanceled()

                # 1) Decide binding: Existing Tab vs Shared Slot
                plot_widget = self.plan.bound_a.get(caller_name, None)
                if plot_widget is None:  # Shared Slot path
                    # (a) Find Measurement class behind the logic caller
                    logic_map = _measurement_callers_in_logic()
                    bound_meth = logic_map.get(caller_name, None)
                    if bound_meth is None:
                        raise RuntimeError(f"Unknown caller '{caller_name}' in logic.")
                    meas_cls = getattr(bound_meth, "__self__", None)
                    if not isinstance(meas_cls, type):
                        raise RuntimeError(f"Caller '{caller_name}' is not bound to a measurement class.")

                    # (b) Build/replace the shared slot on the LEFT side
                    def _ensure_shared_slot():
                        # Close previous slot if still alive
                        if self._c_slot is not None:
                            tw, idx, _plot = self._c_slot["tw"], self._c_slot["idx"], self._c_slot["plot"]
                            try:
                                if tw.indexOf(tw.widget(idx)) != -1:
                                    self.main_gui.close_tab(idx, tw)
                            except Exception:
                                pass
                            self._c_slot = None

                        # Add a new tab at the end on the LEFT side using the requested class
                        tw_left = self.main_gui.tabs_left
                        ok = self.main_gui.add_tab(tw_left, self.main_gui.combo_cls_left, meas_cls=meas_cls, ov=kwargs, origin='task')
                        if ok is False:
                            # Likely a leftover from a previous task: close LAST tab on the left
                            last = tw_left.count() - 1
                            if last >= 0:
                                self.main_gui.close_tab(last, tw_left)
                            self.main_gui.add_tab(tw_left, self.main_gui.combo_cls_left, meas_cls=meas_cls, ov=kwargs, origin='task')
                            # try one more time

                        new_idx = tw_left.count() - 1
                        cont = tw_left.widget(new_idx)
                        plot = cont.layout().itemAt(0).widget() if cont and cont.layout() and cont.layout().count() else None
                        if plot is None:
                            raise RuntimeError("Failed to locate plot widget for shared slot.")
                        self._c_slot = {"tw": tw_left, "idx": new_idx, "plot": plot}
                        return plot

                    plot_widget = self.invoker.call(_ensure_shared_slot)

                # 2) Register plot for stop-all semantics
                self._runner._register_plot(plot_widget)

                # 3) Run caller_in_gui to start plot & wait STOP
                live_proxy = caller_in_gui(self.invoker, plot_widget)

                # 4) Post-check stop: if a stop was requested while we were running,
                #    exit cleanly before yielding back to the task body.
                if self._runner.is_stop_requested():
                    raise TaskCanceled()

                return live_proxy
            return wrapped

        """
        # Patch only symbols present in task module and in plan (A or C).
        names_to_patch = set(self.plan.bound_a.keys()) | set(self.plan.bound_c)
        for name in names_to_patch:
            if hasattr(self._task_mod, name):
                self._originals[name] = getattr(self._task_mod, name)
                setattr(self._task_mod, name, _wrap(name))
            else:
                print(f"[patched_callers] symbol not found in task module: {name}")

        return self
        """
        self._MISSING = object()

        # Patch only symbols present in the task *function's* globals and in plan (A or C).
        g = self.plan.task_func.__globals__
        names_to_patch = set(self.plan.bound_a.keys()) | set(self.plan.bound_c)
        for name in names_to_patch:
            if name in g:
                self._originals[name] = g.get(name, self._MISSING)
                g[name] = _wrap(name)
            else:
                print(f"[patched_callers] symbol not found in task function globals: {name}")

        return self

    def __exit__(self, exc_type, exc, tb):
        # Restore originals in the task *function's* globals
        g = self.plan.task_func.__globals__
        for name, orig in self._originals.items():
            if orig is self._MISSING:
                g.pop(name, None)
            else:
                g[name] = orig
        self._originals.clear()

    def __exit11(self, exc_type, exc, tb):
        # Restore originals
        if self._task_mod is not None:
            for name, fn in self._originals.items():
                setattr(self._task_mod, name, fn)
        self._originals.clear()
        self._task_mod = None


# --------------------------- TaskRunner ----------------------------
class TaskRunner(QtCore.QThread):
    """
    Runs the provided task function in a worker thread while temporarily
    patching caller symbols in the *task module* so that task code will
    receive a proxied `live_plot` via `caller_in_gui(...)`.
    """
    sig_running = QtCore.pyqtSignal(bool)   # True when task starts, False when finishes
    sig_log = QtCore.pyqtSignal(str)

    def __init__(self, main_gui: Any, plan: TaskPlan, parent=None):
        super().__init__(parent)
        self._main_gui = main_gui
        self._plan = plan
        self._invoker = GuiInvoker(parent=main_gui)
        self._stop_requested = False
        self._plots: Set[Any] = set()  # plots we may need to stop on cancel

    # -- public API --
    def request_stop(self):
        """Signal the task to stop ASAP and stop any active plots on the GUI thread."""
        self._stop_requested = True

        def _stop_all_plots():
            for p in list(self._plots):
                try:
                    # Prefer the GUI's standard stop entrypoint
                    if hasattr(p, "on_stop"):
                        p.on_stop()
                    elif getattr(p, "stateui_manager", None):
                        # Fallback: try controller if available
                        ctrl = getattr(p, "live_plot", None)
                        if ctrl and getattr(ctrl, "controller", None):
                            ctrl.controller.stop()
                except Exception:
                    pass

        # Stop plots on GUI thread
        self._invoker.call(_stop_all_plots)

    def is_stop_requested(self) -> bool:
        return self._stop_requested

    # -- internal: patched_callers will register new plots here --
    def _register_plot(self, plot_widget: Any):
        if plot_widget is not None:
            self._plots.add(plot_widget)

    def run(self):
        try:
            self.sig_running.emit(True)
            with patched_callers(self._plan, self._invoker, self._main_gui, self):
                self._plan.task_func(**self._plan.task_kwargs)
        except TaskCanceled:
            # Clean, user-initiated stop; do not log as error
            self.sig_log.emit("[TaskRunner] Task canceled by user.")
        except Exception as e:
            log_error(e)
        finally:
            self.sig_running.emit(False)

# --------------------------------------------------------------------------
# 6) Lightweight TaskDialog to launch tasks with binding
# --------------------------------------------------------------------------

# ---------- helpers to discover tasks and callers ----------

def _discover_tasks() -> Tuple[Any, List[Tuple[str, Callable]]]:
    """
    Discover any callable whose name ends with '_task' in the resolved module's dict.
    This includes functions imported into the notebook namespace.
    """
    mod = get_task_mod()
    tasks: List[Tuple[str, Callable]] = []
    for name, obj in vars(mod).items():  # direct dict scan: includes imported names
        if name.endswith("_task") and callable(obj):
            tasks.append((name, obj))
    tasks.sort(key=lambda x: x[0].lower())
    return mod, tasks


def _list_tabs(main_gui):
    tabs = []
    for tw in (main_gui.tabs_left, main_gui.tabs_right):
        if tw is None:
            continue
        for i in range(tw.count()):
            cont = tw.widget(i)
            plot = cont.layout().itemAt(0).widget() if cont and cont.layout() and cont.layout().count() else None
            if plot is not None:
                tabs.append((tw, i, plot))
    return tabs


def _measurement_callers_in_logic() -> Dict[str, Callable]:
    """
    Return {caller_name: bound classmethod} for BaseMeasurement subclasses.
    Accepts both 'classmethod caller(...)' patterns and common wrapping.
    """
    logic = get_logic_mod()
    result: Dict[str, Callable] = {}

    BM = getattr(logic, "BaseMeasurement", None)

    for name, obj in inspect.getmembers(logic):
        if not callable(obj):
            continue

        # We want classmethod bound to a class: obj.__self__ is a 'type'
        bound_to = getattr(obj, "__self__", None)
        func = getattr(obj, "__func__", None)  # underlying function for classmethod

        if bound_to is None or not isinstance(bound_to, type):
            continue
        if BM is not None and not issubclass(bound_to, BM):
            continue

        # Be tolerant to wrappers: get the underlying function name if possible
        fn_name = getattr(func, "__name__", None) if func is not None else getattr(obj, "__name__", None)
        if fn_name != "caller":
            continue

        result[name] = obj

    return result


def _guess_callers_from_task(task_fn: Callable) -> List[str]:
    """Return measurement callers that are directly called by bare names, e.g. pl(...)."""
    try:
        src = inspect.getsource(task_fn)
    except Exception:
        src = ""
    try:
        tree = ast.parse(src) if src else None
    except Exception:
        tree = None

    logic_callers_map = _measurement_callers_in_logic()
    logic_callers: Set[str] = set(logic_callers_map.keys())
    hits: Set[str] = set()

    if tree is not None:
        class _Visitor(ast.NodeVisitor):
            def visit_Call(self, node: ast.Call):
                # Only count calls where the callee is a bare Name: pl(...)
                func = node.func
                if isinstance(func, ast.Name) and func.id in logic_callers:
                    hits.add(func.id)
                # Continue walking the tree
                self.generic_visit(node)

        _Visitor().visit(tree)

    return sorted(hits)

class _CallerRow(QWidget):
    """
    One line of caller binding:
      [Caller combo]  ( ) Existing Tab   (•) Shared Slot (temp)  [Tab combo when Existing]
      [Remove button]
    """
    def __init__(self, main_gui, caller_candidates: List[str], parent=None):
        super().__init__(parent)
        self._main_gui = main_gui
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0,0,0,0)
        lay.setSpacing(6)

        self.combo_caller = QComboBox()
        self.combo_caller.setFixedSize(100, 30)
        self.combo_caller.setEditable(True)
        for c in caller_candidates:
            self.combo_caller.addItem(c)

        lbl = QLabel("Caller:")
        lbl.setFixedHeight(30)
        lay.addWidget(lbl)
        lay.addWidget(self.combo_caller)

        self.radio_existing = QRadioButton("Existing Tab")
        self.radio_existing.setFixedHeight(30)
        self.radio_shared   = QRadioButton("Shared Slot (temp)")
        self.radio_shared.setFixedHeight(30)
        self.radio_shared.setChecked(True)  # default to Shared Slot
        lay.addWidget(self.radio_existing)
        lay.addWidget(self.radio_shared)

        self.combo_tab = QComboBox()
        self.combo_tab.setFixedSize(100, 30)
        self._refresh_tabs()
        self.combo_tab.setEnabled(False)
        lbl = QLabel("Bind to tab:")
        lbl.setFixedHeight(30)
        lay.addWidget(lbl)
        lay.addWidget(self.combo_tab)

        self.btn_remove = QPushButton("–")
        self.btn_remove.setFixedWidth(24)
        self.btn_remove.setFixedHeight(30)
        lay.addWidget(self.btn_remove)

        self.radio_existing.toggled.connect(lambda checked: self.combo_tab.setEnabled(checked))

    def _refresh_tabs(self):
        """Rebuild the Existing-Tab dropdown from current Main GUI tabs (left/right).
        - Preserve previously selected item if it still exists.
        - Do NOT alter enable/disable state; that is controlled by radio_existing toggling.
        """
        # Remember current selection (by display text + index) to improve robustness
        prev_text = self.combo_tab.currentText()
        prev_idx  = self.combo_tab.currentIndex()

        self.combo_tab.blockSignals(True)
        try:
            self.combo_tab.clear()

            # Collect current tabs from the main GUI
            tabs = _list_tabs(self._main_gui)  # [(tw, idx, plot), ...]
            for tw, idx, plot in tabs:
                # Build user-friendly title
                title = None
                cls_name = getattr(plot, "cls_name", None)
                if cls_name is not None:
                    title = getattr(cls_name, "name", None) or str(cls_name)
                if not title:
                    title = "Unknown"
                side = "L" if tw is self._main_gui.tabs_left else "R"
                text = f"{side}[{idx}] - {title}"
                # Store full tuple as item data for later retrieval
                self.combo_tab.addItem(text, (tw, idx, plot))

            # Try to restore previous selection if it still matches something
            if self.combo_tab.count() > 0:
                # Prefer exact text match
                if prev_text:
                    i = self.combo_tab.findText(prev_text)
                    if i >= 0:
                        self.combo_tab.setCurrentIndex(i)
                    elif 0 <= prev_idx < self.combo_tab.count():
                        self.combo_tab.setCurrentIndex(prev_idx)
                    else:
                        self.combo_tab.setCurrentIndex(0)
                else:
                    self.combo_tab.setCurrentIndex(0)
        finally:
            self.combo_tab.blockSignals(False)

    # -- API to read config --
    def read(self) -> Tuple[str, str, Optional[Tuple]]:
        """Return (caller_name, mode, tab_data|None). mode in {'A','C'} where A=Existing, C=Shared."""
        name = self.combo_caller.currentText().strip()
        mode = 'A' if self.radio_existing.isChecked() else 'C'
        tab_data = self.combo_tab.currentData() if mode == 'A' else None
        return name, mode, tab_data


class TaskDialog(QWidget):
    """
    Dialog-only implementation with per-task kwargs caching.

    What it guarantees:
    - Each task has its own cached kwargs text (self._kwargs_cache).
    - Switching tasks: save previous edits, then load the new task's cached kwargs,
      or populate defaults from the function signature if no cache exists.
    - showEvent: refresh UI without wiping user edits (silent combo rebuild +
      restore previous selection; only populate when needed).
    - "Reset kwargs" button (placed to the LEFT of Start) resets ONLY the current
      task's kwargs to defaults (does NOT clear to empty).
    """
    def __init__(self, main_gui, parent=None):
        super().__init__(parent)
        self._main_gui = main_gui
        self._runner: Optional[TaskRunner] = None
        self._caller_rows: List[_CallerRow] = []
        self._kwargs_cache: Dict[str, str] = {}   # task_name -> kwargs text
        self._last_task_name: Optional[str] = None
        self._build_ui()

    # ---------------- UI ----------------
    def _build_ui(self):
        _mod, tasks = _discover_tasks()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 8, 18, 8)

        # Task chooser
        row = QHBoxLayout()
        lbl = QLabel("Task:")
        lbl.setFixedHeight(30)
        row.addWidget(lbl)
        self.combo_task = QComboBox()
        self.combo_task.setFixedHeight(30)
        for name, fn in tasks:
            self.combo_task.addItem(name, fn)  # keep function in userData
        row.addWidget(self.combo_task, 1)
        layout.addLayout(row)

        # Kwargs editor
        self.edit_kwargs = DynamicPlainTextEdit(
            text="",
            placeholder_text="Each line as  key = value   (leave empty => None; Python literals supported)"
        )
        self.edit_kwargs.setMinimumHeight(120)
        layout.addWidget(self.edit_kwargs)

        # Caller binding group
        box = QGroupBox("Caller Binding")
        v = QVBoxLayout(box)
        self.rows_area = QVBoxLayout()
        v.addLayout(self.rows_area)

        # Row controls
        ctrl = QHBoxLayout()
        self.btn_add_row = QPushButton("Add Caller")
        self.btn_add_row.setFixedHeight(30)
        ctrl.addStretch()
        ctrl.addWidget(self.btn_add_row)
        v.addLayout(ctrl)
        layout.addWidget(box)

        # Status + Stop / Reset / Start (Reset must be to the LEFT of Start)
        runrow = QHBoxLayout()
        self.lbl_running = QLabel("")
        self.lbl_running.setFixedHeight(30)
        runrow.addWidget(self.lbl_running)
        runrow.addStretch()

        
        self.btn_reset = QPushButton("Reset kwargs")  # reset current task to defaults
        self.btn_reset.setFixedSize(130, 30)          # placed left of Start
        self.btn_stop = QPushButton("Stop")
        self.btn_stop.setFixedSize(100, 30)
        self.btn_start = QPushButton("Start")
        self.btn_start.setFixedSize(100, 30)

        self.btn_stop.setEnabled(False)

        runrow.addWidget(self.btn_reset)  # Reset (left)
        runrow.addWidget(self.btn_stop)
        runrow.addWidget(self.btn_start)  # Start (right)
        layout.addLayout(runrow)

        # Signals
        self.btn_add_row.clicked.connect(self._on_add_row)
        self.btn_start.clicked.connect(self._on_start)
        self.btn_stop.clicked.connect(self._on_stop_clicked)
        self.btn_reset.clicked.connect(self._on_reset_clicked)
        self.combo_task.currentIndexChanged.connect(self._on_task_changed)

        # Initialize: if there are tasks, select the first and populate defaults
        if self.combo_task.count() > 0:
            self.combo_task.setCurrentIndex(0)
            self._on_task_changed()

    # ---------------- Rows ----------------
    def _clear_rows(self):
        """Remove all caller binding rows."""
        for i in reversed(range(self.rows_area.count())):
            item = self.rows_area.itemAt(i)
            w = item.widget()
            if w is not None:
                w.setParent(None)
        self._caller_rows.clear()

    def _on_add_row(self, preset_name: Optional[str] = None):
        """Append a caller row; preselect by name if provided."""
        candidates = list(getattr(self, "_task_callers", []))
        row = _CallerRow(self._main_gui, candidates, parent=self)
        if preset_name:
            idx = row.combo_caller.findText(preset_name)
            if idx >= 0:
                row.combo_caller.setCurrentIndex(idx)
        row.btn_remove.clicked.connect(lambda: self._remove_row(row))
        self.rows_area.addWidget(row)
        self._caller_rows.append(row)

    def _remove_row(self, row: _CallerRow):
        """Remove a specific caller row."""
        if row in self._caller_rows:
            self._caller_rows.remove(row)
            row.setParent(None)

    def _refresh_tabs(self):
        """Ask each row to refresh its Existing-Tab dropdown."""
        for row in getattr(self, "_caller_rows", []):
            try:
                row._refresh_tabs()
            except Exception:
                pass

    # ---------------- Kwargs helpers ----------------
    def _current_task_name(self) -> str:
        """Return the visible name of the selected task."""
        return self.combo_task.currentText().strip()

    def _save_current_kwargs(self) -> None:
        """Save the editor text into cache for the CURRENT task."""
        name = self._current_task_name()
        if name:
            self._kwargs_cache[name] = self.edit_kwargs.toPlainText()

    def _load_cached_or_defaults(self, task_name: str, task_fn: Optional[Callable]) -> None:
        """
        Load kwargs text for 'task_name':
        - Use cache if available;
        - Otherwise populate defaults from function signature.
        """
        if not callable(task_fn):
            self.edit_kwargs.setPlainText("")
            return
        cached = self._kwargs_cache.get(task_name, None)
        if cached is not None:
            self.edit_kwargs.setPlainText(cached)
        else:
            self._populate_kwargs_from_signature(task_fn)

    def _ensure_kwargs_present(self) -> None:
        """
        If the editor is empty (first show / no cache), fill it with cached text
        or fallback defaults. If non-empty, DO NOT overwrite user edits.
        """
        if self.edit_kwargs.toPlainText().strip():
            return
        self._load_cached_or_defaults(self._current_task_name(), self.combo_task.currentData())

    def _fill_missing_kwargs_with_none(self, task_fn: Callable, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure all non-var positional/keyword parameters exist in kwargs (value None if missing)."""
        try:
            sig = inspect.signature(task_fn)
        except Exception:
            return kwargs
        for name, p in sig.parameters.items():
            if name == "self":
                continue
            if p.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
                continue
            if name not in kwargs:
                kwargs[name] = None
        return kwargs

    def _populate_kwargs_from_signature(self, task_fn: Callable) -> None:
        """Populate editor with defaults derived from the function signature."""
        try:
            sig = inspect.signature(task_fn)
        except Exception:
            self.edit_kwargs.setPlainText("")
            return
        lines: List[str] = []
        for name, p in sig.parameters.items():
            if name == "self":
                continue
            if p.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
                continue
            if p.default is inspect._empty or p.default is None:
                lines.append(f"{name} = None")
            else:
                lines.append(f"{name} = {python2str(p.default)}")
        self.edit_kwargs.setPlainText("\n".join(lines))

    def _suggest_callers_for_task(self, task_fn: Callable) -> List[str]:
        """Return caller names referenced directly by the task; fallback to all available callers."""
        cand = _guess_callers_from_task(task_fn)
        if not cand:
            cand = sorted(_measurement_callers_in_logic().keys())
        return cand

    def _parse_kwargs(self) -> Dict[str, Any]:
        """Parse the editor text into a dict of kwargs."""
        text = self.edit_kwargs.toPlainText()
        kwargs: Dict[str, Any] = {}
        for raw in text.splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                raise ValueError(f"Missing '=' in line: {raw}")
            k, v = line.split("=", 1)
            key = k.strip()
            val_str = v.strip()
            if val_str == "":
                kwargs[key] = None
            else:
                kwargs[key] = str2python(val_str)
        return kwargs

    # ---------------- Run controls ----------------
    def _set_running_ui(self, running: bool):
        """Enable/disable controls based on running state."""
        self.lbl_running.setText("Running..." if running else "")
        self.btn_start.setEnabled(not running)
        self.btn_stop.setEnabled(running)
        self.btn_reset.setEnabled(not running)
        self.btn_add_row.setEnabled(not running)

    def _on_stop_clicked(self):
        """Request the running task to stop."""
        if self._runner is not None:
            self._runner.request_stop()

    def _on_reset_clicked(self):
        """
        Reset ONLY the current task's kwargs to defaults (not empty),
        and write the new defaults into the cache immediately.
        """
        task_fn = self.combo_task.currentData()
        if callable(task_fn):
            self._populate_kwargs_from_signature(task_fn)
            self._kwargs_cache[self._current_task_name()] = self.edit_kwargs.toPlainText()

    def _on_start(self):
        """Parse kwargs, build the plan, and launch the TaskRunner."""
        try:
            # Persist current editor text into cache for the currently selected task
            self._save_current_kwargs()

            task_fn = self.combo_task.currentData()
            if not callable(task_fn):
                QtWidgets.QMessageBox.warning(self, "Error", "No task selected.")
                return

            kwargs = self._parse_kwargs()
            kwargs = self._fill_missing_kwargs_with_none(task_fn, kwargs)

            # Build plan (caller bindings may be empty)
            plan = TaskPlan(task_func=task_fn, task_kwargs=kwargs)
            for row in self._caller_rows:
                name, mode, tab_data = row.read()
                if not name:
                    continue
                if mode == 'A':
                    if not tab_data:
                        QtWidgets.QMessageBox.warning(self, "Error", f"Caller '{name}': please select a tab for Existing Tab mode.")
                        return
                    _tw, _idx, plot = tab_data
                    plan.bound_a[name] = plot
                else:
                    plan.bound_c.add(name)

            # Create & register runner
            runner = TaskRunner(self._main_gui, plan, parent=self._main_gui)
            self._runner = runner
            if not hasattr(self._main_gui, "_task_runners"):
                self._main_gui._task_runners = []
            self._main_gui._task_runners.append(runner)

            # Wire signals
            runner.sig_running.connect(self._set_running_ui)
            runner.sig_running.connect(self._main_gui._on_task_running_changed)
            runner.finished.connect(lambda: self._on_runner_finished(runner))

            runner.start()
            print("[TaskDialog] runner started")

        except Exception as e:
            log_error(e)
            QMessageBox.critical(self, "Error", f"{type(e).__name__}: {e}")

    def _on_runner_finished(self, runner: TaskRunner):
        """Cleanup after the runner finishes."""
        try:
            if hasattr(self._main_gui, "_task_runners") and runner in self._main_gui._task_runners:
                self._main_gui._task_runners.remove(runner)
        except Exception:
            pass
        finally:
            self._set_running_ui(False)

    # ---------------- Lifecycle ----------------
    def showEvent(self, event):
        """
        Refresh tabs/tasks without overwriting user edits:
        - Save current task's kwargs into cache.
        - Rebuild the task combo silently and try to restore the previous selection.
        - If restoration fails (first show or task list changed), explicitly load
          cache/defaults for the new selection and refresh caller rows.
        - Finally, if the editor is still empty, ensure it is populated.
        """
        # Save current kwargs first
        self._save_current_kwargs()

        # Refresh available tabs for caller rows
        self._refresh_tabs()

        # Rebuild task combo silently
        _mod, tasks = _discover_tasks()
        prev_name = self._current_task_name()

        self.combo_task.blockSignals(True)
        try:
            self.combo_task.clear()
            for name, fn in tasks:
                self.combo_task.addItem(name, fn)

            restored = False
            if prev_name:
                i = self.combo_task.findText(prev_name)
                if i >= 0:
                    self.combo_task.setCurrentIndex(i)
                    restored = True
            if not restored and self.combo_task.count() > 0:
                self.combo_task.setCurrentIndex(0)
        finally:
            self.combo_task.blockSignals(False)

        # If we couldn't restore, load for the current selection and rebuild callers
        current_fn = self.combo_task.currentData()
        current_name = self._current_task_name()
        if not (prev_name and self.combo_task.findText(prev_name) >= 0):
            self._load_cached_or_defaults(current_name, current_fn)
            if callable(current_fn):
                self._task_callers = self._suggest_callers_for_task(current_fn)
                self._clear_rows()
                for n in self._task_callers:
                    self._on_add_row(preset_name=n)
            self._last_task_name = current_name

        # If editor is empty for any reason, populate cache/defaults
        self._ensure_kwargs_present()

        super().showEvent(event)

    def _on_task_changed(self):
        """
        Handle user selection change:
        - Save previous task's edits to cache.
        - Load new task's cached/default kwargs.
        - Rebuild caller rows for the new task.
        """
        # Save previous task edits
        if self._last_task_name:
            self._kwargs_cache[self._last_task_name] = self.edit_kwargs.toPlainText()

        task_fn = self.combo_task.currentData()
        new_name = self._current_task_name()

        if callable(task_fn):
            self._load_cached_or_defaults(new_name, task_fn)
            self._task_callers = self._suggest_callers_for_task(task_fn)
            self._clear_rows()
            for name in self._task_callers:
                self._on_add_row(preset_name=name)

        self._last_task_name = new_name

    # (Optional) Keep default behavior for these events
    def closeEvent(self, event):
        super().closeEvent(event)

    def hideEvent(self, event):
        win = self.window()
        if getattr(win, '_hiding_via_close', False):
            pass
        super().hideEvent(event)



global task_window
task_window = None

def run_task_panel(main_gui):
    global task_window
    if task_window is None:
        task_window = run_fluent_window(use_fluent=True, widget_class=TaskDialog, widget_kwargs={'main_gui':main_gui}, 
        title='TaskGUI@Wanglab, UOregon', ui_path=None, in_GUI=True)
    else:
        task_window = run_fluent_window(in_GUI=True, window_handle = task_window)

    return task_window