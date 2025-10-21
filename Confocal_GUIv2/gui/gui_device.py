from .base import *

import io
import numpy as np
import json
from enum import Enum, auto


from PyQt5.QtCore import QMimeData, QTimer
from PyQt5.QtGui import QPalette, QDrag, QPixmap, QIcon


overrides = {
    'QCheckBox':      ToggleSwitch,
    'QPushButton':    FluentButton,
    'QComboBox':      FluentComboBox,
    'QLineEdit':      FluentLineEdit,
    'QLabel':         FluentLabel,
    'QGroupBox':      FluentGroupBox,
    'QSlider':        FluentSlider,
    'QDoubleSpinBox': FluentDoubleSpinBox,
    'QFrame':         FluentFrame,
    'QMessageBox':    FluentMessageBox,
    'QTextEdit':      FluentTextEdit,
    'QPlainTextEdit': FluentPlainTextEdit,
    'QStackedWidget': FluentStackedWidget,
    'QTabWidget':     FluentTabWidget,
    'QTableWidget':   FluentTableWidget,
    'QHeaderView':    FluentHeaderView,
    'QRadioButton':   FluentRadioButton,

}

for name, cls in overrides.items():
    setattr(QtWidgets, name, cls)
# override before import QPushButton etc.

from PyQt5.QtWidgets import QCheckBox, QPushButton, QComboBox, QLineEdit, QLabel, QGroupBox, QSlider, QDoubleSpinBox, QFrame,\
QMessageBox, QTextEdit, QPlainTextEdit, QStackedWidget, QTabWidget, QTableWidget, QHeaderView, QRadioButton

_base_dir = os.path.dirname(os.path.abspath(__file__))
_PULSES_INIT_DIR  = os.path.normpath(os.path.join(_base_dir, '..', 'device', 'pulses'))
_CONFIGS_INIT_DIR = os.path.normpath(os.path.join(_base_dir, '..', 'device', 'configs'))
_FIGS_INIT_DIR    = os.getcwd()

def get_pulses_dir():
    return _PULSES_INIT_DIR
def set_pulses_dir(pulses_dir):
    global _PULSES_INIT_DIR
    _PULSES_INIT_DIR = pulses_dir
def get_configs_dir():
    return _CONFIGS_INIT_DIR
def set_configs_dir(configs_dir):
    global _CONFIGS_INIT_DIR
    _CONFIGS_INIT_DIR = configs_dir
def get_figs_dir():
    return _FIGS_INIT_DIR
def set_figs_dir(figs_dir):
    global _FIGS_INIT_DIR
    _FIGS_INIT_DIR = figs_dir



class LoadGUI(QWidget):
    """
    Automatically opens a native file dialog once shown.
    Emits `fileSelected` when a choice is made, then closes its window.
    """
    fileSelected = pyqtSignal(str)
    
    def __init__(self, on_file_selected=None, parent=None):
        super().__init__(parent)
        # Connect user callback if provided
        if on_file_selected:
            self.fileSelected.connect(on_file_selected)
        # Use a zero-timeout timer to open dialog after the widget is shown
        QTimer.singleShot(0, self._open_dialog)

    def _open_dialog(self):
        # Show native dialog
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Data File",
            get_figs_dir(),
            "Data Files (*.npz *.jpg);;All Files (*)"
        )
        if file_path:
            # Update default dir
            set_figs_dir(os.path.dirname(file_path))
            self.fileSelected.emit(file_path)
        self.close()


class VisaAndDevicesPanel(QWidget):
    """
    Two-tab utility panel (manual-refresh version) with non-blocking VISA scan:
      - VISA tab:
          * Manual Scan runs in a worker thread (UI stays responsive).
          * Rows stream in live as devices are found.
          * Stop button requests cancellation; it takes effect at the next I/O timeout.
      - Devices tab:
          * Manual refresh only, lists DeviceManager instances.
          * Actions: Refresh / Open GUI / Close Selected / Close All.
    NOTE:
      * Requires QtCore (QThread/QObject/signals) to be imported in the module scope.
      * Keeps original object names used elsewhere (tables/buttons/tabs), only adds `btn_stop`.
      * Assumes your custom QFrame signature (e.g., QFrame(round=('SW','SE','NE'))) still exists.
    """

    # -------------------- Background scan worker --------------------
    class _VisaScanWorker(QtCore.QObject):
        """
        Worker that enumerates VISA resources and NI-DAQmx devices in a background thread.
        Emits one signal per discovered row so the UI can update incrementally.
        """
        rowFound = QtCore.pyqtSignal(str, str)  # (left, right) -> "Resource", "IDN / Note"
        started = QtCore.pyqtSignal()
        finished = QtCore.pyqtSignal()

        def __init__(self, have_pyvisa: bool, timeout_ms: int = 300, include_nidaqmx: bool = True, parent=None):
            super().__init__(parent)
            self._stop = False
            self._have_pyvisa = have_pyvisa
            self._timeout_ms = int(timeout_ms)
            self._include_nidaqmx = include_nidaqmx

        @QtCore.pyqtSlot()
        def run(self):
            """Main worker entry: enumerate VISA then NI-DAQmx; stream rows to the UI."""
            self.started.emit()
            try:
                # --- VISA ---
                if not self._have_pyvisa:
                    self.rowFound.emit("<pyvisa not installed>", "pip install pyvisa")
                else:
                    try:
                        import pyvisa
                        rm = pyvisa.ResourceManager()
                        try:
                            resources = rm.list_resources()
                        except Exception as e:
                            self.rowFound.emit("<list_resources failed>", str(e))
                            resources = ()

                        if not resources:
                            self.rowFound.emit("<no resources>", "")
                        else:
                            for r in resources:
                                if self._stop:
                                    break
                                note = self._visa_idn(r, rm)
                                self.rowFound.emit(f"{r}", note)
                    except Exception as e:
                        # ResourceManager creation or backend issues
                        self.rowFound.emit("<ResourceManager failed>", str(e))

                if self._stop:
                    return

                # --- NI-DAQmx ---
                if self._include_nidaqmx:
                    try:
                        from nidaqmx.system import System
                        devices = list(System.local().devices)
                        if not devices:
                            self.rowFound.emit("<no devices>", "")
                        else:
                            for dev in devices:
                                if self._stop:
                                    break
                                product = getattr(dev, "product_type", "?")
                                serial  = getattr(dev, "serial_number", "?")
                                self.rowFound.emit(f"{dev.name}", f"{product} SN={serial}")
                    except Exception as e:
                        self.rowFound.emit("<unavailable>", str(e))
            finally:
                self.finished.emit()

            # Done

        def request_stop(self):
            """Ask the worker to stop ASAP; takes effect after current I/O returns/timeout."""
            self._stop = True

        # ---- Internal helper for *IDN? with short timeout ----
        def _visa_idn(self, resource, rm):
            try:
                inst = rm.open_resource(resource)
                try:
                    # pyvisa timeout is in milliseconds
                    inst.timeout = self._timeout_ms
                    try:
                        return inst.query("*IDN?").strip()
                    except Exception:
                        return "<no *IDN?>"
                finally:
                    try:
                        inst.close()
                    except Exception:
                        pass
            except Exception:
                return "<open failed>"

    # -------------------- Panel implementation --------------------
    def __init__(self, dm_getter, parent=None):
        super().__init__(parent)

        # Callable that returns the current DeviceManager
        self._get_devices = dm_getter

        # pyvisa state (created on first manual scan)
        self._rm = None
        self._have_pyvisa = None

        # Scan thread state
        self._scan_thread = None
        self._scan_worker = None

        self._build_ui()
        self._wire()

        # Initial fill for Devices tab (manual model)
        self.refresh_devices()

    # ------- UI (flat, minimal nesting) -------
    def _build_ui(self):
        main = QVBoxLayout(self)
        main.setContentsMargins(0, 0, 0, 0)
        main.setSpacing(8)

        # Keep original object name
        self.tabs = QTabWidget(self)
        main.addWidget(self.tabs)

        # ---------- TAB 2: Devices (manual refresh) ----------
        dev_page = QWidget()
        dvbox = QVBoxLayout(dev_page); dvbox.setContentsMargins(0, 0, 0, 0); dvbox.setSpacing(8)

        # Frame wrapping the DM table
        f_dm_tbl = QFrame(round=('SW', 'SE', 'NE'))
        f_dtbl = QVBoxLayout(f_dm_tbl); f_dtbl.setContentsMargins(8, 8, 8, 8); f_dtbl.setSpacing(8)

        # Keep original table name
        self.tbl_dm = QTableWidget(0, 2, f_dm_tbl)
        self.tbl_dm.setHorizontalHeaderLabels(["Name", "Class"])
        self.tbl_dm.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.tbl_dm.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.tbl_dm.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.tbl_dm.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.tbl_dm.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.tbl_dm.verticalHeader().setVisible(False)
        f_dtbl.addWidget(self.tbl_dm)

        dvbox.addWidget(f_dm_tbl)

        # Frame wrapping the action buttons
        f_dm_btns = QFrame()
        row_btn = QHBoxLayout(f_dm_btns); row_btn.setContentsMargins(8, 8, 8, 8); row_btn.setSpacing(8)

        # New manual Refresh button (keep name)
        self.btn_refresh = QPushButton("Refresh")
        self.btn_refresh.setFixedHeight(30)
        row_btn.addWidget(self.btn_refresh)

        # Keep original button names
        self.btn_open = QPushButton("Open GUI")
        self.btn_open.setFixedHeight(30)
        self.btn_close = QPushButton("Close Selected")
        self.btn_close.setFixedHeight(30)
        self.btn_close_all = QPushButton("Close All")
        self.btn_close_all.setFixedHeight(30)
        row_btn.addWidget(self.btn_open)
        row_btn.addWidget(self.btn_close)
        row_btn.addWidget(self.btn_close_all)
        row_btn.addStretch(1)
        self.btn_close.set_color(FLUENT_RED)
        self.btn_close_all.set_color(FLUENT_RED)

        dvbox.addWidget(f_dm_btns)
        self.tabs.addTab(dev_page, "Devices")

        # ---------- TAB 1: VISA (manual scan with Stop) ----------
        visa_page = QWidget()
        vbox = QVBoxLayout(visa_page); vbox.setContentsMargins(0, 0, 0, 0)
        vbox.setSpacing(8)

        # Frame wrapping the VISA table
        f_visa_tbl = QFrame(round=('SW', 'SE', 'NE'))
        f_vtbl = QVBoxLayout(f_visa_tbl)
        f_vtbl.setContentsMargins(8, 8, 8, 8)
        f_vtbl.setSpacing(8)

        # Keep original table name
        self.tbl_visa = QTableWidget(0, 2, f_visa_tbl)
        self.tbl_visa.setHorizontalHeaderLabels(["Resource", "IDN / Note"])
        self.tbl_visa.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.tbl_visa.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.tbl_visa.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.tbl_visa.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.tbl_visa.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.tbl_visa.verticalHeader().setVisible(False)
        f_vtbl.addWidget(self.tbl_visa)
        vbox.addWidget(f_visa_tbl)

        # Scan/Stop row
        f_btn = QFrame()
        row = QHBoxLayout(f_btn)
        row.setContentsMargins(8, 8, 8, 8)
        row.setSpacing(8)
        self.btn_scan = QPushButton("Scan")   # manual VISA refresh (non-blocking)
        self.btn_scan.setFixedSize(100, 30)
        row.addWidget(self.btn_scan)
        self.btn_stop = QPushButton("Stop")   # request cancellation
        self.btn_stop.setFixedSize(100, 30)
        self.btn_stop.setEnabled(False)
        row.addWidget(self.btn_stop)
        row.addStretch(1)
        vbox.addWidget(f_btn)

        self.tabs.addTab(visa_page, "VISA")

    def _wire(self):
        # VISA (non-blocking + cancellable)
        self.btn_scan.clicked.connect(self.scan_visa)
        self.btn_stop.clicked.connect(self.stop_scan)

        # Devices actions (manual refresh model)
        self.btn_refresh.clicked.connect(self.refresh_devices)
        self.btn_open.clicked.connect(self._open_selected_gui)
        self.btn_close.clicked.connect(self._close_selected)
        self.btn_close_all.clicked.connect(self._close_all)

    # ------- helpers -------
    def _current_dm(self):
        """Return DeviceManager from the provided getter."""
        return self._get_devices()


    @staticmethod
    def _clear_table(tbl: QTableWidget):
        tbl.setRowCount(0)

    @staticmethod
    def _add_row(tbl: QTableWidget, *cells):
        row = tbl.rowCount()
        tbl.insertRow(row)
        for col, text in enumerate(cells):
            it = QTableWidgetItem(str(text))
            it.setFlags(QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled)
            tbl.setItem(row, col, it)

    # ------- VISA (non-blocking scan) -------
    def _ensure_pyvisa(self):
        """
        One-time check for pyvisa presence.
        We don't reuse self._rm for scanning anymore; the worker creates its own ResourceManager.
        """
        if self._have_pyvisa is not None:
            return
        try:
            import pyvisa  # noqa: F401
            self._rm = None   # no longer used, kept for backward compatibility
            self._have_pyvisa = True
        except Exception:
            self._rm = None
            self._have_pyvisa = False

    def scan_visa(self):
        """
        Start a background scan for VISA and NI-DAQmx.
        UI stays responsive; rows stream into the table as they are discovered.
        """
        if self._scan_thread is not None:
            return  # Already scanning

        # Clear table and update button states
        self._clear_table(self.tbl_visa)
        self.btn_scan.setEnabled(False)
        self.btn_stop.setEnabled(True)

        # Check pyvisa availability up front
        self._ensure_pyvisa()
        have_pyvisa = bool(self._have_pyvisa)

        # Create thread + worker
        self._scan_thread = QtCore.QThread(self)
        self._scan_worker = self._VisaScanWorker(have_pyvisa=have_pyvisa, timeout_ms=300, include_nidaqmx=True)
        self._scan_worker.moveToThread(self._scan_thread)

        # Wire signals
        self._scan_thread.started.connect(self._scan_worker.run)
        self._scan_worker.rowFound.connect(lambda left, right: self._add_row(self.tbl_visa, left, right))
        self._scan_worker.finished.connect(self._on_scan_finished)

        # Start scanning
        self._scan_thread.start()

    def stop_scan(self):
        """
        Request to stop scanning. Takes effect after current I/O returns or times out.
        """
        if self._scan_worker is not None:
            self._scan_worker.request_stop()

    def _on_scan_finished(self):
        """
        Clean up thread/worker and restore UI buttons after scan completes or is stopped.
        """
        try:
            if self._scan_thread is not None:
                self._scan_thread.quit()
                self._scan_thread.wait()
        finally:
            self._scan_thread = None
            self._scan_worker = None
            # Restore UI
            self.btn_scan.setEnabled(True)
            self.btn_stop.setEnabled(False)

    # ------- Devices (manual refresh only) -------
    def refresh_devices(self):
        """Manual refresh for the DeviceManager instances list."""
        dm = self._current_dm()
        self._clear_table(self.tbl_dm)

        # Prefer dm.list_instances() -> {name: (class_name, instance)}
        mapping = {}
        if hasattr(dm, "list_instances") and callable(getattr(dm, "list_instances", None)):
            try:
                mapping = dm.list_instances()
            except Exception:
                mapping = {}

        if not mapping:
            self._add_row(self.tbl_dm, "<no instances>" if dm else "<not initialized>", "")
            return

        for name, value in mapping.items():
            # Backward-compat: accept either (clsname, inst) or a simple instance with __class__.__name__
            try:
                clsname, _inst = value
            except Exception:
                inst = value
                clsname = getattr(getattr(inst, "__class__", None), "__name__", "?")
            self._add_row(self.tbl_dm, name, clsname)

    def _selected_name(self):
        row = self.tbl_dm.currentRow()
        if row < 0:
            return None
        it = self.tbl_dm.item(row, 0)
        return it.text() if it else None

    def _open_selected_gui(self):
        name = self._selected_name()
        if not name:
            return
        dm = self._current_dm()
        inst = getattr(dm, name, None)  # DeviceManager.__getattr__ returns instance
        if inst is not None and hasattr(inst, "gui"):
            # Call device's GUI; assume device handles its own errors
            try:
                inst.gui(in_GUI=True)
            except Exception:
                pass

    def _close_selected(self):
        name = self._selected_name()
        if not name:
            return
        dm = self._current_dm()
        dm.close_selected(name)
        self.refresh_devices()

    def _close_all(self):
        dm = self._current_dm()
        dm.close_all()
        self.refresh_devices()

    # ------- Lifecycle -------
    def closeEvent(self, event):
        """
        Ensure any active scan is stopped and the thread is joined before the widget closes.
        """
        try:
            if self._scan_worker is not None:
                self._scan_worker.request_stop()
            if self._scan_thread is not None:
                self._scan_thread.quit()
                self._scan_thread.wait()
        finally:
            self._scan_worker = None
            self._scan_thread = None
        super().closeEvent(event)

    def hideEvent(self, event):
        win = self.window()
        if getattr(win, '_hiding_via_close', False):
            try:
                if self._scan_worker is not None:
                    self._scan_worker.request_stop()
                if self._scan_thread is not None:
                    self._scan_thread.quit()
                    self._scan_thread.wait()
            finally:
                self._scan_worker = None
                self._scan_thread = None
        super().hideEvent(event)


class DeviceManagerGUI(QWidget):
    """
    PulseGUI-like shell:
      - Top bar: status dot + title (save/load state shown by StateUIManager)
      - Left: device config table (Select/Name/Type/Params)
      - Right: a single tab embedding VisaAndDevicesPanel
      - Bottom: +/-, Load, Save, Apply & Load, Reload, Close All

    No instance polling/check logic here. The right-side panel manages its own state.
    """

    def __init__(self, config=None, lookup_dict=None, action=None,
                 parent=None):
        super().__init__(parent)

        # Bind DM getter exactly once (callable -> DeviceManager)
        from Confocal_GUIv2.device import get_devices, BaseDevice, RemoteDevice
        self._get_dm = lambda: get_devices()


        self._initial_config = config or {}
        self.lookup = lookup_dict or {}
        self.device_types = sorted([
            name for name, cls in self.lookup.items()
            if inspect.isclass(cls) and (issubclass(cls, BaseDevice) or issubclass(cls, RemoteDevice))
        ])
        self.action = action

        # file/run snapshots
        self._address_str = None
        self._last_apply_state = None
        self._last_save_state = None
        self._last_load_state = None

        self._build_ui()
        self._load_config(self._initial_config)

        # file/run state manager
        self.stateui_manager = StateUIManager(
            status_dots=[self.status_dot],
            label_names=[self.label_title],
            btn_ons=[self.btn_apply],
            btn_saves=[self.btn_save],
            address_getter=lambda: self._address_str
        )
        self.stateui_manager.runstate  = StateUIManager.RunState.INIT
        self.stateui_manager.filestate = StateUIManager.FileState.UNTITLED

        # detect unsaved/unsynced changes
        self._poll_timer = QTimer(self)
        self._poll_timer.setInterval(200)
        self._poll_timer.timeout.connect(self._check_changed)
        self._poll_timer.start()

    # ----- UI -----
    def _build_ui(self):
        main = QVBoxLayout(self)
        main.setContentsMargins(18, 8, 18, 8)
        main.setSpacing(6)

        # top bar: status dot + title
        header = QFrame()
        h = QHBoxLayout(header)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(8)
        self.status_dot = QLabel()
        self.status_dot.setFixedSize(16, 16)
        self.label_title = QLabel('')
        self.label_title.setFixedHeight(30)
        self.label_title.setAlignment(Qt.AlignCenter)
        h.addStretch(1)
        h.addWidget(self.status_dot)
        h.addSpacing(4)
        h.addWidget(self.label_title)
        h.addStretch(1)
        main.addWidget(header)

        # body: left (config) + right (tools)
        body = QHBoxLayout()
        body.setSpacing(8)

        # left side: config table
        left = QVBoxLayout()
        left.setSpacing(8)

        content_frame = QFrame()
        content_vbox = QVBoxLayout(content_frame)
        # header row
        hh = QHBoxLayout(); hh.setContentsMargins(0, 0, 0, 0); hh.setSpacing(0)
        for text, width in ("Select", 60), ("Name", 150), ("Type", 200), ("Params", 400):
            lbl = QLabel(text); lbl.setFixedSize(width, 30); lbl.setAlignment(Qt.AlignCenter)
            hh.addWidget(lbl)
        content_vbox.addLayout(hh)

        # rows container
        self.row_layout = QVBoxLayout()
        self.row_layout.setSpacing(6)

        content_vbox.addLayout(self.row_layout)
        left.addWidget(content_frame)

        # bottom buttons (left side)
        btn_frame = QFrame()
        btn_bar = QHBoxLayout(btn_frame)
        btn_bar.setContentsMargins(8,8,8,8)
        btn_bar.setSpacing(8)
        self.btn_add = QPushButton("+ Add Device"); self.btn_add.clicked.connect(self._add_row)
        self.btn_del = QPushButton("- Remove Selected"); self.btn_del.clicked.connect(self._remove_selected)
        self.btn_load = QPushButton("Load config from file")
        self.btn_save = QPushButton("Save config to file*")
        self.btn_apply = QPushButton("Init devices")
        self.btn_load.clicked.connect(self._load_file)
        self.btn_save.clicked.connect(self._save_file)
        self.btn_apply.clicked.connect(self._on_apply)

        self.btn_apply.set_color(FLUENT_GREEN)
        self.btn_save.set_color(FLUENT_YELLOW)
        self.btn_load.set_color(FLUENT_ORANGE)

        for b in (self.btn_add, self.btn_del):
            b.setFixedHeight(30)
            btn_bar.addWidget(b)
        btn_bar.addStretch(1)
        for b in (self.btn_load, self.btn_save, self.btn_apply):
            b.setFixedHeight(30)
            btn_bar.addWidget(b)

        left.addStretch()
        left.addWidget(btn_frame)

        # right side: tools => just one tab embedding VisaAndDevicesPanel
        right = QVBoxLayout()
        right.setContentsMargins(0,0,0,0)
        self.panel = VisaAndDevicesPanel(dm_getter=self._get_dm)
        self.panel.setMinimumWidth(800)
        # more space to show IDN results
        right.addWidget(self.panel)

        body.addLayout(left)
        body.addLayout(right)
        main.addLayout(body)
        self.setLayout(main)

    # ----- row management -----
    def _create_row(self, name='', dtype='', params_text=''):
        row = QWidget()
        h = QHBoxLayout(row); h.setContentsMargins(0, 0, 0, 0); h.setSpacing(10)

        cb = QCheckBox(); cb.setFixedSize(60, 30); h.addWidget(cb)

        le_name = QLineEdit(name); le_name.setFixedSize(150, 30); h.addWidget(le_name)

        combo = QComboBox(); combo.addItems(self.device_types)
        if dtype: combo.setCurrentText(dtype)

        combo.setFixedSize(200, 30); h.addWidget(combo)
        placeholder_text = 'e.g., piezo_lb:50 or laser_handle:$device:laser'
        te_params = DynamicPlainTextEdit(text = params_text, placeholder_text=placeholder_text)
        te_params.setFixedWidth(400); h.addWidget(te_params)

        self.row_layout.addWidget(row)
        row.widgets = (cb, le_name, combo, te_params)
        combo.currentTextChanged.connect(lambda _, r=row: self._on_type_changed(r))
        return row

    def _add_row(self):
        row = self._create_row()
        self._on_type_changed(row)

    def _remove_selected(self):
        for i in reversed(range(self.row_layout.count())):
            w = self.row_layout.itemAt(i).widget()
            if w.widgets[0].isChecked():
                w.setParent(None)

    # ----- state tracking -----

    def _device_types_now(self):
        """Build fresh device_types from the live lookup dict."""
        from Confocal_GUIv2.device import BaseDevice, RemoteDevice
        return sorted([
            name for name, cls in self.lookup.items()
            if inspect.isclass(cls) and (issubclass(cls, BaseDevice) or issubclass(cls, RemoteDevice))
        ])

    def _refresh_all_type_combos(self):
        """Rebuild all 'Type' comboboxes; keep current text; mark unknown in red."""
        self.device_types = self._device_types_now()
        for i in range(self.row_layout.count()):
            row = self.row_layout.itemAt(i).widget()
            if not hasattr(row, "widgets"):  # safety
                continue
            _cb, _le_name, combo, _te_params = row.widgets
            cur = combo.currentText().strip()

            combo.blockSignals(True)
            combo.clear()
            combo.addItems(self.device_types)
            combo.setCurrentText(cur)  # preserve what user/config had
            combo.blockSignals(False)

    def _get_gui_state(self):
        """Robust, low-strictness snapshot of current UI config.
        - Never raises: all errors are swallowed and the row is skipped.
        - Accepts only 'key: value' lines; other lines are ignored.
        - Falls back to raw string if str2python() fails.
        """
        cfg = {}
        rl = getattr(self, "row_layout", None)
        if not rl:
            return cfg

        for i in range(rl.count()):
            try:
                w = rl.itemAt(i).widget()
                cb, le_name, combo, te_params = getattr(w, "widgets", (None, None, None, None))

                # Name/type with safe fallbacks
                name = ((le_name.text() if le_name else "") or "").strip() or f"unnamed_{i}"
                dtype = ((combo.currentText() if combo else "") or "").strip()

                # Parse params: only lines like "key: value"
                text = (te_params.toPlainText() if te_params else "") or ""
                params = {}
                for raw in text.splitlines():
                    s = raw.strip()
                    if not s or ":" not in s:
                        continue
                    k, v = s.split(":", 1)
                    k = k.strip()
                    if not k:
                        continue
                    v = v.strip()
                    try:
                        params[k] = str2python(v)
                    except Exception:
                        params[k] = v  # fallback to raw string

                cfg[name] = {"type": dtype, "params": params}

            except Exception:
                # Swallow any row-level error and continue
                continue

        return cfg

    def _current_dm(self):
        """Return DeviceManager from the provided getter."""
        try:
            return self._get_dm()
        except Exception:
            return None

    def _check_changed(self):
        cur_cfg = self._get_gui_state()
        dm = self._current_dm()

        if dm is not None:
            last_on_state = dm.loaded_config
            if (last_on_state != cur_cfg) or (set(dm._instances) != set(cur_cfg)):
                if self.stateui_manager.runstate == StateUIManager.RunState.RUNNING:
                    self.stateui_manager.runstate = StateUIManager.RunState.UNSYNCED
            else:
                if self.stateui_manager.runstate == StateUIManager.RunState.UNSYNCED:
                    self.stateui_manager.runstate = StateUIManager.RunState.RUNNING

        if self._last_save_state is not None:
            self.stateui_manager.filestate = (
                StateUIManager.FileState.SAVE if self._last_save_state == cur_cfg else StateUIManager.FileState.UNSAVED
            )
        if self._last_load_state is not None:
            self.stateui_manager.filestate = (
                StateUIManager.FileState.LOAD if self._last_load_state == cur_cfg else StateUIManager.FileState.UNSAVED
            )

    def _fmt_for_editor(self, v):
        DEVICE_PREFIX = "$device:"
        if isinstance(v, str) and v.startswith(DEVICE_PREFIX):
            return v
        return repr(v)

    # ----- config I/O -----
    def _load_config(self, cfg):
        # clear rows
        for i in reversed(range(self.row_layout.count())):
            self.row_layout.itemAt(i).widget().setParent(None)

        # load rows
        for name, entry in cfg.items():
            row = self._create_row(name, entry.get('type', ''), '')
            self._on_type_changed(row)
            te = row.widgets[3]
            params = entry.get('params', None)
            if params is not None:
                lines = [f"{k}: {self._fmt_for_editor(v)}" for k, v in params.items()]
                te.setPlainText("\n".join(lines))
                te._adjust_height()

    def _on_type_changed(self, row):
        type_name = row.widgets[2].currentText()
        te_params = row.widgets[3]
        cls = self.lookup.get(type_name)
        lines = []
        if cls:
            sig = inspect.signature(cls.__init__)
            for pname, param in sig.parameters.items():
                if pname == 'self':
                    continue
                if pname.endswith('_handle'):
                    lines.append(f"{pname}: $device:{pname[:-7]}")
                else:
                    default = param.default if param.default is not inspect._empty else None
                    lines.append(f"{pname}: {repr(default)}")
        te_params.setPlainText("\n".join(lines))

    def _gather_config(self):
        cfg = {}
        for i in range(self.row_layout.count()):
            w = self.row_layout.itemAt(i).widget()
            cb, le_name, combo, te_params = w.widgets

            name = le_name.text().strip()
            if not name:
                QMessageBox.warning(self, 'Error', f'Row {i+1}: name empty')
                return None

            dtype = combo.currentText()
            text = te_params.toPlainText().strip()
            d = {}
            if text:
                for lineno, line in enumerate(text.splitlines(), 1):
                    if not line.strip():
                        continue
                    if ':' not in line:
                        QMessageBox.warning(self, 'Error', f'Row {i+1}, line {lineno}: format should be key: value')
                        return None
                    key, val = line.split(':', 1)
                    key = key.strip()
                    d[key] = str2python(val)
            cfg[name] = {'type': dtype, 'params': d}
        return cfg

    def _load_file(self):
        path, _ = QFileDialog.getOpenFileName(self, 'Open Config', get_configs_dir(), 'JSON Files (*.json)')
        if path:
            with open(path, 'r') as f:
                cfg = json.load(f)
            self._load_config(cfg)
            self._address_str = path
            self._last_save_state = None
            self._last_load_state = self._get_gui_state()
            self.stateui_manager.filestate = self.stateui_manager.FileState.LOAD

    def _save_file(self):
        cfg = self._gather_config()
        if cfg is None: return
        path, _ = QFileDialog.getSaveFileName(self, 'Save Config', get_configs_dir(), 'JSON Files (*.json)')
        if path:
            with open(path, 'w') as f:
                json.dump(cfg, f, indent=2)
            self._address_str = path
            self._last_load_state = None
            self._last_save_state = self._get_gui_state()
            self.stateui_manager.filestate = self.stateui_manager.FileState.SAVE

    def _on_apply(self):
        cfg = self._gather_config()
        if cfg is None:
            return
        if self.action:
            try:
                self.action(config=cfg, lookup_dict=self.lookup)
                self.stateui_manager.runstate = self.stateui_manager.RunState.RUNNING
                # Let the right panel refresh its DM list after apply.
                self.panel.refresh_devices()
            except Exception as e:
                QMessageBox.warning(self, "Load devices", f"Failed: {e}")

    def hideEvent(self, event):
        win = self.window()
        if getattr(win, '_hiding_via_close', False):
            if self._poll_timer.isActive():
                self._poll_timer.stop()
        super().hideEvent(event)

    def showEvent(self, event):
        self._refresh_all_type_combos()
        if not self._poll_timer.isActive():
            self._poll_timer.start()

        super().showEvent(event)


class DeviceGUI(QWidget):
    def __init__(self, device, parent=None):
        super().__init__(parent)
        self.device = device
        self._active_refresh_prop_list = []
        self._refresh_interval = 200 #ms
        self.MAX_LEN = 20
        # max len for display
        self.init_ui()

    def init_func_ui(self, prop):
        sig = inspect.signature(getattr(self.device, prop))
        group = QGroupBox(f"{prop}{sig}")
        layout = QVBoxLayout(group)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        # Argument input
        arg_box = QHBoxLayout()
        arg_label = QLabel("Arg:")
        arg_label.setFixedSize(100, 30)
        arg_box.addWidget(arg_label, alignment=Qt.AlignLeft)
        arg_edit = QLineEdit()
        arg_edit.setPlaceholderText("e.g. (1, 2) or (param1=1, param2=2)")
        arg_edit.setFixedSize(300, 30)
        arg_box.addWidget(arg_edit)
        layout.addLayout(arg_box)

        # Call + result
        call_box = QHBoxLayout()
        call_btn = QPushButton("Apply")
        call_btn.setFixedSize(100, 30)
        result_edit = QTextEdit()
        result_edit.setReadOnly(True)
        result_edit.setFixedSize(300, 150)
        call_box.addWidget(call_btn)
        call_box.addWidget(result_edit)
        layout.addLayout(call_box)

        def make_caller(p, edit, output):
            def _caller():
                text = edit.text().strip()
                if not (text.startswith('(') and text.endswith(')')):
                    text = f'({text})'
                expr = f"self.device.{p}{text}"
                buf = io.StringIO()
                old = sys.stdout
                sys.stdout = buf
                try:
                    res = eval(expr)
                except Exception as e:
                    sys.stdout = old
                    QMessageBox.warning(self, "Call Error", f"Failed: {e}")
                    return
                finally:
                    sys.stdout = old
                out = buf.getvalue() or ''
                if res is not None:
                    out += f"\nReturn: {res}"
                output.setPlainText(out.strip())
            return _caller

        call_btn.clicked.connect(make_caller(prop, arg_edit, result_edit))
        group.adjustSize()
        group.setFixedSize(group.width(), group.height())
        return group

    def _make_prop_group(self, prop, meta):
        lb = getattr(self.device, f"{prop}_lb", None)
        ub = getattr(self.device, f"{prop}_ub", None)

        inspect_cls = getattr(self.device, '_device_cls', self.device.__class__)
        attr = inspect.getattr_static(inspect_cls, prop, None)
        has_setter = False
        if isinstance(attr, property):
            has_setter = attr.fset is not None
        elif hasattr(attr, '__set__'):
            has_setter = getattr(attr, 'fset', None) is not None
        # may not have setter


        group = QGroupBox(prop)
        layout = QVBoxLayout(group)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        typ = meta['gui_type']
        # Bounds
        box = QHBoxLayout()
        if typ == 'float':
            lb_s = float2str_eng(lb, length=self.MAX_LEN) if lb is not None else '-∞'
            ub_s = float2str_eng(ub, length=self.MAX_LEN) if ub is not None else '∞'
            label_range = QLabel('Range:')
            label_range.setFixedSize(100, 30)
            box.addWidget(label_range, alignment=Qt.AlignLeft)
            rng = QLineEdit(f"{lb_s} to {ub_s}")
        elif typ == 'bool':
            label_range = QLabel('Valid:')
            label_range.setFixedSize(100, 30)
            box.addWidget(label_range, alignment=Qt.AlignLeft)
            preview = ", ".join(['True', 'False'])
            rng = QLineEdit(f"{preview}")
        elif typ == 'str':
            label_range = QLabel('Valid:')
            label_range.setFixedSize(100, 30)
            box.addWidget(label_range, alignment=Qt.AlignLeft)
            valid = getattr(self.device, f"{prop}_valid", [])
            preview = ", ".join(valid)
            rng = QLineEdit(f"{preview}")

        rng.setEnabled(False)
        rng.setFixedSize(300, 30)
        box.addWidget(rng)
        layout.addLayout(box)

        # Current
        box = QHBoxLayout()
        label_current = QLabel('Current:')
        label_current.setFixedSize(100, 30)
        box.addWidget(label_current, alignment=Qt.AlignLeft)
        current = QLineEdit()
        current.setEnabled(False)
        current.setFixedSize(300, 30)
        box.addWidget(current)
        layout.addLayout(box)

        # Stack for input modes
        stack = QStackedWidget()
        stack.setStyleSheet("background: transparent; border: none;")
        stack.setGraphicsEffect(None)
        # remove Fluentstackedwidget default background and effect

        # Page 0: plain text input
        w0 = QWidget()
        v0 = QVBoxLayout(w0)
        v0.setContentsMargins(0, 0, 0, 0)
        h0 = QHBoxLayout()
        h0.addWidget(QLabel('Set:'), alignment=Qt.AlignLeft)
        editor = None
        if typ == 'float':
            editor = FloatLineEdit()
            editor.setMaxLength(self.MAX_LEN)
            editor.lb = lb
            editor.ub = ub
            editor.setFixedHeight(30)
        elif typ == 'bool':
            editor = QComboBox(); editor.addItems(['True','False'])
        elif typ == 'str':
            editor = QComboBox(); editor.addItems(getattr(self.device, f"{prop}_valid", []))

        editor.setFixedWidth(300)
        editor.setEnabled(has_setter)
        h0.addWidget(editor)
        v0.addLayout(h0)
        btn_apply = QPushButton('Apply')
        btn_apply.setFixedHeight(30)
        btn_apply.setFixedWidth(300)
        btn_apply.setEnabled(has_setter)
        btn_apply.clicked.connect(lambda _, p=prop, e=editor: self.apply_value(p,e))
        btn_sw = QPushButton('Switch')
        btn_sw.setFixedHeight(30)
        btn_sw.setFixedWidth(100)
        btn_sw.clicked.connect(lambda _, s=stack: s.setCurrentIndex(1))
        hb = QHBoxLayout()
        hb.addWidget(btn_sw)
        hb.addWidget(btn_apply)
        v0.addLayout(hb)
        stack.addWidget(w0)

        # Page 1: interactive spin/combobox
        w1 = QWidget()
        v1 = QVBoxLayout(w1)
        v1.setContentsMargins(0, 0, 0, 0)
        h1 = QHBoxLayout()
        h1.addWidget(QLabel('Set:'), alignment=Qt.AlignLeft)
        if typ == 'float':
            spin = QDoubleSpinBox(length=self.MAX_LEN, allow_minus=True)
            spin.setFixedHeight(30)
            if lb is not None: spin.setMinimum(lb)
            if ub is not None: spin.setMaximum(ub)
            spin.valueChanged.connect(lambda v, p=prop, s=spin: self._on_spin(p,s))
        elif typ == 'bool':
            spin = QComboBox(); spin.addItems(['True','False'])
            spin.currentTextChanged.connect(lambda v, p=prop, s=spin: self._on_spin(p,s))
        else:
            spin = QComboBox(); spin.addItems(getattr(self.device, f"{prop}_valid", []))
            spin.currentTextChanged.connect(lambda v, p=prop, s=spin: self._on_spin(p,s))
        spin.setFixedWidth(300)
        spin.setEnabled(has_setter)
        h1.addWidget(spin)
        v1.addLayout(h1)
        cb = QCheckBox('Bind'); cb.setFixedSize(300, 30)
        cb.toggled.connect(lambda checked, p=prop, s=spin: self._on_spin(p,s))
        cb.setEnabled(has_setter)
        sw2 = QPushButton('Switch'); sw2.setFixedSize(100, 30)
        sw2.clicked.connect(lambda _, s=stack: s.setCurrentIndex(0))
        hb2 = QHBoxLayout(); hb2.addWidget(sw2); hb2.addWidget(cb)
        v1.addLayout(hb2)
        stack.addWidget(w1)

        layout.addWidget(stack)
        setattr(self, f'_current_{prop}', current)
        setattr(self, f'_editor_{prop}', editor)
        setattr(self, f'_spin_{prop}', spin)
        setattr(self, f'_bind_{prop}', cb)

        group.adjustSize()
        group.setFixedSize(group.width(), group.height())
        self._active_refresh_prop_list.append(prop)
        return group

    def init_ui(self):
        # root vertical layout
        main = QVBoxLayout(self)
        main.setContentsMargins(18,8,18,8)
        main.setSpacing(6)
        # title
        frame = QFrame()
        frame_layout = QHBoxLayout(frame)
        frame_layout.setSpacing(0)
        title = QLabel(f"{self.device.__class__.__module__}.{self.device.__class__.__name__}")
        title.setAlignment(Qt.AlignCenter)
        frame_layout.addWidget(title)
        main.addWidget(frame)

        # horizontal container for columns
        hbox = QHBoxLayout()
        hbox.setSpacing(6)

        num_per_col = 3
        items = list(self.device.gui_dict.items())
        total = len(items)
        ncols = int(np.ceil(total / num_per_col)) if total else 1

        # create column layouts
        cols = []
        for _ in range(ncols):
            col = QVBoxLayout()
            col.setSpacing(6)
            cols.append(col)
            hbox.addLayout(col)

        # distribute groups
        for idx, (prop, meta) in enumerate(items):
            col = cols[idx // num_per_col]
            if meta['gui_type'] == 'func':
                grp = self.init_func_ui(prop)
            else:
                grp = self._make_prop_group(prop, meta)
            col.addWidget(grp, alignment=Qt.AlignTop)

        hbox.addStretch(1)
        main.addLayout(hbox)
        main.addStretch(1)
        self.setLayout(main)

        self._refresh_timer = QTimer(self)
        self._refresh_timer.setInterval(self._refresh_interval)
        self._refresh_timer.timeout.connect(self._refresh_all_props)
        self._refresh_timer.start()

    def _on_spin(self, prop, spin):
        if getattr(self, f'_bind_{prop}').isChecked():
            typ = self.device.gui_dict[prop]['gui_type']
            val = spin.value() if typ=='float' else spin.currentText()
            if typ=='bool':
                val = (val=='True')
            setattr(self.device, prop, val)

    def apply_value(self, prop, editor):
        typ = self.device.gui_dict[prop]['gui_type']
        try:
            if typ == 'float':
                txt = editor.text()
                if not txt: raise ValueError('Input empty')
                v = float(txt)
                lb = getattr(editor, 'lb', None)
                ub = getattr(editor, 'ub', None)
                if (lb is not None and v < lb) or (ub is not None and v > ub):
                    lb_s = float2str_eng(lb, length=self.MAX_LEN) if lb is not None else '-∞'
                    ub_s = float2str_eng(ub, length=self.MAX_LEN) if ub is not None else '∞'
                    raise ValueError(f'Allowed [{lb_s} to {ub_s}]')
                new = v
            elif typ=='bool':
                new = (editor.currentText()=='True')
            else:
                new = editor.currentText()
            setattr(self.device, prop, new)
        except Exception as e:
            QMessageBox.warning(self, 'Error', str(e))

    def refresh_status(self, prop):
        val = getattr(self.device, prop)
        edit = getattr(self, f'_current_{prop}')
        edit.setText(float2str_eng(val, length=self.MAX_LEN) if isinstance(val, float) else str(val))

    def _refresh_all_props(self):
        for prop in self._active_refresh_prop_list:
            self.refresh_status(prop)

    def hideEvent(self, event):
        win = self.window()
        if getattr(win, '_hiding_via_close', False):
            if self._refresh_timer.isActive():
                self._refresh_timer.stop()
        super().hideEvent(event)

    def showEvent(self, event):
        self._refresh_all_props()
        if not self._refresh_timer.isActive():
            self._refresh_timer.start()
        super().showEvent(event)

    def closeEvent(self, event):
        super().closeEvent(event)


class StateUIManager(QObject):

    runstateChanged = pyqtSignal(object)
    filestateChanged = pyqtSignal(object)

    class RunState(Enum):
        INIT    = auto()
        RUNNING = auto()
        STOP = auto()
        UNSYNCED = auto()

    class FileState(Enum):
        UNTITLED    = auto() # for pulsegui init state
        SAVE = auto()
        LOAD = auto()
        UNSAVED = auto()

    def __init__(self, status_dots=None, label_names=None, btn_ons=None, btn_saves=None, address_getter=None):
        super().__init__()
        self.status_dots = status_dots if status_dots is not None else []
        self.label_names = label_names if label_names is not None else []
        self.btn_ons = btn_ons if btn_ons is not None else []
        self.btn_saves = btn_saves if btn_saves is not None else []
        self.address_getter = address_getter if address_getter is not None else (lambda: '')
        self._runstate = self.RunState.INIT
        self._filestate = self.FileState.UNTITLED

    @property
    def runstate(self):
        return self._runstate

    @runstate.setter
    def runstate(self, value):
        self._runstate = value
        self.runstateChanged.emit(value)
        self._update_ui()

    @property
    def filestate(self):
        return self._filestate

    @filestate.setter
    def filestate(self, value):
        self._filestate = value
        self.filestateChanged.emit(value)
        self._update_ui()

    def _compact_addr(self):
        keep = 2 # until the last two '/'
        addr = self.address_getter()
        if not addr:
            return ''
        s = str(addr).replace('\\', '/').rstrip('/*/ ')
        parts = [p for p in s.split('/') if p]
        if len(parts) <= keep:
            return '/'.join(parts)
        return '…/' + '/'.join(parts[-keep:])

    @staticmethod
    def _add_star(widget):
        text = widget.text()
        if not text or text.endswith('*'): return
        widget.setText(text + '*')

    @staticmethod
    def _remove_star(widget):
        text = widget.text()
        if text and text.endswith('*'):
            widget.setText(text[:-1])

    def _update_ui(self):

        color_map = {
            self.RunState.INIT:    FLUENT_GREY,
            self.RunState.RUNNING: FLUENT_GREEN,
            self.RunState.UNSYNCED:FLUENT_ORANGE,
            self.RunState.STOP:    FLUENT_RED,
        }

        col = color_map[self.runstate]
        if col is not None:
            for status_dot in self.status_dots:
                status_dot.setStyleSheet(f"background:{col}; border-radius:8px;")

        if self.runstate == self.RunState.RUNNING:
            for b in self.btn_ons: self._remove_star(b)
        elif self.runstate == self.RunState.STOP:
            for b in self.btn_ons: self._add_star(b)
        elif self.runstate == self.RunState.UNSYNCED:
            for b in self.btn_ons: self._add_star(b)

        addr = self._compact_addr()
        if self.filestate == self.FileState.UNTITLED:
            for lbl in self.label_names: lbl.setText('Untitled*')
            for b in self.btn_saves: 
                self._add_star(b)
                b.set_color(FLUENT_YELLOW)
        elif self.filestate == self.FileState.LOAD:
            for lbl in self.label_names: lbl.setText(f'[Load from] {addr}')
            for b in self.btn_saves: 
                self._remove_star(b)
                b.set_color(ACCENT_COLOR)
        elif self.filestate == self.FileState.SAVE:
            for lbl in self.label_names: lbl.setText(f'[Save to] {addr}')
            for b in self.btn_saves: 
                self._remove_star(b)
                b.set_color(ACCENT_COLOR)
        elif self.filestate == self.FileState.UNSAVED:
            for lbl in self.label_names: self._add_star(lbl)
            for b in self.btn_saves: 
                self._add_star(b)
                b.set_color(FLUENT_YELLOW)



class PulseGUI(QWidget):
    """
    GUI:

        Off Pulse/On Pulse:
            button to off and on pulse, using pulse sequence currently in the GUI.
            On pulse will also save current pulse sequence.

        Remove/Add Column:
            button to remove or add one more pulse column

        Save to/Load from file:
            button to save pulse to file(*pulse.npz) or load from saved pulse(*pulse.npz)

        Add/Delete Bracket:
            button to add a bracket which defines which part of pulse sequence should be repeated

        Info settings:
            if checkbox is checked, will automatically repeat current sequence another time for reference,
            where the second sequence disables signal and replace DAQ gate with DAQ_ref gate while clock will
            not be repeated for the ref pulse

        x:
            number in Channel delay or pulse duration can be a number or expression, if expression, then corresponding
            time will be replaced by pulse.x property when pulse.read_data() is called
            e.g.
                first run
                pulse.x = 100
                pulse1 duration: 1000-x -> 1000-100=900  

                second run
                pulse.x = 200
                pulse1 duration: 1000-x -> 1000-200=800

            which enables configure pulse sequence before measurement, and gives measurement an option to change pulse
            on demand by only changing pulse.x or config_instances['pulse'].x 

        Pulse:
            you can drag and insert all pulses into any new position
    """



    def __init__(self, device_handle, parent=None):
        super().__init__(parent)
        self.device_handle = device_handle
        self.t_resolution = self.device_handle.t_resolution
        self.channel_names_map = [f'Ch{ch}' for ch in range(8)]
        self.drag_container = None
        self.MAX_LEN = 16

        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(18, 8, 18, 8)
        hbox = QHBoxLayout()
        self.main_layout.addLayout(hbox)

        frame = QFrame()
        frame_layout = QHBoxLayout(frame)
        frame_layout.setContentsMargins(8, 8, 8, 8)
        frame_layout.setSpacing(0)

        self.status_dot = QLabel()
        self.status_dot.setFixedSize(16, 16)
        frame_layout.addStretch(1)
        frame_layout.addWidget(self.status_dot)
        frame_layout.setSpacing(4)

        self.label_name = QLabel('')
        self.label_name.setFixedHeight(30)
        self.label_name.setAlignment(Qt.AlignCenter)
        frame_layout.addWidget(self.label_name)
        frame_layout.addStretch(1)

        hbox.addWidget(frame)
        self.main_layout.addStretch()

        self.layout_dataset = QHBoxLayout()
        self.main_layout.addLayout(self.layout_dataset)

        self.layout_controls = QHBoxLayout()
        self.main_layout.addLayout(self.layout_controls)

        self.layout_btns = QVBoxLayout()
        row1 = QHBoxLayout()
        for text, slot, attr in [
            ('Off Pulse',     self.off_pulse,     'btn_off_pulse'),
            ('On Pulse',      self.on_pulse,      'btn_on_pulse'),
            ('Remove Column', self.remove_column, 'btn_remove_column'),
            ('Add Column',    self.add_column,    'btn_add_column'),
        ]:
            btn = QPushButton(text)
            btn.setFixedSize(150, 100)
            btn.clicked.connect(slot)
            row1.addWidget(btn)
            
            setattr(self, attr, btn)

        self.layout_btns.addLayout(row1)
        row2 = QHBoxLayout()
        # Save to file
        self.btn_savefile = QPushButton('Save to file*')
        self.btn_savefile.setFixedSize(150, 100)
        self.btn_savefile.clicked.connect(self.save_to_file)
        row2.addWidget(self.btn_savefile)
        # Load from file
        self.btn_loadfile = QPushButton('Load from file')
        self.btn_loadfile.setFixedSize(150, 100)
        self.btn_loadfile.clicked.connect(self.load_from_file)
        row2.addWidget(self.btn_loadfile)
        # Save Pulse
        self.btn_sync = QPushButton('Sync with device')
        self.btn_sync.setFixedSize(150, 100)
        self.btn_sync.clicked.connect(self.on_sync)
        row2.addWidget(self.btn_sync)
        # Add/Delete Bracket
        self.btn_add_bracket = QPushButton('Add Bracket')
        self.btn_add_bracket.setFixedSize(150, 100)
        self.btn_add_bracket.clicked.connect(self.on_add_bracket)
        row2.addWidget(self.btn_add_bracket)
        self.layout_btns.addLayout(row2)

        self.btn_sync.set_color(FLUENT_ORANGE)
        self.btn_savefile.set_color(FLUENT_YELLOW)
        self.btn_loadfile.set_color(FLUENT_ORANGE)
        self.btn_on_pulse.set_color(FLUENT_GREEN)
        self.btn_off_pulse.set_color(FLUENT_RED)

        self.layout_ref = QVBoxLayout()
        self.btn_ref_info = self.add_ref_info()
        self.layout_ref.addWidget(self.btn_ref_info)

        self.layout_controls.addLayout(self.layout_btns)
        self.layout_controls.addLayout(self.layout_ref)
        self.layout_controls.addStretch()

        self.load_data()


        self.address_str = None
        self.stateui_manager = StateUIManager(status_dots=[self.status_dot,], 
            label_names=[self.label_name,], btn_ons=[self.btn_on_pulse,], 
            btn_saves=[self.btn_savefile,], address_getter=lambda: self.address_str)
        self.last_on_state = None
        self.last_save_state = None
        self.last_load_state = None

        self.stateui_manager.runstate = self.stateui_manager.RunState.INIT
        self.stateui_manager.filestate = self.stateui_manager.FileState.UNTITLED



        self._poll_timer = QTimer(self)
        self._poll_timer.setInterval(200)
        self._poll_timer.timeout.connect(self._check_device_change)
        self._poll_timer.start()




    def off_pulse(self):


        # Create a sequence object
        self.device_handle.off_pulse()
        # need to repeat 8 times because pulse streamer will pad sequence to multiple of 8ns, otherwise unexpected changes of pulse


    def on_pulse(self):
        ok = self.save_data()
        if ok:
            self.device_handle.on_pulse()

    def auto_fixed_size(self, widget, direction='wh'):
        widget.adjustSize()
        w, h = widget.width(), widget.height()
        if direction == 'wh':
            widget.setFixedSize(w, h)
        elif direction == 'w':
            widget.setFixedWidth(w)
        elif direction == 'h':
            widget.setFixedHeight(h)

    def add_ref_info(self):
        group_box = QGroupBox('Info settings')
        group_layout = QVBoxLayout()

        self.checkbox = QCheckBox('Auto append ref')
        group_layout.addWidget(self.checkbox)

        self.comboboxes = {}
        combobox_labels = ['Signal', 'DAQ', 'DAQ_ref', 'Clock']
        for label_text in combobox_labels:
            h_layout = QHBoxLayout()
            label = QLabel(label_text)
            combobox = QComboBox()
            combobox.setFixedSize(80, 30)
            combobox.addItem('None')
            combobox.addItems([f'Ch{channel}' for channel in range(8)])
            self.comboboxes[label_text] = combobox

            h_layout.addWidget(label)
            h_layout.addWidget(combobox)
            group_layout.addLayout(h_layout)

        group_box.setLayout(group_layout)
        self.auto_fixed_size(group_box)

        return group_box

    def on_time_unit_change(self, text, line_edit):
        if text in ('str (ns)', 'ns'):
            line_edit.set_resolution(self.t_resolution)
        if text in ('us',):
            line_edit.set_resolution(self.t_resolution/100)
        if text in ('ms',):
            line_edit.set_resolution(self.t_resolution/100)
        # /100 not /1e3 or /1e6 for a better display

    def handle_text_change(self, text, combo_box):
        """
        Switch to 'str (ns)' ONLY when the expression contains 'x'.
        For pure numeric expressions (including intermediate typing like '-', '100-'),
        keep the unit combo enabled and do not flip to 'str (ns)' until we know it's an 'x' expr.
        """
        s = (text or "").strip()
        if not s:
            combo_box.setCurrentText('str (ns)')
            combo_box.setEnabled(False)
            return

        # If the user uses 'x' (anywhere) -> force string(ns)
        if 'x' in s.lower():
            if combo_box.currentText() != 'str (ns)':
                combo_box.setCurrentText('str (ns)')
            combo_box.setEnabled(False)
            return

        # Numeric-only path: never auto-switch to 'str (ns)' just because of Intermediate
        editor = self.sender()
        val = editor.validator() if editor is not None else None
        state = None
        if isinstance(val, QValidator):
            state, _, _ = val.validate(s, 0)

        if state == QValidator.Acceptable:
            # Valid pure number -> enable units; default to ns if currently 'str (ns)'
            combo_box.setEnabled(True)
            if combo_box.currentText() == 'str (ns)':
                combo_box.setCurrentText('ns')
        else:
            # Still typing (Intermediate/Invalid) -> keep units, keep enabled
            combo_box.setEnabled(True)
            # DO NOT flip to 'str (ns)' here


    def on_add_bracket(self, start_index=0, end_index=-1):
        """
        If brackets do not exist yet:
          1) Find the first pulse index and the last pulse index.
          2) Insert the start bracket right before the first pulse.
          3) Insert the end bracket right after the last pulse.
        If brackets already exist, remove them.
        """
        if not self.bracket_exists:
            # 1) Identify the indices of the first and last pulse
            first_pulse_index = 0
            last_pulse_index = len(self.drag_container.items)
            end_index = last_pulse_index if (end_index==-1) else (end_index+1)
            # 2) Create the start bracket widget
            start_box = QGroupBox("Start")
            start_box.setFixedWidth(100)
            vb1 = QVBoxLayout(start_box)
            sublayout = QVBoxLayout()
            vb1.addLayout(sublayout)
            btn = QLabel("Start\nof\nrepeat")
            sublayout.addWidget(btn)


            # 3) Create the end bracket widget
            end_box = QGroupBox("End")
            end_box.setFixedWidth(100)
            vb2 = QVBoxLayout(end_box)
            sublayout = QVBoxLayout()
            vb2.addLayout(sublayout)
            btn = QLabel("End\nof\nrepeat")
            sublayout.addWidget(btn)
            sp = QDoubleSpinBox()
            sp.setFixedHeight(30)
            sp.setDecimals(0)
            sp.setRange(1, 999)
            sp.setValue(self.repeat_info[2])
            sublayout.addWidget(sp)

            # -- Inserting the brackets --
            # NOTE: Insert the end bracket first so the index of the first pulse won't shift.
            #       We want the end bracket after the last pulse, so its index is last_pulse_index + 1.
            #self.drag_container.insert_item(last_pulse_index + 1, end_box, "bracket_end")

            # Now insert the start bracket right before the first pulse (no shift occurs yet).
            self.drag_container.insert_item(start_index, start_box, "bracket_start")
            self.drag_container.insert_item(end_index+1, end_box, "bracket_end")
            #self.drag_container.refresh_layout()

            # Toggle flag and button text
            self.bracket_exists = True
            self.btn_add_bracket.setText("Delete Bracket")

        else:
            # Remove the bracket items only, leaving pulses intact
            self.delete_bracket_only()
            self.bracket_exists = False
            self.btn_add_bracket.setText("Add Bracket")


    def delete_bracket_only(self):
        """
        Remove items of type 'bracket_start' and 'bracket_end', keeping only pulses.
        """
        new_list = []
        for it in self.drag_container.items:
            if it.item_type not in ("bracket_start", "bracket_end"):
                new_list.append(it)
        self.drag_container.items = new_list
        self.drag_container.refresh_layout()

    def on_sync(self):
        self.load_data()
        self.stateui_manager.filestate = self.stateui_manager.FileState.UNTITLED
        self.last_load_state = None 
        self.last_save_state = None

    def _best_unit_and_text(self, value_ns: int):
        """
        Choose display unit by the shortest numeric length (excluding '.').
        Special case: if value is 0, always use 'ns'.
        Tiebreaker for non-zero: prefer the largest unit (ms > us > ns).
        Returns (unit, text) with a compact decimal string (no scientific notation).
        """
        # Special case: zero -> '0' ns
        if int(value_ns) == 0:
            return ('ns', '0')

        candidates = [('ns', 1), ('us', 1e3), ('ms', 1e6)]
        # Lower is better in tie (ms first, then us, then ns)
        priority = {'ms': 0, 'us': 1, 'ns': 2}

        best = None
        best_key = None
        for unit, factor in candidates:
            v = value_ns / factor
            # Format with enough precision, then strip trailing zeros and trailing dot
            s = f"{v:.9f}".rstrip('0').rstrip('.')
            if s == '':
                s = '0'
            # Count only digits (exclude the decimal point)
            digits = sum(c.isdigit() for c in s)

            key = (digits, priority[unit])
            if best_key is None or key < best_key:
                best_key = key
                best = (unit, s)
        return best


    def load_data(self):
        """
        load self.device_handle.delay_array, self.device_handle.data_matrix back to GUI
        to recover GUI state after reopen GUI window
        """
        self.channel_names_map = [f'Ch{ch}' for ch in range(8)]
        while self.layout_dataset.count() > 0:
            item = self.layout_dataset.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()
        # remove all column is exists
        self.delay_array = self.device_handle.delay_array
        self.data_matrix = self.device_handle.data_matrix
        self.channel_names = self.device_handle.channel_names
        self.repeat_info = self.device_handle.repeat_info
        self.ref_info = self.device_handle.ref_info

        self.add_channel_names()
        for j, name_value in enumerate(self.channel_names):
            sublayout = self.names_sublayout[j]
            line_edit = sublayout.itemAt(1).widget() 
            line_edit.setText(str(name_value))

        self.add_delay()

        for j, delay_value in enumerate(self.delay_array):
            sublayout = self.delay_sublayout[j]

            line_edit = sublayout.itemAt(1).widget()
            combo_box = sublayout.itemAt(2).widget()
            
            if isinstance(delay_value, str):
                combo_box.setCurrentText('str (ns)')
                line_edit.setText(str(delay_value))
            else:
                # Pick the shortest display (digits only), tie → ns
                unit, txt = self._best_unit_and_text(int(delay_value))
                combo_box.setCurrentText(unit)
                line_edit.setText(txt)

        self.bracket_exists = False
        self.drag_container = DragContainer()
        self.layout_dataset.addWidget(self.drag_container)

        for i, row_data in enumerate(self.data_matrix):
            self.add_column()
            row_widget = self.drag_container.items[i].widget
            row_layout = row_widget.layout()

            sublayout = row_layout.itemAt(0)  
            line_edit = sublayout.itemAt(1).widget() 
            combo_box = sublayout.itemAt(2).widget()  
                
            if isinstance(row_data[0], str):
                combo_box.setCurrentText('str (ns)')
                line_edit.setText(str(row_data[0]))
            else:
                # Numeric (ns internal): choose best unit by shortest digits
                unit, txt = self._best_unit_and_text(int(row_data[0]))
                combo_box.setCurrentText(unit)
                line_edit.setText(txt)


            for j in range(1, 9):  
                checkbox = row_layout.itemAt(j).widget()
                checkbox.setChecked(bool(row_data[j]))
        if self.repeat_info[1]!=-1:
            # otherwise just default repeat_info setting and no bracket
            self.on_add_bracket(self.repeat_info[0], self.repeat_info[1])
        # load ref_info
        checkbox = self.btn_ref_info.layout().itemAt(0).widget()
        checkbox.setChecked(self.ref_info.get('is_ref', False))
        for ii, type in enumerate(['signal', 'DAQ', 'DAQ_ref', 'clock']):
            combobox = self.btn_ref_info.layout().itemAt(ii+1).layout().itemAt(1).widget()
            ch = self.ref_info.get(type, None)
            if ch is not None:
                combobox.setCurrentText(f'Ch{ch}')
            else:
                combobox.setCurrentText(f'None')

    def _validate_scalar_editor(self, editor: QLineEdit, where: str) -> bool:
        """
        Minimal validation for a duration/delay editor.
        Trust the editor's validator: the text must be Acceptable (final & valid).
        Anything else (Intermediate/Invalid) -> warn and abort.
        """
        s = editor.text() or ""
        val = editor.validator()
        if val is None:
            return True  # nothing to validate

        state, _, _ = val.validate(s, 0)
        if state != QValidator.Acceptable:
            return False
        return True

    def current_state_valid(self):
        """
        Validate all editors via their validators only.
        Abort on the first non-Acceptable field.
        """
        # --- Delays (8 channels) ---
        for j in range(8):
            sublayout = self.delay_sublayout[j]
            editor = sublayout.itemAt(1).widget()   # FloatOrXLineEdit
            where  = f"Delay of {self.channel_names_map[j]}"
            if not self._validate_scalar_editor(editor, where):
                return False

        # --- Pulse durations ---
        pulse_items = [it for it in self.drag_container.items if it.item_type == 'pulse']
        for idx, it in enumerate(pulse_items):
            row_layout = it.widget.layout()
            dur_layout = row_layout.itemAt(0)        # the VBox with label/edit/unit
            editor     = dur_layout.itemAt(1).widget()
            where      = f"Pulse{idx} duration"
            if not self._validate_scalar_editor(editor, where):
                return False

        return True        

    def save_data(self):
        if not self.current_state_valid():
            QMessageBox.warning(self, 'Invalid input', 'invalid or incomplete expression.')
            return False

        # --- All good → commit to device_handle ---        
        self.device_handle.data_matrix = self.read_data()
        self.device_handle.delay_array = self.read_delay()
        self.device_handle.channel_names = self.read_channel_names()
        self.device_handle.repeat_info = self.read_repeat_info()
        self.device_handle.ref_info = self.read_ref_info()

        return True

                    
    def read_data(self):
        pulse_list = [item for item in self.drag_container.items if item.item_type=='pulse']
        count = len(pulse_list) #number of pulses 
        data_matrix = [[0]*9 for _ in range(count)] 

        for i in range(count):
            item = pulse_list[i]
            widget = item.widget
            layout = widget.layout()
            for j in range(1):
                item_sub = layout.itemAt(j)
                layout_sub = item_sub.layout()
                duration_unit = layout_sub.itemAt(2).widget().currentText()
                if(duration_unit == 'str (ns)'):
                    duration_num = layout_sub.itemAt(1).widget().text()
                elif(duration_unit == 'ns'):
                    duration_num = eval(layout_sub.itemAt(1).widget().text())
                    duration_num = int(duration_num*1e0)
                elif(duration_unit == 'us'):
                    duration_num = eval(layout_sub.itemAt(1).widget().text())
                    duration_num = int(duration_num*1e3)
                elif(duration_unit == 'ms'):
                    duration_num = eval(layout_sub.itemAt(1).widget().text())
                    duration_num = int(duration_num*1e6)
                    
                data_matrix[i][j] = duration_num
                
            for j in range(1,9):
                item = layout.itemAt(j)
                widget = item.widget()
                if(widget.isChecked()):
                    data_matrix[i][j] = 1
        
        return data_matrix

    def read_delay(self):


        delay_array = [0, 0, 0, 0, 0, 0, 0, 0]
        for j in range(8):
            item_sub = self.delay_sublayout[j]
            layout_sub = item_sub.layout()
            duration_unit = layout_sub.itemAt(2).widget().currentText()
            if(duration_unit == 'str (ns)'):
                duration_num = layout_sub.itemAt(1).widget().text()
            elif(duration_unit == 'ns'):
                duration_num = eval(layout_sub.itemAt(1).widget().text())
                duration_num = int(duration_num*1e0)
            elif(duration_unit == 'us'):
                duration_num = eval(layout_sub.itemAt(1).widget().text())
                duration_num = int(duration_num*1e3)
            elif(duration_unit == 'ms'):
                duration_num = eval(layout_sub.itemAt(1).widget().text())
                duration_num = int(duration_num*1e6)
                
            delay_array[j] = duration_num

        return delay_array

    def read_channel_names(self):

        channel_names = ['', '', '', '', '', '', '', '']
        for j in range(8):
            item_sub = self.names_sublayout[j]
            layout_sub = item_sub.layout()
                
            channel_names[j] = layout_sub.itemAt(1).widget().text()

        return channel_names

    def read_repeat_info(self):
        if not self.bracket_exists:
            return [0, -1, 1]
            # default repeat_info
        else:
            bracket_index_list = [ii for ii, item in enumerate(self.drag_container.items) if item.item_type!='pulse']
            start_index = bracket_index_list[0]
            end_index = bracket_index_list[1] - 2 # [start_index, end_index] pulses area inside bracket, include end_index

            widget = self.drag_container.items[bracket_index_list[1]].widget
            layout = widget.layout()
            item_sub = layout.itemAt(0)
            layout_sub = item_sub.layout()
            repeat = layout_sub.itemAt(1).widget().value()

            return [start_index, end_index, repeat]

    def read_ref_info(self):
        new_ref_info = {}
        checkbox = self.btn_ref_info.layout().itemAt(0).widget()
        new_ref_info['is_ref'] = checkbox.isChecked()
        combobox = self.btn_ref_info.layout().itemAt(1).layout().itemAt(1).widget()
        new_ref_info['signal'] = None if combobox.currentText()=='None' else int(combobox.currentText()[-1])
        combobox = self.btn_ref_info.layout().itemAt(2).layout().itemAt(1).widget()
        new_ref_info['DAQ'] = None if combobox.currentText()=='None' else int(combobox.currentText()[-1])
        combobox = self.btn_ref_info.layout().itemAt(3).layout().itemAt(1).widget()
        new_ref_info['DAQ_ref'] = None if combobox.currentText()=='None' else int(combobox.currentText()[-1])
        combobox = self.btn_ref_info.layout().itemAt(4).layout().itemAt(1).widget()
        new_ref_info['clock'] = None if combobox.currentText()=='None' else int(combobox.currentText()[-1])
        return new_ref_info
    
        
    def remove_column(self):
        pulse_list = [ii for ii, item in enumerate(self.drag_container.items) if item.item_type=='pulse']
        count = len(pulse_list)
        #print(count)
        if(count>=3):
            widget = self.drag_container.items[pulse_list[-1]].widget
            widget.deleteLater()
            self.drag_container.items = [item for ii, item in enumerate(self.drag_container.items) if ii!=pulse_list[-1]]
            self.drag_container.refresh_layout()
            
    def add_column(self):
        pulse_list = [item for item in self.drag_container.items if item.item_type=='pulse']
        count = len(pulse_list)

        row = QGroupBox(f'Period{count}')

        row.setMinimumWidth(100)

        self.drag_container.add_item(row, "pulse")
        layout_data = QVBoxLayout(row)
        layout_data.setContentsMargins(8, 8, 8, 8)
        layout_data.setSpacing(8)
        sublayout = QVBoxLayout()
        layout_data.addLayout(sublayout)
        btn = QLabel('Duration:')
        btn.setFixedSize(80,30)
        sublayout.addWidget(btn)
        btn = FloatOrXLineEdit('10')
        btn.set_resolution(self.t_resolution)
        btn.set_allow_any(False)
        btn.setFixedSize(80,30)
        sublayout.addWidget(btn)
        btn2 = QComboBox()
        btn2.addItems(['ns','us' ,'ms', 'str (ns)'])
        btn2.setFixedSize(80,30)
        sublayout.addWidget(btn2)
        btn.textChanged.connect(lambda text, cb=btn2: self.handle_text_change(text, cb))
        btn2.currentTextChanged.connect(lambda text, cb=btn: self.on_time_unit_change(text, cb))
        
        for index in range(1, 9):
            btn = QCheckBox()
            btn.setFixedHeight(30)
            btn.setText(self.channel_names_map[index-1])
            btn.setCheckable(True)
            btn.setMinimumWidth(80)
            layout_data.addWidget(btn)

    def on_set_x(self):
        editor = self.edit_set_x
        if not self._validate_scalar_editor(editor, "x (ns)"):
            return
        self.device_handle.x = str2python(editor.text())


    def update_x(self):
        value = float2str(self.device_handle.x, length=self.MAX_LEN)
        self.edit_display_x.setText(value)

        lb = getattr(self.device_handle, 'x_lb', None)
        ub = getattr(self.device_handle, 'x_ub', None)
        lb_s = float2str_eng(lb, length=8) if lb is not None else '-∞'
        ub_s = float2str_eng(ub, length=8) if ub is not None else '∞'
        self.edit_bound_x.setText(f"{lb_s} to {ub_s}")

    def add_delay(self):
        count = self.layout_dataset.count()
        row = QGroupBox('Delay and X')
        self.layout_dataset.addWidget(row)
        layout_data = QVBoxLayout(row)
        layout_data.setContentsMargins(8, 8, 8, 8)
        layout_data.setSpacing(8)

        sublayout = QHBoxLayout()
        layout_data.addLayout(sublayout)

        self.btn_set_x = QPushButton(f'Set x (ns):')
        self.btn_set_x.setMinimumWidth(70)
        self.btn_set_x.setFixedHeight(30)
        sublayout.addWidget(self.btn_set_x)
        self.btn_set_x.clicked.connect(self.on_set_x)
        self.edit_set_x = FloatLineEdit()
        self.edit_set_x.setFixedSize(168,30)
        self.edit_set_x.set_resolution(self.t_resolution)
        self.edit_set_x.set_allow_any(True)
        sublayout.addWidget(self.edit_set_x)
        self.edit_set_x.setMaxLength(self.MAX_LEN)

        sublayout = QHBoxLayout()
        layout_data.addLayout(sublayout)
        label_range = QLabel('Range:')
        label_range.setMinimumWidth(70)
        label_range.setFixedHeight(30)
        sublayout.addWidget(label_range)
        self.edit_bound_x = QLineEdit()
        self.edit_bound_x.setFixedSize(168,30)
        self.edit_bound_x.setEnabled(False)
        sublayout.addWidget(self.edit_bound_x)


        sublayout = QHBoxLayout()
        layout_data.addLayout(sublayout)
        btn = QLabel(f'x (ns):')
        btn.setMinimumWidth(70)
        btn.setFixedHeight(30)
        sublayout.addWidget(btn)
        self.edit_display_x = FloatLineEdit()
        self.edit_display_x.setMaxLength(self.MAX_LEN)
        self.edit_display_x.setFixedSize(168,30)
        self.edit_display_x.setEnabled(False)
        sublayout.addWidget(self.edit_display_x)


        self.delay_sublayout = []
        
        for index in range(1, 9):
            sublayout = QHBoxLayout()
            self.delay_sublayout.append(sublayout)
            layout_data.addLayout(sublayout)
            btn = QLabel(f'{self.channel_names_map[index-1]} delay:')
            btn.setMinimumWidth(70)
            btn.setFixedHeight(30)
            sublayout.addWidget(btn)
            btn = FloatOrXLineEdit('0')
            btn.set_resolution(self.t_resolution)
            btn.set_allow_any(True)
            btn.setFixedSize(80,30)
            sublayout.addWidget(btn)
            btn2 = QComboBox()
            btn2.addItems(['ns','us' ,'ms', 'str (ns)'])
            btn2.setFixedSize(80,30)
            sublayout.addWidget(btn2)
            btn.textChanged.connect(lambda text, cb=btn2: self.handle_text_change(text, cb))
            btn2.currentTextChanged.connect(lambda text, cb=btn: self.on_time_unit_change(text, cb))

        self.auto_fixed_size(row)

        self.update_x()
        self.x_timer = QTimer(self)
        self.x_timer.setInterval(200)
        self.x_timer.timeout.connect(self.update_x)
        self.x_timer.start()


    def add_channel_names(self):
        count = self.layout_dataset.count()
        row = QGroupBox('Channel Names and Duration')
        self.layout_dataset.addWidget(row)
        layout_data = QVBoxLayout(row)
        layout_data.setContentsMargins(8, 8, 8, 8)
        layout_data.setSpacing(8)

        sublayout = QVBoxLayout()
        layout_data.addLayout(sublayout)

        lbl = QLabel("Total duration in ns:")
        lbl.setFixedHeight(30)
        self.edit_total = QLineEdit('')
        self.edit_total.setFixedHeight(30)
        self.edit_total.setEnabled(False)
        sublayout.addWidget(lbl)
        sublayout.addWidget(self.edit_total)

        layout_data.addSpacing(114-30-8-30-8)
        self.names_sublayout = []
        
        for index in range(1, 9):
            sublayout = QHBoxLayout()
            self.names_sublayout.append(sublayout)
            layout_data.addLayout(sublayout)
            btn = QLabel(f'Ch{index-1} name:')
            btn.setFixedHeight(30)
            btn.setMinimumWidth(70)
            sublayout.addWidget(btn)
            btn = QLineEdit('')
            btn.setMinimumWidth(70)
            btn.setFixedHeight(30)
            btn.textChanged.connect(lambda text, channel=(index-1): self.replace_channel_names(text, channel))
            sublayout.addWidget(btn)

        self.auto_fixed_size(row)

        self._update_total_duration_label()


    def save_to_file(self):
        self.save_data()

        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getSaveFileName(self,'select',get_pulses_dir(),'data_figure (*.npz)',options=options)

        if fileName == '':
            return

        if '.npz' in fileName[-4:]:
            fileName = fileName[:-4]

        set_pulses_dir(os.path.dirname(fileName))
        self.device_handle.save_to_file(addr = fileName)

        self.address_str = fileName
        self.last_load_state = None
        self.last_save_state = self.snapshot()
        self.stateui_manager.filestate = self.stateui_manager.FileState.SAVE

    def load_from_file(self):
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getOpenFileName(self,'select',get_pulses_dir(),'data_figure (*.npz)',options=options)

        if fileName == '':
            return

        set_pulses_dir(os.path.dirname(fileName))
        self.device_handle.load_from_file(addr = fileName[:-4])
        self.load_data()

        self.address_str = fileName[:-4]
        self.last_save_state = None
        self.last_load_state = self.snapshot()
        self.stateui_manager.filestate = self.stateui_manager.FileState.LOAD

    def replace_channel_names(self, text, channel):
        if text == '':
            self.channel_names_map[channel] = f'Ch{channel}'
        else:
            self.channel_names_map[channel] = text

        # make sure load works where channel_names loaded before all other

        item = self.layout_dataset.itemAt(1)#second is delay
        if item is not None:
            item_sub = self.delay_sublayout[channel]
            layout_sub = item_sub.layout()
            layout_sub.itemAt(0).widget().setText(f'{self.channel_names_map[channel]} delay')
            # replace delay row

        if self.drag_container is None:
            return
        else:
            pulse_list = [item for item in self.drag_container.items if item.item_type=='pulse']
            for item in pulse_list:
                if item is not None:
                    widget = item.widget
                    layout = widget.layout()
                    item = layout.itemAt(channel+1)
                    item.widget().setText(f'{self.channel_names_map[channel]}')

            # replace pulse row

    def snapshot(self):
        return {
            'data_matrix':  self.read_data(),
            'delay_array':  self.read_delay(),
            'channel_names':self.read_channel_names(),
            'ref_info':     self.read_ref_info(),
            'repeat_info':  self.read_repeat_info(),
        }

    def _update_total_duration_label(self):
        """Refresh the 'Total:' expression above the Delay column (string only)."""
        try:
            if not self.current_state_valid():
                self.edit_total.setText("")
                return
            dm  = self.read_data()
            ri  = self.read_repeat_info()
            rfi = self.read_ref_info()
            expr = self.device_handle.total_duration_str(
                data_matrix=dm, repeat_info=ri, ref_info=rfi
            )
            self.edit_total.setText(f"{expr}")
        except Exception:
            self.edit_total.setText("")


    def _check_device_change(self):
        if not self.current_state_valid():
            self._update_total_duration_label()
            return
        current = self.snapshot()
        on = self.device_handle.on
        last_on_state = self.device_handle.last_on_state

        if last_on_state is not None:
            if last_on_state!=current:
                if on is True:
                    self.stateui_manager.runstate = self.stateui_manager.RunState.UNSYNCED
                else:
                    self.stateui_manager.runstate = self.stateui_manager.RunState.STOP
            else:
                if on is True:
                    self.stateui_manager.runstate = self.stateui_manager.RunState.RUNNING
                else:
                    self.stateui_manager.runstate = self.stateui_manager.RunState.STOP
        else:
            self.stateui_manager.runstate = self.stateui_manager.RunState.INIT

        if self.last_save_state is not None:
            if self.last_save_state!=current:
                self.stateui_manager.filestate = self.stateui_manager.FileState.UNSAVED
            else:
                self.stateui_manager.filestate = self.stateui_manager.FileState.SAVE

        if self.last_load_state is not None:
            if self.last_load_state!=current:
                self.stateui_manager.filestate = self.stateui_manager.FileState.UNSAVED
            else:
                self.stateui_manager.filestate = self.stateui_manager.FileState.LOAD

        self._update_total_duration_label()


                
    def hideEvent(self, event):
        win = self.window()
        if getattr(win, '_hiding_via_close', False):
            if self._poll_timer.isActive():
                self._poll_timer.stop()
            if self.x_timer.isActive():
                self.x_timer.stop()
        super().hideEvent(event)

    def showEvent(self, event):
        if not self._poll_timer.isActive():
            self._poll_timer.start()
        if not self.x_timer.isActive():
            self.x_timer.start()

        super().showEvent(event)

    def closeEvent(self, event):
        super().closeEvent(event)

class DraggableItem:
    """
    Save properties
    """
    def __init__(self, widget, item_type):
        self.widget = widget
        self.item_type = item_type



class DragContainer(QWidget):
    """
    Drag container for dragable bracket in pulse control gui
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)

        self.layout_main = QHBoxLayout(self)
        self.layout_main.setContentsMargins(0, 0, 0, 0)
        self.layout_main.setSpacing(5)
        self.items = []

        self.dragStartPos = None
        self.draggingIndex = None

        # maintain a indicator between containers but hide 
        self.insert_indicator = QFrame()
        self.insert_indicator.setFrameShape(QFrame.VLine)
        self.insert_indicator.setFrameShadow(QFrame.Raised)
        self.insert_indicator.hide()

    def add_item(self, widget, item_type):
        it = DraggableItem(widget, item_type)
        self.items.append(it)
        self.layout_main.addWidget(widget)

    def insert_item(self, index, widget, item_type):
        it = DraggableItem(widget, item_type)
        self.items.insert(index, it)
        self.refresh_layout()

    def refresh_layout(self):
        # remove all widget
        while self.layout_main.count() > 0:
            c = self.layout_main.takeAt(0)
            w = c.widget()
            if w:
                w.setParent(None)
        # reinsert follows order
        for it in self.items:
            self.layout_main.addWidget(it.widget)
        # add end indicator
        self.layout_main.addWidget(self.insert_indicator)
        self.insert_indicator.hide()

    def index_of_widget(self, w):
        for i, it in enumerate(self.items):
            if it.widget == w:
                return i
        return -1

    # ========== Drag & Drop ==========
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.dragStartPos = event.pos()
            clicked_index = self.find_child_index_by_pos(event.pos())
            if clicked_index != -1:
                self.draggingIndex = clicked_index
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if (event.buttons() & Qt.LeftButton) and self.dragStartPos is not None:
            distance = (event.pos() - self.dragStartPos).manhattanLength()
            if distance > QApplication.startDragDistance():
                if self.draggingIndex is not None:
                    self.startDrag(self.draggingIndex)
                    self.draggingIndex = None
                    self.dragStartPos = None
        super().mouseMoveEvent(event)

    def change_style_sheet(self, widget, add_style:str):
        self.old_ss = widget.styleSheet() or ""
        self.new_ss = self.old_ss + (";" if self.old_ss and not self.old_ss.endswith(";") else "") \
         + add_style
        widget.setStyleSheet(self.new_ss)

    def startDrag(self, index):
        drag = QDrag(self)
        mime = QMimeData()
        mime.setData("application/x-drag-bracket", str(index).encode("utf-8"))
        drag.setMimeData(mime)

        # highlight selected drag container
        self.index = index
        self.change_style_sheet(self.items[self.index].widget, "border: 2px solid grey;")

        dropAction = drag.exec_(Qt.MoveAction)

        # end of drag
        if self.items[self.index].item_type == 'pulse':
            self.items[self.index].widget.setStyleSheet(self.old_ss)
            self.update_pulse_index()
        else:
            self.items[self.index].widget.setStyleSheet(self.old_ss)


    def update_pulse_index(self):
        pulse_list = [item for item in self.items if item.item_type=='pulse']
        for ii, item in enumerate(pulse_list):
            widget = item.widget
            widget.setTitle(f'Period{ii}')

    def dragEnterEvent(self, event):
        if event.mimeData().hasFormat("application/x-drag-bracket"):
            event.acceptProposedAction()
        else:
            super().dragEnterEvent(event)

    def dragMoveEvent(self, event):
        if event.mimeData().hasFormat("application/x-drag-bracket"):
            event.acceptProposedAction()
            # pos of drag
            insert_pos = self.get_item_at_pos(event.pos())
            self.show_insert_indicator(insert_pos)
        else:
            super().dragMoveEvent(event)

    def dropEvent(self, event):
        if event.mimeData().hasFormat("application/x-drag-bracket"):
            # 1) get item of drag container
            raw_data = event.mimeData().data("application/x-drag-bracket")
            old_bytes = bytes(raw_data)
            old_index = int(old_bytes.decode("utf-8"))
            dragged_item = self.items[old_index]

            # 2) new pos
            insert_pos = self.get_item_at_pos(event.pos())
            # 3) new item array
            new_items = self.items[:]             
            new_items.remove(dragged_item)
            if insert_pos > old_index:
                insert_pos -= 1
            new_items.insert(insert_pos, dragged_item)

            # 4) check constarin
            if not self.check_bracket_constraints(new_items):
                event.ignore()
                self.insert_indicator.hide()
                return
            else:
                event.setDropAction(Qt.MoveAction)
                event.accept()
                self.index = insert_pos
                self.items = new_items
                self.refresh_layout()
                self.insert_indicator.hide()

        else:
            super().dropEvent(event)


    def dragLeaveEvent(self, event):
        self.insert_indicator.hide()
        super().dragLeaveEvent(event)

    # ========== sub functions ==========
    def find_child_index_by_pos(self, pos):
        for i, it in enumerate(self.items):
            if it.widget.geometry().contains(pos):
                return i
        return -1

    def get_item_at_pos(self, pos):
        x = pos.x()
        for i, it in enumerate(self.items):
            w = it.widget
            geo = w.geometry()
            mid = geo.x() + geo.width() // 2
            if x < mid:
                return i
        return len(self.items)

    def show_insert_indicator(self, index):
        self.layout_main.removeWidget(self.insert_indicator)
        self.layout_main.insertWidget(index, self.insert_indicator)
        self.insert_indicator.show()

    def check_bracket_constraints(self, item_list=None):
        if item_list is None:
            item_list = self.items  

        start_idx = None
        end_idx = None
        for i, it in enumerate(item_list):
            if it.item_type == "bracket_start":
                start_idx = i
            elif it.item_type == "bracket_end":
                end_idx = i

        if start_idx is not None and end_idx is not None:
            # 1) end must larger than start
            if end_idx <= start_idx:
                return False
            # 2) at least two pulses away
            if end_idx < start_idx + 3:
                return False
        return True

