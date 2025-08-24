from .gui_device import *

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure



class BaseLivePlotGUI(QWidget):
    # all the shared methods of LivePlot1DGUI, LivePlot2DGUI


    def __init__(self, parent=None, cls_name=None, handled_in_control=None, overrides=None):
        super().__init__(parent)
        self.parent = parent
        self.cls_name = cls_name
        if handled_in_control is None:
            self.handled_in_control = {}
        else:
            self.handled_in_control = handled_in_control
        self._ov = overrides or {}  # task-provided overrides
        self._caller_params = inspect.signature(self.cls_name.caller).parameters # cache caller signature parameters once

        sig = inspect.signature(self.cls_name.caller)
        if 'fig_sub' in sig.parameters:
            self.has_sub = True
        else:
            self.has_sub = False

        if self.has_sub is True:
            from Confocal_GUIv2.logic import PLEMeasurement
            self.canvas_sub = self._make_placeholder_plot(cls_name=PLEMeasurement)   # draw a safe placeholder into fig
            self.canvas_sub.setWindowFlags(Qt.Window)  
            self.canvas_sub.setWindowTitle('Sub Figure')  
            self.canvas_sub.hide()

        self.init_ui_base()

        self.combo_fit.addItems(['lorent', 'decay', 'rabi', 'lorent_zeeman', 'center'])
        self.load_default_base()

        if self.has_sub is True:
            self.btn_fig_sub.clicked.connect(
                lambda: self.canvas_sub.setVisible(not self.canvas_sub.isVisible())
            )

        self.btn_settings.clicked.connect(self.on_settings)
        self.btn_unit.clicked.connect(self.on_unit)
        self.btn_fit.clicked.connect(self.on_fit)
        self.btn_fit_in.clicked.connect(self.on_fit_in)
        self.btn_save.clicked.connect(self.on_save)
        self.btn_start.clicked.connect(self.on_start)
        self.btn_stop.clicked.connect(self.on_stop)
        self.btn_load.clicked.connect(self.on_load)
        self.spin_repeat.valueChanged.connect(self.estimate_run_time)

        self.address_str = None
        self.stateui_manager = StateUIManager(status_dots=[self.status_dot,], 
            label_names=[self.label_name,], btn_ons=[self.btn_start,], 
            btn_saves=[self.btn_save,], address_getter=lambda: self.address_str)
        self.stateui_manager.runstate = self.stateui_manager.RunState.INIT
        self.stateui_manager.filestate = self.stateui_manager.FileState.UNTITLED


    def get_default(self, name: str, fallback):
        """Prefer overrides[name], else caller signature default, else fallback."""
        if name in self._ov:
            return self._ov[name]
        p = self._caller_params.get(name)
        if p is not None and p.default is not inspect._empty:
            return p.default
        return fallback

    def _make_placeholder_plot(self, cls_name):
        """Create a safe placeholder plot using the Measurement's plotter.
        This does NOT touch devices or controller; just draws an empty-but-valid figure.
        """
        # Build neutral labels without depending on device states
        from Confocal_GUIv2.live_plot import Live1D as _Live1D, LiveLiveDis as _LiveLiveDis, Live2DDis as _Live2DDis
        # Explicit string -> class mapping (avoid globals() in a function scope)
        from Confocal_GUIv2.live_plot import apply_rcparams
        _plotter_map = {
            'Live1D': _Live1D,
            'LiveLiveDis': _LiveLiveDis,
            'Live2DDis': _Live2DDis,
        }
        apply_rcparams()
        # otherwise the first fig will has style different from rcparams
        fig = Figure()
        canvas = FigureCanvasQTAgg(fig)

        # Build neutral labels without depending on device states
        unit = getattr(cls_name, 'unit', '1')
        xlabel = getattr(cls_name, 'xlabel', 'X')
        ylabel = getattr(cls_name, 'ylabel', 'Y')
        zlabel = getattr(cls_name, 'zlabel', 'Z')
        if unit not in ('1', None, '') and getattr(cls_name, 'plotter', '') in ('Live1D', 'LiveLiveDis'):
            xlabel = f"{xlabel} ({unit})"
        labels = [xlabel, ylabel, zlabel]

        plotter_name = getattr(cls_name, 'plotter', '')

        # 1D family: Live1D / LiveLiveDis
        if plotter_name in ('Live1D', 'LiveLiveDis'):
            N = 200
            x = np.linspace(0.0, 1.0, N).reshape(-1, 1)  # shape (N, 1)
            y = np.zeros((N, 1), dtype=float)           # avoid all-NaN; keep relim stable
            plotter_cls = _plotter_map[plotter_name]
            live = plotter_cls(
                data_x=x, data_y=y, labels=labels, update_time=0.1,
                fig=fig, relim_mode='normal'
            )
            live.init_figure_and_data()
            live.after_plot()
            return canvas

        # 2D: Live2DDis
        if plotter_name == 'Live2DDis':
            nx, ny = 30, 30
            xv = np.linspace(-1.0, 1.0, nx)
            yv = np.linspace(-1.0, 1.0, ny)
            xx, yy = np.meshgrid(xv, yv)
            pts = np.column_stack([xx.ravel(), yy.ravel()])  # shape (nx*ny, 2)
            z = np.zeros((pts.shape[0], 1), dtype=float)
            plotter_cls = _plotter_map['Live2DDis']
            live = plotter_cls(
                data_x=pts, data_y=z, labels=labels, update_time=0.1,
                fig=fig, relim_mode='tight'
            )
            live.init_figure_and_data()
            live.after_plot()
            return canvas


    def init_ui_base(self):
        # Main vertical layout
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)


        # ─── Frame + Grid ───
        self.plot_btn_frame = QFrame(parent = self, round=('SW', 'SE', 'NE'))
        grid = QGridLayout(self.plot_btn_frame)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(0)

        frame = QWidget()
        frame_layout = QHBoxLayout(frame)
        frame_layout.setContentsMargins(8, 0, 8, 0)
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
        grid.addWidget(frame, 0, 0)

        # 1) Plot canvas
        self.canvas = self._make_placeholder_plot(cls_name=self.cls_name)   # draw a safe placeholder into fig
        self.canvas.setParent(self.plot_btn_frame)    # attach parent for Qt ownership

        w_in, h_in = self.canvas.figure.get_size_inches()            # size now reflects plotter's fixed layout
        dpi = self.canvas.figure.get_dpi()
        self.canvas.setFixedSize(round(w_in * dpi), round(h_in * dpi))
        grid.addWidget(self.canvas, 1, 0)


        # 2) Settings panel (stacked)
        self.settings_stack = QStackedWidget(self.plot_btn_frame)
        self.settings_stack.addWidget(QWidget())  # blank

        # Real settings page
        settings_page = QGroupBox('Setting params (less used)')
        sp_layout = QVBoxLayout(settings_page)
        sp_layout.setContentsMargins(8, 8, 8, 8)
        sp_layout.setSpacing(8)


        # ── create two-column container ──
        col_container = QHBoxLayout()                  # horizontal box
        left_col  = QVBoxLayout()                      # first 10 rows
        right_col = QVBoxLayout()                      # from 11th onward
        right_most_col = QVBoxLayout()                 # from 21th onward
        col_container.addLayout(left_col)
        col_container.addLayout(right_col)
        col_container.addLayout(right_most_col)
        sp_layout.addLayout(col_container)

        # ── hijack all addLayout calls on sp_layout ──
        self._row_count = 1
        def _route(layout_item):
            """English comment: dispatch each row to left or right column"""
            if self._row_count <= 10:
                left_col.addLayout(layout_item)
            elif self._row_count <= 20:
                right_col.addLayout(layout_item)
            else:
                right_most_col.addLayout(layout_item)
            self._row_count += 1

        old_add = sp_layout.addLayout
        sp_layout.addLayout = lambda item: _route(item)

        self.check_enable_data_x = QCheckBox('Enable')
        self.spin_repeat = QDoubleSpinBox()
        self.combo_relim = QComboBox()
        self.spin_update_time = QDoubleSpinBox()
        self.combo_counter_mode = QComboBox()
        self.combo_data_mode = QComboBox()
        self.combo_update_mode = QComboBox()

        settings_widgets = [
            ('Raw data_x:', self.check_enable_data_x),
            ("Repeat:", self.spin_repeat),
            ("Relim:", self.combo_relim),
            ("Update time:", self.spin_update_time),
            ("Counter mode:", self.combo_counter_mode),
            ("Data mode:", self.combo_data_mode),
            ("Update mode:", self.combo_update_mode),
        ]
        label_w, widget_w = 110, 100
        for text, widget in settings_widgets:
            row = QHBoxLayout()
            row.setSpacing(8)
            lbl = QLabel(text)
            lbl.setFixedSize(label_w, 30)
            lbl.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
            widget.setFixedSize(widget_w, 30)
            row.addWidget(lbl)
            row.addWidget(widget)
            sp_layout.addLayout(row)
        self._add_devices(sp_layout)
        self._add_extra_settings(sp_layout)

        left_col.addStretch()
        right_col.addStretch()
        right_most_col.addStretch()
        sp_layout.addLayout = old_add

        self.settings_stack.addWidget(settings_page)
        self.settings_stack.setVisible(False)
        self.settings_stack.adjustSize()
        self.settings_stack.setFixedSize(self.settings_stack.width(), self.settings_stack.height())

        # 3) Overlay settings panel
        horizontal_offset = 8
        hbox = QHBoxLayout()
        hbox.setContentsMargins(0, 0, 0, 0)
        hbox.addSpacing(horizontal_offset)
        hbox.addWidget(self.settings_stack)
        hbox.addStretch()

        vertical_offset = 8
        vbox = QVBoxLayout()
        vbox.setContentsMargins(0, 0, 0, 0)
        vbox.addStretch()
        vbox.addLayout(hbox)
        vbox.addSpacing(vertical_offset)
        grid.addLayout(vbox, 1, 0)

        # 4) Control buttons
        control_layout = QHBoxLayout()
        control_layout.setContentsMargins(8, 0, 8, 8)
        control_layout.setSpacing(8)

        self.btn_settings = QPushButton("Settings")
        self.btn_settings.setFixedSize(80, 30)
        control_layout.addWidget(self.btn_settings)

        self.combo_fit = QComboBox()
        self.btn_unit = QPushButton("Unit")
        self.btn_fit = QPushButton("Fit")
        self.btn_fit_in = QPushButton(':')
        self.combo_fit = QComboBox()
        self.btn_save = QPushButton("Save")
        self.btn_start = QPushButton("Start")
        self.btn_stop = QPushButton("Stop")
        self.btn_load = QPushButton("Load")
        fit_layout = QHBoxLayout()
        fit_layout.setSpacing(1)
        fit_layout.setContentsMargins(0, 0, 0, 0)
        self.btn_fit.setFixedSize(35, 30)
        self.btn_fit_in.setFixedSize(15, 30)
        self.combo_fit.setFixedSize(80, 30)
        fit_layout.addWidget(self.btn_fit)
        fit_layout.addWidget(self.btn_fit_in)
        fit_layout.addWidget(self.combo_fit)

        for widget in (self.btn_unit, self.btn_save, self.btn_start, self.btn_stop, self.btn_load):
            widget.setFixedSize(70, 30)
            control_layout.addWidget(widget)
        self.btn_start.set_color(FLUENT_GREEN)
        self.btn_stop.set_color(FLUENT_RED)
        self.btn_save.set_color(FLUENT_YELLOW)
        self.btn_load.set_color(FLUENT_ORANGE)
        control_layout.addStretch()
        control_layout.addLayout(fit_layout)


        grid.addLayout(control_layout, 2, 0)
        self.main_layout.addWidget(self.plot_btn_frame)
        self.main_layout.addSpacing(8)

        # add fit in 
        self.fit_stack = QStackedWidget(self.plot_btn_frame)
        self.fit_stack.addWidget(QWidget())  # blank

        fit_page = QFrame()
        fit_in_layout = QHBoxLayout(fit_page)
        fit_in_layout.setContentsMargins(8, 8, 8, 8)
        fit_in_layout.setSpacing(4)
        lbl = QLabel('Fit params:')
        lbl.setFixedSize(100, 30)
        lbl.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
        self.edit_fit_in = QLineEdit()
        self.edit_fit_in.setFixedSize(200, 30)
        fit_in_layout.addWidget(lbl)
        fit_in_layout.addWidget(self.edit_fit_in)
        self.fit_stack.addWidget(fit_page)
        self.fit_stack.setVisible(False)
        self.fit_stack.adjustSize()
        self.fit_stack.setFixedSize(self.fit_stack.width(), self.fit_stack.height())

        horizontal_offset = 8
        hbox = QHBoxLayout()
        hbox.setContentsMargins(0, 0, 0, 0)
        hbox.addStretch()
        hbox.addWidget(self.fit_stack)
        hbox.addSpacing(horizontal_offset)
        vertical_offset = 8
        vbox = QVBoxLayout()
        vbox.setContentsMargins(0, 0, 0, 0)
        vbox.addStretch()
        vbox.addLayout(hbox)
        vbox.addSpacing(vertical_offset)
        grid.addLayout(vbox, 1, 0)

        # add btn_sub_fig
        if self.has_sub is True:
            self.btn_fig_sub = QPushButton('Sub')
            self.btn_fig_sub.setFixedSize(30, 30)

            horizontal_offset = 8
            hbox = QHBoxLayout()
            hbox.setContentsMargins(0, 0, 0, 0)
            hbox.addSpacing(horizontal_offset)
            hbox.addWidget(self.btn_fig_sub)
            hbox.addStretch()
            vertical_offset = 8
            vbox = QVBoxLayout()
            vbox.setContentsMargins(0, 0, 0, 0)
            vbox.addSpacing(vertical_offset)
            vbox.addLayout(hbox)
            vbox.addStretch()
            grid.addLayout(vbox, 1, 0)

    def load_default_base(self):
        # pull defaults
        repeat_default = self.get_default('repeat', 1)
        update_time_default = self.get_default('update_time', 0.1)
        counter_mode_default = self.get_default('counter_mode', 'apd')
        data_mode_default = self.get_default('data_mode', 'single')
        update_mode_default = self.get_default('update_mode', 'add')
        relim_mode_default = self.get_default('relim_mode', 'tight')

        self.spin_repeat.setRange(1, 1000)
        self.spin_repeat.setSingleStep(1)
        self.spin_repeat.setValue(repeat_default)

        self.spin_update_time.setRange(0.02, 10)
        self.spin_update_time.setSingleStep(0.1)
        self.spin_update_time.setValue(update_time_default)


        self.combo_relim.addItems(['normal', 'tight'])
        self.combo_relim.setCurrentText(relim_mode_default)
        self.combo_data_mode.setCurrentText(data_mode_default)
        self.combo_counter_mode.setCurrentText(counter_mode_default)
        self.combo_update_mode.addItems(self.cls_name.update_mode_valid)
        self.combo_update_mode.setCurrentText(update_mode_default)

    def on_device_counter(self):
        from Confocal_GUIv2.device import get_devices
        counter_override_name = self.device_widgets.get('counter')[0].currentText()
        dm = get_devices()
        counter_override_device = get_devices(counter_override_name) if dm else None
        self.combo_data_mode.clear()
        self.combo_counter_mode.clear() 
        if counter_override_device is None:
            self.combo_data_mode.addItems([])
            self.combo_counter_mode.addItems([])
        else:
            self.combo_data_mode.addItems(counter_override_device.data_mode_valid)
            self.combo_counter_mode.addItems(counter_override_device.counter_mode_valid)

    def read_params_base(self):

        repeat = int(self.spin_repeat.value())
        update_time = self.spin_update_time.value()
        relim_mode = self.combo_relim.currentText()
        data_mode = self.combo_data_mode.currentText()
        counter_mode = self.combo_counter_mode.currentText()
        update_mode = self.combo_update_mode.currentText()


        params_dict = {'repeat':repeat, 'update_time':update_time, 'relim_mode':relim_mode,
        'data_mode':data_mode, 'counter_mode':counter_mode, 'update_mode':update_mode}


        for name, (widget, default) in self.extra_widgets.items():
            raw = widget.text().strip()
            params_dict.update({name:str2python(raw)})

        for name, (widget, default) in self.device_widgets.items():
            params_dict.update({name:widget.currentText()})

        return params_dict


    def register_auto_range(self, controller=None):
        if controller is None:
            return
        else:
            controller.live_plot.area.callback = self._read_range
            controller.live_plot.zoom.callback = self._read_range
            controller.live_plot.cross.callback = self._read_x
            return

    def register_done_state(self):
        self.stateui_manager.runstate = self.stateui_manager.RunState.STOP
        self.stateui_manager.filestate = self.stateui_manager.FileState.UNTITLED
        self._restore_and_retrigger_set_checks() # restore set_x, set_y
        return

    def register_load_done_state(self, addr, controller=None):
        try:
            ok = self.load_devices_for_state(controller=controller)
        except Exception as e:
            # some devices are missing
            ok = False

        if not ok:
            return self.register_load_unsynced_state(addr=addr)

        self.stateui_manager.runstate = self.stateui_manager.RunState.STOP
        self.stateui_manager.filestate = self.stateui_manager.FileState.LOAD
        self._restore_and_retrigger_set_checks() # restore set_x, set_y
        cb = getattr(self, "_prefill_from_loaded_info", None)
        if callable(cb):
            cb()
        # possible other one time operations, e.g. load bound_pts
        return

    def register_load_unsynced_state(self, addr):
        self.stateui_manager.runstate = self.stateui_manager.RunState.UNSYNCED
        self.stateui_manager.filestate = self.stateui_manager.FileState.LOAD
        self._restore_and_retrigger_set_checks() # restore set_x, set_y
        return

    def closeEvent(self, event):
        if hasattr(self, 'canvas_sub'):
            self.canvas_sub.close()
        if self.stateui_manager.runstate in (self.stateui_manager.RunState.RUNNING,):
            self.live_plot.controller.stop()
        from Confocal_GUIv2.live_plot import save_and_close_previous
        save_and_close_previous()
        super().closeEvent(event)

    def hideEvent(self, event):
        win = self.window()
        if getattr(win, '_hiding_via_close', False):
            if hasattr(self, 'canvas_sub'):
                self.canvas_sub.close()
            if self.stateui_manager.runstate in (self.stateui_manager.RunState.RUNNING,):
                self.live_plot.controller.stop()


            if self.timer_x.isActive():
                self.timer_x.stop()
            if self.timer_estimate.isActive():
                self.timer_estimate.stop()
        super().hideEvent(event)

    def showEvent(self, event):
        if not self.timer_x.isActive():
            self.timer_x.start()
        if not self.timer_estimate.isActive():
            self.timer_estimate.start()
        super().showEvent(event)

    def all_inputs_valid(self, show_message: bool = False):
        """
        Minimal validation with field-specific English messages.
        - If data_x mode enabled: validate data_x + generic spin boxes; skip x/y ranges, steps, bound_pts
        - Else (scan mode): validate ranges/steps/bound_pts; skip data_x
        - On failure: when show_message=True, show which field is invalid (English). No focus change.
        """
        # whether data_x mode is enabled
        enable_data = False
        if hasattr(self, 'check_enable_data_x'):
            try:
                enable_data = bool(self.check_enable_data_x.isChecked())
            except Exception:
                enable_data = False

        # human-readable names for common fields (use only if the widget exists)
        name_pairs = [
            ('edit_xmin', 'X min'), ('edit_xmax', 'X max'),
            ('edit_ymin', 'Y min'), ('edit_ymax', 'Y max'),
            ('edit_bound_pts', 'Bound pts'), ('edit_data_x', 'data_x'),
            ('spin_xstep', 'X step'), ('spin_ystep', 'Y step'),
            ('spin_exposure', 'Exposure'), ('spin_sample_num', 'Pts'),
            ('spin_repeat', 'Repeat'),
        ]
        w2name = {}
        for attr, label in name_pairs:
            w = getattr(self, attr, None)
            if w is not None:
                w2name[w] = label

        def should_check(w):
            if w is None:
                return False
            if enable_data:
                # data_x mode: skip x/y ranges, steps, and bound_pts
                skip = {
                    getattr(self, 'edit_xmin', None),
                    getattr(self, 'edit_xmax', None),
                    getattr(self, 'edit_ymin', None),
                    getattr(self, 'edit_ymax', None),
                    getattr(self, 'edit_bound_pts', None),
                    getattr(self, 'spin_xstep', None),
                    getattr(self, 'spin_ystep', None),
                }
                return w not in skip
            else:
                # scan mode: skip data_x
                return w is not getattr(self, 'edit_data_x', None)

        def warn_invalid(field_label: str, kind: str):
            if not show_message:
                return
            if kind == 'invalid':
                QMessageBox.warning(self, 'Error', f'{field_label} is invalid.')
            elif kind == 'range':
                QMessageBox.warning(self, 'Error', f'{field_label} is out of range.')

        # QLineEdit with validator
        for w in getattr(self, 'check_validator_widgets', []):
            if not should_check(w):
                continue
            validator = w.validator()
            if validator is None:
                continue
            state, _, _ = validator.validate(w.text(), 0)
            if state != QValidator.Acceptable:
                warn_invalid(w2name.get(w, 'This field'), 'invalid')
                return False

        # Spin boxes range check
        for w in getattr(self, 'check_range_widgets', []):
            if not should_check(w):
                continue
            try:
                if not (w.minimum() <= w.value() <= w.maximum()):
                    warn_invalid(w2name.get(w, 'This field'), 'range')
                    return False
            except Exception:
                # not a spinbox-like widget; ignore
                pass

        return True



    def estimate_run_time(self):

        if self.stateui_manager.runstate in (self.stateui_manager.RunState.RUNNING,):
            points_total = self.live_plot.controller.measurement.points_total
            points_done = self.live_plot.controller.measurement.points_done
            if points_done<=1:
                return
            ratio = points_done/points_total
            time_use = time.time() - self.time_start
            if self.live_plot.controller.measurement.counter_mode == 'apd_sample':
                self.edit_display_time.setText(
                (
                    f'Current plot finishes in '
                    f'{time_use:.1f}s / {(time_use/ratio):.1f}s, '
                    f'{(ratio*100):.1f}%, '
                    f'{time_use/points_done:.3f}s/point'
                ))
            else:
                self.overhead_time = time_use/points_done - self.live_plot.controller.measurement.exposure
                self.edit_display_time.setText(
                (
                    f'Current plot finishes in '
                    f'{time_use:.1f}s / {(time_use/ratio):.1f}s, '
                    f'{(ratio*100):.1f}%, '
                    f'{(self.overhead_time + self.live_plot.controller.measurement.exposure):.3f}s/point'
                ))
        else:
            if not self.all_inputs_valid(show_message=False):
                return
            params_dict = self.read_params()
            if params_dict is None:
                return
            if params_dict['counter_mode']=='apd_sample':
                self.edit_display_time.setText(
                (
                    f'New plot finishes in unknown time'
                ))
            else:
                time_est = len(params_dict['data_x_generated'])*(params_dict['exposure']+self.overhead_time)*params_dict['repeat']
                # considering the overhead
                self.edit_display_time.setText(
                (
                    f'New plot finishes in {time_est:.1f}s'
                ))


    def on_set_x_toggled(self):
        if self.stateui_manager.runstate not in (self.stateui_manager.RunState.STOP,):
            return
        is_set = self.check_set_x.isChecked()
        if not is_set:
            self.live_plot.controller.measurement.to_final_state()
        else:
            value = self.spin_set_x.value()
            self.live_plot.controller.measurement.to_initial_state()
            self.live_plot.controller.measurement.set_x((value, None))

    def on_set_x_changed(self):
        if self.stateui_manager.runstate not in (self.stateui_manager.RunState.STOP,):
            return
        is_set = self.check_set_x.isChecked()
        if is_set:
            value = self.spin_set_x.value()
            self.live_plot.controller.measurement.to_initial_state()
            self.live_plot.controller.measurement.set_x((value, None))

    def on_set_y_toggled(self):
        if self.stateui_manager.runstate not in (self.stateui_manager.RunState.STOP,):
            return
        is_set = self.check_set_y.isChecked()
        if not is_set:
            self.live_plot.controller.measurement.to_final_state()
        else:
            value = self.spin_set_y.value()
            self.live_plot.controller.measurement.to_initial_state()
            self.live_plot.controller.measurement.set_x((None, value))

    def on_set_y_changed(self):
        if self.stateui_manager.runstate not in (self.stateui_manager.RunState.STOP,):
            return
        is_set = self.check_set_y.isChecked()
        if is_set:
            value = self.spin_set_y.value()
            self.live_plot.controller.measurement.to_initial_state()
            self.live_plot.controller.measurement.set_x((None, value))


    def on_unit(self):
        if self.stateui_manager.runstate in (self.stateui_manager.RunState.RUNNING, self.stateui_manager.RunState.INIT):
            QMessageBox.warning(self, 'Error', f'No figure to change unit.')
        else:
            self.live_plot.data_figure.change_unit()
            self.print_log(f'{self.cls_name.name} change unit')

    def on_fit(self):
        if self.stateui_manager.runstate in (self.stateui_manager.RunState.RUNNING, self.stateui_manager.RunState.INIT):
            QMessageBox.warning(self, 'Error', f'No figure to fit.')
        else:
            if self.live_plot.data_figure.fit is not None:
                self.live_plot.data_figure.clear()
            else:
                fit_func = self.combo_fit.currentText()
                fit_in = self.edit_fit_in.text()
                popt_str_, popt = eval(f'self.live_plot.data_figure.{fit_func}({fit_in})')
                self.print_log(f'{self.cls_name.name} fit result {popt_str_[0]} are {popt}')

    def on_fit_in(self):
        if self.fit_stack.isVisible():
            self.fit_stack.setVisible(False)
        else:
            self.fit_stack.setCurrentIndex(1)
            self.fit_stack.setVisible(True)
            self.fit_stack.raise_()

    def on_save(self):
        if self.stateui_manager.runstate in (self.stateui_manager.RunState.RUNNING, self.stateui_manager.RunState.INIT):
            QMessageBox.warning(self, 'Error', f'No figure to save.')
        else:
            addr = self.parent.read_addr() if self.parent is not None else ''
            extra_info = None
            saved_addr = self.live_plot.data_figure.save(addr=addr, extra_info=extra_info)
            self.address_str = saved_addr
            self.stateui_manager.filestate = self.stateui_manager.FileState.SAVE
            self.print_log(f'{self.cls_name.name} save to {saved_addr}')


    def print_log(self, log_info):
        if self.parent is not None:
            self.parent.print_log(log_info)


    def _remember_and_disable_set_checks(self):
        for name in ('check_set_x', 'check_set_y'):
            if hasattr(self, name):
                chk = getattr(self, name)
                chk.setProperty('_prev_checked', chk.isChecked())
                chk.setChecked(False)
                chk.setEnabled(False)
        for name in ('spin_set_x', 'spin_set_y'):
            if hasattr(self, name):
                spin = getattr(self, name)
                spin.setEnabled(False)

    def _restore_and_retrigger_set_checks(self):
        for name in ('check_set_x', 'check_set_y'):
            if hasattr(self, name):
                chk = getattr(self, name)
                prev = bool(chk.property('_prev_checked') or False)
                chk.setEnabled(True)
                if prev:
                    chk.setChecked(True)
        for name in ('spin_set_x', 'spin_set_y'):
            if hasattr(self, name):
                spin = getattr(self, name)
                spin.setEnabled(True)

    def check_start_state(self):

        if not self.all_inputs_valid(show_message=True):
            return False

        if self.stateui_manager.runstate in (self.stateui_manager.RunState.RUNNING,):
            self.on_stop()
            return True
            # auto restart

        if (self.parent is not None) and self.parent.any_is_running():
            on_start_state = self.parent.check_on_start.state()
            if on_start_state == 0: # Replace
                self.parent.stop_all()
                return True
            elif on_start_state == 1: # Block
                QMessageBox.warning(self, 'Error', f'Other live plot is running, stop before start.')
                return False
            elif on_start_state == 2: # Parallel
                return True
        return True



    def load_devices_for_state(self, controller=None):
        if controller is None:
            return False
        from Confocal_GUIv2.device import get_devices
        if get_devices() is None:
            return False
        # which means no devices available
        for device_name in controller.measurement.KEYS_FOR_DEVICES:
            widget = self.device_widgets.get(device_name)[0]
            device = get_devices(widget.currentText())
            setattr(controller.measurement, device_name, device)
            # load devices for the use of read_x, set_x e.g.
        return True

    @reentrancy_guard # will not respond the second call until on_start finished calling
    def on_start(self):

        is_ready = self.check_start_state()
        if not is_ready:
            return
        params_dict = self.read_params()
        sig = inspect.signature(self.cls_name.caller)
        filtered = {k: v for k, v in params_dict.items() if k in sig.parameters}
        filtered.update({'fig':self.canvas.figure, 'qt_parent':self, 'auto_save_and_close':True, 'is_GUI':False})
        if self.has_sub is True:
            filtered['fig_sub'] = self.canvas_sub.figure
            self.canvas_sub.show()
            QtWidgets.QApplication.processEvents()
            self.canvas_sub.draw()
        self._remember_and_disable_set_checks() # diable set_x, set_y, and trigger to_final_state
        self.stateui_manager.runstate = self.stateui_manager.RunState.RUNNING
        self.stateui_manager.filestate = self.stateui_manager.FileState.UNTITLED
        self.live_plot = self.cls_name.caller(**filtered)
        if self.live_plot is None:
            self.stateui_manager.runstate = self.stateui_manager.RunState.INIT
            self._restore_and_retrigger_set_checks()
            return

        self.live_plot.controller.register_after_task(func=self.register_auto_range)
        # register callback to selector of live_plot
        self.live_plot.controller.register_after_task(func=self.register_done_state)
        self.time_start = time.time()
        self.print_log(f'Start {self.cls_name.name}')


    def on_stop(self):
        if self.stateui_manager.runstate in (self.stateui_manager.RunState.RUNNING,):
            self.live_plot.controller.stop()
            self.print_log(f'Stop {self.cls_name.name}')

    def on_load(self):
        if self.stateui_manager.runstate in (self.stateui_manager.RunState.RUNNING,):
            QMessageBox.warning(self, 'Error', f'Live plot is running, stop before start.')
            return
        from Confocal_GUIv2.logic import load
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Data File",
            get_figs_dir(),
            "Data Files (*.npz *.jpg);;All Files (*)"
        )
        if file_path:
            # Update default dir
            set_figs_dir(os.path.dirname(file_path))
            addr = file_path[:-4] + '*'
        else:
            addr = ''
        self._remember_and_disable_set_checks() # make sure disable set_x, set_y
        live_plot_load = load(addr=addr, fig=self.canvas.figure, is_GUI=False)
        if live_plot_load is None:
            self._restore_and_retrigger_set_checks() # restore set_x, set_y
            return
        else:
            self.address_str = addr
            self.stateui_manager.filestate = self.stateui_manager.FileState.LOAD
            self.live_plot = live_plot_load

        if self.live_plot.controller.measurement.__class__ is self.cls_name:
            self.live_plot.controller.register_after_task(func=self.register_auto_range)
            # register callback to selector of live_plot
            self.live_plot.controller.register_after_task(func=self._read_range)
            self.live_plot.controller.register_after_task(
                lambda controller, a=addr: self.register_load_done_state(addr=a, controller=controller)
            )
            self.live_plot.controller.register_after_task(func=self._read_range)
        else:
            self.live_plot.controller.register_after_task(func=lambda a=addr: self.register_load_unsynced_state(addr=a))

        self.print_log(f'{self.cls_name.name} load fig from {addr}')

    def on_load_pulse(self):
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getOpenFileName(self,'select',get_pulses_dir(),'data_figure (*.npz)',options=options)

        if fileName == '':
            return

        set_pulses_dir(os.path.dirname(fileName))
        self.pulse_file = fileName
        localname = self.pulse_file.replace('\\', '/').split('/')[-1].rsplit('.', 1)[0]
        self.btn_pulse.setText(f'{localname}')

    def on_clear_pulse(self):
        self.pulse_file = None
        self.btn_pulse.setText(f'Using current pulse')

    def _add_devices(self, layout):
        self.device_widgets = {}  # name -> widget
        from Confocal_GUIv2.device import get_devices
        _dm = get_devices()
        device_list = list(_dm._instances.keys()) if _dm is not None else []

        for name in self.cls_name.KEYS_FOR_DEVICES:
            default = self.get_default(name, None)

            w = QComboBox()
            w.addItems(device_list)
            w.setCurrentText(default)

            row = QHBoxLayout()
            row.setSpacing(8)
            lbl = QLabel(f'{name}:')
            lbl.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
            lbl.setFixedSize(110, 30)
            w.setFixedSize(100, 30)
            row.addWidget(lbl)
            row.addWidget(w)
            layout.addLayout(row)

            self.device_widgets[name] = (w, default)

        counter_widget = self.device_widgets.get('counter', None)
        if counter_widget is not None:
            counter_widget[0].currentTextChanged.connect(self.on_device_counter)
        self.on_device_counter()

    def _add_extra_settings(self, layout):
        self.extra_widgets = {}  # name -> widget
        for name in self.cls_name.KEYS_FOR_EXTRA:
            if name not in self.handled_in_control:
                default = self.get_default(name, None)
                w = QLineEdit()
                w.setText(python2str(default))


                row = QHBoxLayout()
                row.setSpacing(8)
                lbl = QLabel(f'{name}:')
                lbl.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
                lbl.setFixedSize(110, 30)
                w.setFixedSize(100, 30)
                row.addWidget(lbl)
                row.addWidget(w)
                layout.addLayout(row)
                self.extra_widgets[name] = (w, default)

    def on_settings(self):
        if self.settings_stack.isVisible():
            self.settings_stack.setVisible(False)
        else:
            self.settings_stack.setCurrentIndex(1)
            self.settings_stack.setVisible(True)
            self.settings_stack.raise_()

    def on_counter_mode_changed(self):
        if self.combo_counter_mode.currentText()=='apd_sample':
            if self.stack_exp.currentIndex()==0:
                self.stack_exp.setCurrentIndex(1)
                self.lbl_exp_mode.setText('Pts:')
        else:
            if self.stack_exp.currentIndex()==1:
                self.stack_exp.setCurrentIndex(0)
                self.lbl_exp_mode.setText('Exp:')


class LivePlot1DGUI(BaseLivePlotGUI):
    def __init__(self, cls_name=None, parent=None, overrides=None):
        super().__init__(parent=parent, cls_name=cls_name, handled_in_control={'x_array'}, overrides=overrides)
        self.MAX_LEN = 10
        # max len for display
        self.live_plot = None
        self.overhead_time = 0 
        # estimated 0ms overhead for single data point in 1D unless get estimate from last measurement
        self.time_start = 0
        self.pulse_file = None
        self.init_ui()
        self.check_validator_widgets = [self.edit_xmin, self.edit_xmax, self.edit_data_x]
        self.check_range_widgets = [self.spin_xstep, self.spin_exposure, self.spin_repeat, self.spin_sample_num]

        self.on_clear_pulse()
        self.load_default()


        self.check_set_x.toggled.connect(self.on_set_x_toggled)
        self.spin_set_x.valueChanged.connect(self.on_set_x_changed)   
        self.combo_counter_mode.currentTextChanged.connect(self.on_counter_mode_changed)
        self.check_enable_data_x.clicked.connect(self.on_enable_data_x)

        self.btn_pulse.clicked.connect(self.on_load_pulse)
        self.btn_clear.clicked.connect(self.on_clear_pulse)
        self.spin_xstep.valueChanged.connect(self.on_step_changed)
        self.edit_xmin.textChanged.connect(self.on_step_changed)
        self.spin_xstep.valueChanged.connect(self.estimate_run_time)
        self.spin_exposure.valueChanged.connect(self.estimate_run_time)
        self.spin_sample_num.valueChanged.connect(self.estimate_run_time)
        self.edit_xmin.textChanged.connect(self.estimate_run_time)
        self.edit_xmax.textChanged.connect(self.estimate_run_time)
        self.on_step_changed()
        self.estimate_run_time()

        self.timer_x = QtCore.QTimer(self)
        self.timer_x.setInterval(200)  # Interval in milliseconds
        self.timer_x.timeout.connect(self._read_device_x)
        self.timer_x.start()

        self.timer_estimate = QtCore.QTimer(self)
        self.timer_estimate.setInterval(1000)  # Interval in milliseconds
        self.timer_estimate.timeout.connect(self.estimate_run_time)
        self.timer_estimate.start()

    def on_enable_data_x(self):
        enabled = self.check_enable_data_x.isChecked()
        if enabled is False:
            self.stack_data_x.setCurrentIndex(0)
        else:
            self.stack_data_x.setCurrentIndex(1)

    def _read_range(self):
        if self.stateui_manager.runstate not in (self.stateui_manager.RunState.STOP,):
            return
        if self.live_plot.area.range[0] is None:
            xlim = self.live_plot.fig.axes[0].get_xlim()
            ylim = self.live_plot.fig.axes[0].get_ylim()
            xl, xh, yl, yh = np.min(xlim), np.max(xlim), np.min(ylim), np.max(ylim)
        else:
            xl, xh, yl, yh = self.live_plot.area.range
        
        new_xl, new_xh = np.sort([self.live_plot.data_figure.transform_back(xl), 
            self.live_plot.data_figure.transform_back(xh)])
        self.edit_xmin.setText(float2str(new_xl, length=self.MAX_LEN))
        self.edit_xmax.setText(float2str(new_xh, length=self.MAX_LEN))

    def _read_x(self):
        if self.stateui_manager.runstate not in (self.stateui_manager.RunState.STOP,):
            return
        if self.live_plot.data_figure.cross.xy is None:
            return
        else:
            x = self.live_plot.data_figure.cross.xy[0]
            self.spin_set_x.setValue(self.live_plot.data_figure.transform_back(x))

    def _read_device_x(self):
        if self.stateui_manager.runstate not in (self.stateui_manager.RunState.STOP, self.stateui_manager.RunState.RUNNING):
            return
        x = self.live_plot.controller.measurement.read_x()
        self.edit_x.setText(float2str(x, length=self.MAX_LEN))
        # when liveplot is running, this can add extra overhead

    def load_default(self):
        # pull defaults
        data_x_default = self.get_default('data_x', None)
        x_array_default = self.get_default('x_array', None)
        exposure_default = self.get_default('exposure', 0.1)
        sample_num_default = self.get_default('sample_num', 1000)


        # set spinboxes only when default is present
        if x_array_default is not None:
            # x_array_default is an array-like: [min, ..., max]
            self.edit_xmin.setText(float2str(x_array_default[0], length=self.MAX_LEN))
            self.edit_xmax.setText(float2str(x_array_default[-1], length=self.MAX_LEN))
            self.spin_xstep.setValue(x_array_default[1] - x_array_default[0])
            self.spin_xstep.setSingleStep((x_array_default[1] - x_array_default[0])/2)
            self.spin_set_x.setSingleStep(x_array_default[1] - x_array_default[0])
            self.spin_set_x.setValue(x_array_default[0])
        if data_x_default is not None:
            self.edit_data_x.setText(python2str(data_x_default))

        self.spin_exposure.setRange(0.01, 1000)
        self.spin_exposure.setSingleStep(0.1)
        self.spin_exposure.setValue(exposure_default)
        self.spin_sample_num.setRange(1, 10000000)
        self.spin_sample_num.setSingleStep(100)
        self.spin_sample_num.setValue(sample_num_default)


    def read_params(self):
        if self.check_enable_data_x.isChecked():
            data_x = str2python(self.edit_data_x.text())
            x_array = None
        else:
            x_array_min = str2python(self.edit_xmin.text())
            x_array_max = str2python(self.edit_xmax.text())
            x_array_step = self.spin_xstep.value()
            x_array = np.arange(x_array_min, x_array_max+x_array_step, x_array_step)
            data_x = None
        exposure = self.spin_exposure.value()
        sample_num = self.spin_sample_num.value()
        pulse_file = self.pulse_file if self.check_pulse.isChecked() else None
        data_x_generated = self.cls_name.generate_data_x(x_array=x_array, data_x=data_x)


        params_dict_base = self.read_params_base()
        params_dict = {'data_x':data_x, 'x_array':x_array, 'exposure':exposure, 'sample_num':sample_num, 'pulse_file':pulse_file,
        'data_x_generated':data_x_generated}

        return {**params_dict, **params_dict_base}

    def on_step_changed(self):
        unit = self.cls_name.unit
        if unit=='nm':
            spl = 299792458
            x_array_min = str2python(self.edit_xmin.text())
            x_array_max = str2python(self.edit_xmax.text())
            x_array_step = self.spin_xstep.value()
            try:
                step_in_MHz = 1000*np.abs(spl/(x_array_min+x_array_step) - spl/x_array_min)
            except:
                step_in_MHz = 0
            self.edit_display_step.setText(f'{step_in_MHz:.2f}MHz')
        else:
            self.edit_display_step.setText(f'')


    def init_ui(self):

        # 5) Parameter input row
        self.group_control = QGroupBox(f'Control params (frequently used), unit in {self.cls_name.unit}')
        layout = QVBoxLayout(self.group_control)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # first row
        control_layout = QHBoxLayout()
        control_layout.setSpacing(4)
        self.edit_xmin = FloatLineEdit()
        self.edit_xmax = FloatLineEdit()
        self.check_set_x = QCheckBox()
        self.spin_set_x = QDoubleSpinBox(length=self.MAX_LEN, allow_minus=True)
        self.label_x = QLabel()
        self.edit_x = FloatLineEdit()

        widget0 = QWidget()
        hbox = QHBoxLayout(widget0)
        hbox.setContentsMargins(0, 0, 0, 0)
        hbox.setSpacing(4)
        params = [
            ("Min:", self.edit_xmin),
            ("Max:", self.edit_xmax),
        ]
        for label_text, edit in params:
            lbl = QLabel(label_text)
            lbl.setFixedSize(45, 30)
            lbl.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
            edit.setFixedSize(100, 30)
            edit.setMaxLength(self.MAX_LEN)
            hbox.addWidget(lbl)
            hbox.addWidget(edit)

        widget1 = QWidget()
        hbox = QHBoxLayout(widget1)
        hbox.setContentsMargins(0, 0, 0, 0)
        hbox.setSpacing(4)
        lbl = QLabel('data_x:')
        lbl.setFixedSize(45, 30)
        lbl.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
        self.edit_data_x = DataXLineEdit(n_dim=1)
        self.edit_data_x.setFixedHeight(30)
        self.edit_data_x.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        hbox.addWidget(lbl)
        hbox.addWidget(self.edit_data_x)

        self.stack_data_x = QStackedWidget(shadow=False)
        self.stack_data_x.setFixedSize(302, 30)
        self.stack_data_x.addWidget(widget0) # index 0
        self.stack_data_x.addWidget(widget1)    # index 1
        self.stack_data_x.setCurrentIndex(0)
        control_layout.addWidget(self.stack_data_x)


        control_layout.addSpacing(20)
        self.check_set_x.setText('Set X')
        self.check_set_x.setFixedSize(60, 30)
        self.spin_set_x.setFixedSize(100+22, 30)
        control_layout.addWidget(self.check_set_x)
        control_layout.addWidget(self.spin_set_x)

        self.label_x.setText('X:')
        self.label_x.setFixedSize(15, 30)
        self.label_x.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
        self.edit_x.setFixedSize(100, 30)
        self.edit_x.setMaxLength(self.MAX_LEN)
        self.edit_x.setEnabled(False)
        control_layout.addStretch()
        control_layout.addWidget(self.label_x)
        control_layout.addWidget(self.edit_x)

        control_layout.setAlignment(Qt.AlignLeft)
        layout.addLayout(control_layout)
     
        # second row
        control_layout = QHBoxLayout()
        control_layout.setSpacing(4)
        self.spin_xstep = QDoubleSpinBox(length=self.MAX_LEN-2, allow_minus=False)
        params = [
            ("Step:", self.spin_xstep),
        ]
        for label_text, spin in params:
            lbl = QLabel(label_text)
            lbl.setFixedSize(45, 30)
            lbl.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
            spin.setFixedSize(100, 30)
            control_layout.addWidget(lbl)
            control_layout.addWidget(spin)


        self.lbl_exp_mode = QLabel('Exp:')
        self.lbl_exp_mode.setFixedSize(45, 30)
        self.lbl_exp_mode.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
        self.spin_exposure = QDoubleSpinBox(length=8, allow_minus=False)
        self.spin_sample_num = QDoubleSpinBox(length=8, allow_minus=False)
        self.stack_exp = QStackedWidget(shadow=False)
        self.stack_exp.setFixedSize(100, 30)
        self.stack_exp.addWidget(self.spin_exposure)      # index 0
        self.stack_exp.addWidget(self.spin_sample_num)    # index 1
        self.stack_exp.setCurrentIndex(0)
        self.on_counter_mode_changed()
        control_layout.addWidget(self.lbl_exp_mode)
        control_layout.addWidget(self.stack_exp)


        control_layout.addSpacing(20)
        self.check_pulse = QCheckBox()
        self.btn_pulse = QPushButton()
        self.check_pulse.setText('Pulse')
        self.check_pulse.setFixedSize(60, 30)
        self.btn_pulse.setFixedHeight(30)
        self.btn_pulse.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.btn_clear = QPushButton()
        self.btn_clear.setText('X')
        self.btn_clear.setFixedSize(30, 30)
        pulse_layout = QHBoxLayout()
        pulse_layout.setSpacing(1)
        pulse_layout.setContentsMargins(0, 0, 0, 0)
        pulse_layout.addWidget(self.btn_pulse)
        pulse_layout.addWidget(self.btn_clear)
        control_layout.addWidget(self.check_pulse)
        control_layout.addLayout(pulse_layout)

        control_layout.setAlignment(Qt.AlignLeft)
        layout.addLayout(control_layout)



        # third row
        control_layout = QHBoxLayout()
        control_layout.setSpacing(4)

        lbl = QLabel('Step:')
        lbl.setFixedSize(45, 30)
        lbl.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
        control_layout.addWidget(lbl)

        self.edit_display_step = QLineEdit('0MHz')
        self.edit_display_step.setFixedSize(100, 30)
        self.edit_display_step.setEnabled(False)
        control_layout.addWidget(self.edit_display_step)

        self.edit_display_time = QLineEdit()
        self.edit_display_time.setFixedHeight(30)
        self.edit_display_time.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        self.edit_display_time.setEnabled(False)
        control_layout.addWidget(self.edit_display_time)

        control_layout.setAlignment(Qt.AlignLeft)
        layout.addLayout(control_layout)



        self.main_layout.addWidget(self.group_control)


class LivePlot2DGUI(BaseLivePlotGUI):
    def __init__(self, cls_name=None, parent=None, overrides=None):
        super().__init__(parent=parent, cls_name=cls_name, handled_in_control={'x_array', 'y_array', 'bound_pts'}, overrides=overrides)
        self.MAX_LEN = 6
        # max length for display
        self.live_plot = None
        self.overhead_time = 0 
        self.is_load = False
        # estimated 0ms overhead for single data point in 2D unless get estimate from last measurement
        self.time_start = 0
        self.pulse_file = None
        self.init_ui()

        self.check_validator_widgets = [self.edit_xmin, self.edit_xmax, self.edit_ymin, self.edit_ymax, self.edit_bound_pts, self.edit_data_x]
        self.check_range_widgets = [self.spin_xstep, self.spin_ystep, self.spin_exposure, self.spin_repeat, self.spin_sample_num]

        self.on_clear_pulse()
        self.load_default()


        self.check_set_x.toggled.connect(self.on_set_x_toggled)
        self.spin_set_x.valueChanged.connect(self.on_set_x_changed)
        self.check_set_y.toggled.connect(self.on_set_y_toggled)
        self.spin_set_y.valueChanged.connect(self.on_set_y_changed)   
        self.combo_counter_mode.currentTextChanged.connect(self.on_counter_mode_changed)
        self.check_enable_data_x.clicked.connect(self.on_enable_data_x)

        self.btn_pulse.clicked.connect(self.on_load_pulse)
        self.btn_clear.clicked.connect(self.on_clear_pulse)
        self.btn_pts_clear.clicked.connect(self.on_pts_clear)

        self.spin_xstep.valueChanged.connect(self.estimate_run_time)
        self.edit_xmin.textChanged.connect(self.estimate_run_time)
        self.edit_xmax.textChanged.connect(self.estimate_run_time)
        self.spin_ystep.valueChanged.connect(self.estimate_run_time)
        self.edit_ymin.textChanged.connect(self.estimate_run_time)
        self.edit_ymax.textChanged.connect(self.estimate_run_time)
        self.edit_bound_pts.textChanged.connect(self.estimate_run_time)

        self.spin_exposure.valueChanged.connect(self.estimate_run_time)
        self.spin_sample_num.valueChanged.connect(self.estimate_run_time)
        self.estimate_run_time()

        self.timer_x = QtCore.QTimer(self)
        self.timer_x.setInterval(200)  # Interval in milliseconds
        self.timer_x.timeout.connect(self._read_device_x)
        self.timer_x.start()

        self.timer_estimate = QtCore.QTimer(self)
        self.timer_estimate.setInterval(1000)  # Interval in milliseconds
        self.timer_estimate.timeout.connect(self.estimate_run_time)
        self.timer_estimate.start()

    def on_enable_data_x(self):
        enabled = self.check_enable_data_x.isChecked()
        if enabled is False:
            self.stack_data_x.setCurrentIndex(0)
            self.stack_data_x_below.setCurrentIndex(0)
        else:
            self.stack_data_x.setCurrentIndex(1)
            self.stack_data_x_below.setCurrentIndex(1)

    def _prefill_from_loaded_info(self):
        # one time load called in register_load_done_state in Base
        caller_kwargs = self.live_plot.controller.measurement.info.get('caller_kwargs_for_save', {}) or {}
        bound_pts = caller_kwargs.get('bound_pts', None)
        if bound_pts is not None:
            pts_str = ', '.join(
                f'({pt[0]}, {pt[1]})' for pt in bound_pts
            )
            self.edit_bound_pts.setText(pts_str)

    def _read_range(self):
        if self.stateui_manager.runstate not in (self.stateui_manager.RunState.STOP,):
            return

        if self.live_plot.area.range[0] is None:
            xlim = self.live_plot.fig.axes[0].get_xlim()
            ylim = self.live_plot.fig.axes[0].get_ylim()
            xl, xh, yl, yh = np.min(xlim), np.max(xlim), np.min(ylim), np.max(ylim)
        else:
            xl, xh, yl, yh = self.live_plot.area.range

        self.edit_xmin.setText(float2str(self.live_plot.data_figure._align_to_grid(xl, type='x'), length=self.MAX_LEN))
        self.edit_xmax.setText(float2str(self.live_plot.data_figure._align_to_grid(xh, type='x'), length=self.MAX_LEN))
        self.edit_ymin.setText(float2str(self.live_plot.data_figure._align_to_grid(yl, type='y'), length=self.MAX_LEN))
        self.edit_ymax.setText(float2str(self.live_plot.data_figure._align_to_grid(yh, type='y'), length=self.MAX_LEN))

    def _read_x(self):
        if self.stateui_manager.runstate not in (self.stateui_manager.RunState.STOP,):
            return
        if self.live_plot.data_figure.cross.xy is None:
            return
        else:
            _xy = self.live_plot.data_figure.cross.xy #cross selector        
            self.spin_set_x.setValue(_xy[0])
            self.spin_set_y.setValue(_xy[1])

        if self.check_pts_add.isChecked():
            self._add_bound_point()

    def _add_bound_point(self):
        xy = self.live_plot.data_figure.cross.xy
        if xy is None:
            return
        x_str = float2str(self.live_plot.data_figure._align_to_grid(xy[0], type='x'), length=self.MAX_LEN)
        y_str = float2str(self.live_plot.data_figure._align_to_grid(xy[1], type='y'), length=self.MAX_LEN)
        pt = f'({x_str}, {y_str})'
        curr = self.edit_bound_pts.text().strip()
        new = pt if not curr else f'{curr}, {pt}'
        self.edit_bound_pts.setText(new)


    def _read_device_x(self):
        if self.stateui_manager.runstate not in (self.stateui_manager.RunState.STOP, self.stateui_manager.RunState.RUNNING):
            return
        x, y = self.live_plot.controller.measurement.read_x()
        self.edit_x.setText(float2str(x, length=self.MAX_LEN))
        self.edit_y.setText(float2str(y, length=self.MAX_LEN))
        # when liveplot is running, this can add extra overhead


    def load_default(self):
        # pull defaults
        x_array_default = self.get_default('x_array', None)
        y_array_default = self.get_default('y_array', None)
        bound_pts_default = self.get_default('bound_pts', None)
        data_x_default = self.get_default('data_x', None)

        exposure_default = self.get_default('exposure', 0.1)
        sample_num_default = self.get_default('sample_num', 1000)

        # set spinboxes only when default is present
        if x_array_default is not None:
            self.edit_xmin.setText(float2str(x_array_default[0], length=self.MAX_LEN))
            self.edit_xmax.setText(float2str(x_array_default[-1], length=self.MAX_LEN))
            self.spin_xstep.setValue(x_array_default[1] - x_array_default[0])
            self.spin_xstep.setSingleStep((x_array_default[1] - x_array_default[0])/2)
            self.spin_set_x.setSingleStep(x_array_default[1] - x_array_default[0])
            self.spin_set_x.setValue(x_array_default[0])

        if y_array_default is not None:
            self.edit_ymin.setText(float2str(y_array_default[0], length=self.MAX_LEN))
            self.edit_ymax.setText(float2str(y_array_default[-1], length=self.MAX_LEN))
            self.spin_ystep.setValue(y_array_default[1] - y_array_default[0])
            self.spin_ystep.setSingleStep((y_array_default[1] - y_array_default[0])/2)
            self.spin_set_y.setSingleStep(y_array_default[1] - y_array_default[0])
            self.spin_set_y.setValue(y_array_default[0])

        if bound_pts_default is not None:
            pts_str = ', '.join(
                f'({pt[0]}, {pt[1]})' for pt in bound_pts_default
            )
            self.edit_bound_pts.setText(pts_str)

        if data_x_default is not None:
            self.edit_data_x.setText(python2str(data_x_default))

        self.spin_exposure.setRange(0.01, 1000)
        self.spin_exposure.setSingleStep(0.1)
        self.spin_exposure.setValue(exposure_default)

        self.spin_sample_num.setRange(1, 10000000)
        self.spin_sample_num.setSingleStep(100)
        self.spin_sample_num.setValue(sample_num_default)


    def read_params(self):

        if self.check_enable_data_x.isChecked():
            data_x = str2python(self.edit_data_x.text())
            x_array = None
            y_array = None
            bound_pts = None
        else:
            x_array_min = str2python(self.edit_xmin.text())
            x_array_max = str2python(self.edit_xmax.text())
            x_array_step = self.spin_xstep.value()
            x_array = np.arange(x_array_min, x_array_max+x_array_step, x_array_step)
            y_array_min = str2python(self.edit_ymin.text())
            y_array_max = str2python(self.edit_ymax.text())
            y_array_step = self.spin_ystep.value()
            y_array = np.arange(y_array_min, y_array_max+y_array_step, y_array_step)
            bound_pts = str2python('[' + self.edit_bound_pts.text() + ']')
            data_x = None

        data_x_generated = self.cls_name.generate_data_x(x_array=x_array, y_array=y_array, bound_pts=bound_pts, data_x=data_x)

        exposure = self.spin_exposure.value()
        sample_num = self.spin_sample_num.value()
        pulse_file = self.pulse_file if self.check_pulse.isChecked() else None

        params_dict_base = self.read_params_base()
        params_dict = {'x_array':x_array, 'y_array':y_array, 'bound_pts':bound_pts, 'exposure':exposure, 'sample_num':sample_num, 
        'pulse_file':pulse_file, 'data_x':data_x, 'data_x_generated':data_x_generated}


        return {**params_dict, **params_dict_base}


    def on_pts_clear(self):
        self.edit_bound_pts.setText(f'')


    def init_ui(self):
        # 5) Parameter input row
        self.group_control = QGroupBox(f'Control params (frequently used), unit in {self.cls_name.unit}')
        layout = QVBoxLayout(self.group_control)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # first row
        control_layout = QHBoxLayout()
        control_layout.setSpacing(4)
        self.edit_xmin = FloatLineEdit()
        self.edit_xmax = FloatLineEdit()
        self.spin_xstep = QDoubleSpinBox(length=self.MAX_LEN-2, allow_minus=False)
        self.check_set_x = QCheckBox()
        self.spin_set_x = QDoubleSpinBox(length=self.MAX_LEN, allow_minus=True)
        self.label_x = QLabel()
        self.edit_x = FloatLineEdit()

        widget0 = QWidget()
        hbox = QHBoxLayout(widget0)
        hbox.setContentsMargins(0, 0, 0, 0)
        hbox.setSpacing(4)        
        params = [
            ("Min:", self.edit_xmin),
            ("Max:", self.edit_xmax),
            ("Step:", self.spin_xstep),
        ]
        for label_text, edit in params:
            lbl = QLabel(label_text)
            lbl.setFixedSize(45, 30)
            lbl.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
            edit.setFixedSize(70, 30)
            hbox.addWidget(lbl)
            hbox.addWidget(edit)
        self.edit_xmin.setMaxLength(self.MAX_LEN)
        self.edit_xmax.setMaxLength(self.MAX_LEN)

        widget1 = QWidget()
        hbox = QHBoxLayout(widget1)
        hbox.setContentsMargins(0, 0, 0, 0)
        hbox.setSpacing(4)
        lbl = QLabel('data_x:')
        lbl.setFixedSize(45, 30)
        lbl.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
        self.edit_data_x = DataXLineEdit(n_dim=2)
        self.edit_data_x.setFixedHeight(30)
        self.edit_data_x.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        hbox.addWidget(lbl)
        hbox.addWidget(self.edit_data_x)

        self.stack_data_x = QStackedWidget(shadow=False)
        self.stack_data_x.setFixedSize(365, 30)
        self.stack_data_x.addWidget(widget0) # index 0
        self.stack_data_x.addWidget(widget1)    # index 1
        self.stack_data_x.setCurrentIndex(0)
        control_layout.addWidget(self.stack_data_x)


        control_layout.addStretch()
        self.check_set_x.setText('Set X')
        self.check_set_x.setFixedSize(60, 30)
        self.spin_set_x.setFixedSize(70+22, 30)
        control_layout.addWidget(self.check_set_x)
        control_layout.addWidget(self.spin_set_x)

        self.label_x.setText('X:')
        self.label_x.setFixedSize(15, 30)
        self.label_x.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
        self.edit_x.setFixedSize(70, 30)
        self.edit_x.setMaxLength(self.MAX_LEN)
        self.edit_x.setEnabled(False)
        control_layout.addWidget(self.label_x)
        control_layout.addWidget(self.edit_x)

        control_layout.setAlignment(Qt.AlignLeft)
        layout.addLayout(control_layout)

        # second row
        control_layout = QHBoxLayout()
        control_layout.setSpacing(4)
        self.edit_ymin = FloatLineEdit()
        self.edit_ymax = FloatLineEdit()
        self.spin_ystep = QDoubleSpinBox(length=self.MAX_LEN-2, allow_minus=False)
        self.check_set_y = QCheckBox()
        self.spin_set_y = QDoubleSpinBox(length=self.MAX_LEN, allow_minus=True)
        self.label_y = QLabel()
        self.edit_y = FloatLineEdit()

        widget0 = QWidget()
        hbox = QHBoxLayout(widget0)
        hbox.setContentsMargins(0, 0, 0, 0)
        hbox.setSpacing(4)
        params = [
            ("Min:", self.edit_ymin),
            ("Max:", self.edit_ymax),
            ("Step:", self.spin_ystep),

        ]
        for label_text, edit in params:
            lbl = QLabel(label_text)
            lbl.setFixedSize(45, 30)
            lbl.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
            edit.setFixedSize(70, 30)
            hbox.addWidget(lbl)
            hbox.addWidget(edit)
        self.edit_ymin.setMaxLength(self.MAX_LEN)
        self.edit_ymax.setMaxLength(self.MAX_LEN)

        widget1 = QWidget()
        hbox = QHBoxLayout(widget1)
        hbox.setContentsMargins(0, 0, 0, 0)
        hbox.setSpacing(4)

        self.stack_data_x_below = QStackedWidget(shadow=False)
        self.stack_data_x_below.setFixedSize(365, 30)
        self.stack_data_x_below.addWidget(widget0) # index 0
        self.stack_data_x_below.addWidget(widget1)    # index 1
        self.stack_data_x_below.setCurrentIndex(0)
        control_layout.addWidget(self.stack_data_x_below)


        control_layout.addStretch()
        self.check_set_y.setText('Set Y')
        self.check_set_y.setFixedSize(60, 30)
        self.spin_set_y.setFixedSize(70+22, 30)
        control_layout.addWidget(self.check_set_y)
        control_layout.addWidget(self.spin_set_y)

        self.label_y.setText('Y:')
        self.label_y.setFixedSize(15, 30)
        self.label_y.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
        self.edit_y.setFixedSize(70, 30)
        self.edit_y.setMaxLength(self.MAX_LEN)
        self.edit_y.setEnabled(False)
        control_layout.addWidget(self.label_y)
        control_layout.addWidget(self.edit_y)

        control_layout.setAlignment(Qt.AlignLeft)
        layout.addLayout(control_layout)
     
        # third row
        control_layout = QHBoxLayout()
        control_layout.setSpacing(4)

        self.lbl_exp_mode = QLabel('Exp:')
        self.lbl_exp_mode.setFixedSize(45, 30)
        self.lbl_exp_mode.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
        self.spin_exposure = QDoubleSpinBox(length=8, allow_minus=False)
        self.spin_sample_num = QDoubleSpinBox(length=8, allow_minus=False)
        self.stack_exp = QStackedWidget(shadow=False)
        self.stack_exp.setFixedSize(100, 30)
        self.stack_exp.addWidget(self.spin_exposure)      # index 0
        self.stack_exp.addWidget(self.spin_sample_num)    # index 1
        self.stack_exp.setCurrentIndex(0)
        self.on_counter_mode_changed()
        control_layout.addWidget(self.lbl_exp_mode)
        control_layout.addWidget(self.stack_exp)

        control_layout.addSpacing(20)
        self.check_pulse = QCheckBox()
        self.btn_pulse = QPushButton()
        self.check_pulse.setText('Pulse')
        self.check_pulse.setFixedSize(60, 30)
        self.btn_pulse.setFixedHeight(30)
        self.btn_pulse.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.btn_clear = QPushButton()
        self.btn_clear.setText('X')
        self.btn_clear.setFixedSize(30, 30)
        pulse_layout = QHBoxLayout()
        pulse_layout.setSpacing(1)
        pulse_layout.setContentsMargins(0, 0, 0, 0)
        pulse_layout.addWidget(self.btn_pulse)
        pulse_layout.addWidget(self.btn_clear)
        control_layout.addWidget(self.check_pulse)
        control_layout.addLayout(pulse_layout)

        control_layout.setAlignment(Qt.AlignLeft)
        layout.addLayout(control_layout)

        # fourth row
        control_layout = QHBoxLayout()
        control_layout.setSpacing(4)

        lbl = QLabel("Bound pts:")
        lbl.setFixedSize(80, 30)
        lbl.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)


        self.edit_bound_pts = CoordinateListLineEdit()
        self.edit_bound_pts.setFixedHeight(30)
        self.edit_bound_pts.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)


        self.btn_pts_clear = QPushButton()
        self.btn_pts_clear.setText('X')
        self.btn_pts_clear.setFixedSize(30, 30)
        self.check_pts_add = QCheckBox()
        self.check_pts_add.setText('Add')
        self.check_pts_add.setFixedSize(60, 30)
        pts_layout = QHBoxLayout()
        pts_layout.setSpacing(1)
        pts_layout.setContentsMargins(0, 0, 0, 0)
        pts_layout.addWidget(self.edit_bound_pts)
        pts_layout.addWidget(self.check_pts_add)
        pts_layout.addWidget(self.btn_pts_clear)

        control_layout.addWidget(lbl)
        control_layout.addLayout(pts_layout)


        control_layout.setAlignment(Qt.AlignLeft)
        layout.addLayout(control_layout)

        # fifth row
        control_layout = QHBoxLayout()
        control_layout.setSpacing(4)


        self.edit_display_time = QLineEdit()
        self.edit_display_time.setFixedHeight(30)
        self.edit_display_time.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.edit_display_time.setEnabled(False)
        control_layout.addWidget(self.edit_display_time)

        control_layout.setAlignment(Qt.AlignLeft)
        layout.addLayout(control_layout)


        self.main_layout.addWidget(self.group_control)

