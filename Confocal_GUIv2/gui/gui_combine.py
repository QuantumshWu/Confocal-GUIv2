from .gui_device import *
from .gui_individual import LivePlot1DGUI, LivePlot2DGUI
from .gui_task import run_task_panel

class LivePlotMainGUI(QWidget):

    def __init__(self, cls_name_left=None, cls_name_right=None, parent=None):
        super().__init__(parent)
        self._plots = []
        self._showed_widgets = []
        self._init_ui()
        self._load_measurement_classes()
        if cls_name_left is not None:
            # a reference to class
            self.add_tab(self.tabs_left, self.combo_cls_left, meas_cls=cls_name_left)
        if cls_name_right is not None:
            # a reference to class
            self.add_tab(self.tabs_right, self.combo_cls_right, meas_cls=cls_name_right)


    def _init_ui(self):
        # Instantiate combo boxes and add buttons before layouts
        self.combo_cls_left = QComboBox()
        self.combo_cls_left.setFixedSize(100, 30)
        self.btn_add_left = QPushButton("Add plot")
        self.btn_add_left.setFixedSize(120, 30)
        # 1) Add the QTabWidget into the grid at (0,0)
        self.tabs_left = QTabWidget()
        self.tabs_left.setTabsClosable(True)
        self.tabs_left.setFixedSize(700, 800)

        self.combo_cls_right = QComboBox()
        self.combo_cls_right.setFixedSize(100, 30)
        self.btn_add_right = QPushButton("Add plot")
        self.btn_add_right.setFixedSize(120, 30)
        # 1) Add the QTabWidget into the grid at (0,0)
        self.tabs_right = QTabWidget()
        self.tabs_right.setTabsClosable(True)
        self.tabs_right.setFixedSize(700, 800)

        # Main vertical layout for the widget
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(18,8,18,8)
        main_layout.setSpacing(8)


        # Horizontal split: left and right panels
        main_hbox = QHBoxLayout()
        main_hbox.setContentsMargins(0,0,0,0)
        main_hbox.setSpacing(16)
        main_layout.addLayout(main_hbox)

        self.init_tab_and_btn(main_hbox, self.tabs_left, self.combo_cls_left, self.btn_add_left)
        self.init_tab_and_btn(main_hbox, self.tabs_right, self.combo_cls_right, self.btn_add_right)

        control_frame = QFrame()
        control_vbox = QVBoxLayout(control_frame)
        control_vbox.setContentsMargins(8, 8, 8, 8)
        control_vbox.setSpacing(8)
        main_layout.addWidget(control_frame)

        control_hbox = QHBoxLayout()
        control_hbox.setContentsMargins(0,0,0,0)
        control_hbox.setSpacing(4)
        lbl = QLabel('Save:')
        lbl.setFixedSize(35, 30)
        lbl.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
        control_hbox.addWidget(lbl)
        self.edit_save = QLineEdit()
        self.edit_save.setFixedSize(350, 30)
        control_hbox.addWidget(self.edit_save)
        self.btn_device = QPushButton('Set device')
        self.btn_device.setFixedSize(100, 30)
        control_hbox.addWidget(self.btn_device)
        self.combo_device = QComboBox()
        self.combo_device.setFixedSize(120, 30)
        control_hbox.addWidget(self.combo_device)

        self.btn_device.clicked.connect(self.on_set_device)

        control_hbox.addStretch()

        lbl = QLabel('On Start')
        lbl.setFixedSize(60, 30)
        control_hbox.addWidget(lbl)
        self.check_on_start = TriStateToggleSwitch(labels = ['Replace', 'Block', 'Parallel'])
        self.check_on_start.setFixedSize(200, 30)
        control_hbox.addWidget(self.check_on_start)
        lbl = QLabel('other plots')
        lbl.setFixedSize(80, 30)
        control_hbox.addWidget(lbl)


        control_hbox.addSpacing(150)
        # Add "Run Task" button
        self.btn_task = QPushButton('Task Panel')
        self.btn_task.setFixedSize(100, 30)
        control_hbox.addWidget(self.btn_task)
        self.btn_task.clicked.connect(self.on_task_button_clicked)


        control_vbox.addLayout(control_hbox)

        log_hbox = QHBoxLayout()
        log_hbox.setContentsMargins(0,0,0,0)
        log_hbox.setSpacing(4)
        lbl = QLabel('Log:')
        lbl.setFixedSize(35, 30)
        lbl.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
        log_hbox.addWidget(lbl)
        self.edit_log = QLineEdit()
        self.edit_log.setFixedHeight(30)
        self.edit_log.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        log_hbox.addWidget(self.edit_log)
        control_vbox.addLayout(log_hbox)

        current_time = time.localtime()
        current_date = time.strftime("%Y-%m-%d", current_time)
        time_str = current_date.replace('-', '_')
        self.edit_save.setText(f'{time_str}/')
        # set save address to date of today

    def any_task_running(self) -> bool:
        runners = getattr(self, "_task_runners", [])
        return any(r.isRunning() for r in runners)

    def _on_task_running_changed(self, running: bool):
        """Slot for TaskRunner.sig_running to update button state."""
        if self.any_task_running():
            self.btn_task.set_color(FLUENT_GREEN)
        else:
            self.btn_task.set_color(ACCENT_COLOR)

    def on_task_button_clicked(self):
        self.task_panel = run_task_panel(self)

    def stop_all_tasks(self):
        """Request stop for all active runners (non-blocking)."""
        runners = getattr(self, "_task_runners", [])
        for r in list(runners):
            try:
                r.request_stop()
            except Exception:
                pass


    def on_set_device(self):
        device_name = self.combo_device.currentText()
        from Confocal_GUIv2.device import get_devices
        if device_name != '':
            device = get_devices(device_name)
            showed_widget = device.gui(in_GUI=True)
            self._showed_widgets.append(showed_widget)
            showed_widget.destroyed.connect(lambda _, w=showed_widget: self._on_child_closed(w))
        # allow open multiple child widgets but make sure close them

    def hideEvent(self, event):
        win = self.window()
        if getattr(win, '_hiding_via_close', False):
            self.stop_all_tasks()
            if getattr(self, 'task_panel', None) is not None:
                self.task_panel.close()
            for showed_widget in self._showed_widgets:
                showed_widget.close()
        super().hideEvent(event)


    def _on_child_closed(self, showed_widget):
        if showed_widget in self._showed_widgets:
            self._showed_widgets.remove(showed_widget)


    def init_tab_and_btn(self, main_hbox, tab_widget, combo_widget, btn_widget):
        # ── LEFT PANEL ──
        # Create a plain QWidget as container
        container = QWidget()
        # Use a grid layout on this container to overlay widgets
        grid = QGridLayout(container)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(0)

        # Prevent collapse when no tabs are present
        vbox = QVBoxLayout()
        vbox.setContentsMargins(0, 0, 0, 0)
        vbox.setSpacing(0)
        vbox.addWidget(tab_widget)
        vbox.addStretch()
        grid.addLayout(vbox, 0, 0)

        # 2) Create an HBoxLayout for overlaying the combo+button
        combo_btn = QHBoxLayout()
        combo_btn.setContentsMargins(0, 0, 0, 0)
        combo_btn.setSpacing(8)
        combo_btn.addWidget(combo_widget)
        combo_btn.addWidget(btn_widget)
        horizontal_offset = 0
        hbox = QHBoxLayout()
        hbox.setContentsMargins(0, 0, 0, 0)
        hbox.addStretch()
        hbox.addLayout(combo_btn)
        hbox.addSpacing(horizontal_offset)
        vertical_offset = 0
        vbox = QVBoxLayout()
        vbox.setContentsMargins(0, 0, 0, 0)
        vbox.addSpacing(vertical_offset)
        vbox.addLayout(hbox)
        vbox.addStretch()
        grid.addLayout(vbox, 0, 0)
        # 3) Add the container widget into the main hbox
        main_hbox.addWidget(container)

        # ── SIGNAL CONNECTIONS ──
        btn_widget.clicked.connect(
            lambda: self.add_tab(tab_widget, combo_widget)
        )
        tab_widget.tabCloseRequested.connect(
            lambda idx: self.close_tab(idx, tab_widget)
        )


    def _load_measurement_classes(self):
        from Confocal_GUIv2.logic import BaseMeasurement
        from Confocal_GUIv2.device import get_devices
        dm = get_devices()
        lookup = getattr(dm, "_lookup", {})

        items = []
        for name, obj in list(lookup.items()):
            # ple/odmr/pl/live
            if inspect.ismethod(obj):
                owner = getattr(obj, "__self__", None)
                if isinstance(owner, type) and issubclass(owner, BaseMeasurement):
                    items.append((name, obj))

        items.sort(key=lambda t: t[0].lower())

        if lookup!={}:
            self.combo_cls_left.clear()
            self.combo_cls_right.clear()
        for name, caller in items:
            self.combo_cls_left.addItem(name, caller)
            self.combo_cls_right.addItem(name, caller)


        device_list = list(dm._instances.keys()) if dm is not None else []
        if device_list!=[]:
            self.combo_device.clear()
        self.combo_device.addItems(device_list)
        if 'pulse' in device_list:
            self.combo_device.setCurrentText('pulse')

    def add_tab(self, tab_widget, combo_widget, meas_cls=None, ov=None, origin='user'):
        SOFT_MAX_LEFT = 2   # user clicks
        HARD_MAX_LEFT = 3   # tasks may exceed soft limit but not this
        HARD_MAX_RIGHT = 3

        if tab_widget is self.tabs_left:
            limit = SOFT_MAX_LEFT if origin == 'user' else HARD_MAX_LEFT
        else:
            limit = HARD_MAX_RIGHT

        if tab_widget.count() >= limit:
            QtWidgets.QMessageBox.warning(
                self,
                'Page limit',
                f'{limit} pages at most'
            )
            return False

        if meas_cls is None:
            meas_cls = combo_widget.currentData().__self__
        if '2D' in meas_cls.plotter:
            plot = LivePlot2DGUI(cls_name=meas_cls, parent=self, overrides=ov)
        else:
            plot = LivePlot1DGUI(cls_name=meas_cls, parent=self, overrides=ov)

        self._plots.append(plot)
        plot.adjustSize()
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0,0,0,0)
        layout.addWidget(plot)
        layout.addStretch()

        tab_widget.setMaximumSize(16777215, 16777215)
        tab_widget.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding,
            QtWidgets.QSizePolicy.Expanding
        )

        idx = tab_widget.addTab(container, '')
        tab_widget.setCurrentIndex(idx)
        tab_widget.adjustSize()
        tab_widget.setFixedSize(tab_widget.size())


        tab_widget.setTabText(idx, meas_cls.name)


        def _on_state_changed(new_state, tw=tab_widget, cont=container):
            i = tw.indexOf(cont)
            if i != -1:
                self._update_tab_icon(tw, i, new_state)

        plot.stateui_manager.runstateChanged.connect(_on_state_changed)
        _on_state_changed(plot.stateui_manager.runstate)

        return True

    def _update_tab_icon(self, tab_widget, index, state):

        name = getattr(state, 'name', '')
        color_map = {
            'INIT':     FLUENT_GREY,
            'RUNNING':  FLUENT_GREEN,
            'UNSYNCED': FLUENT_ORANGE,
            'STOP':     FLUENT_RED,
        }
        col = color_map.get(name)
        if col is None or index < 0:
            return

        size = 16
        pix = QPixmap(size, size)
        pix.fill(Qt.transparent)
        p = QPainter(pix)
        p.setRenderHint(QPainter.Antialiasing)
        p.setBrush(QColor(col))
        p.setPen(Qt.NoPen)
        p.drawEllipse(0, 0, size, size)
        p.end()

        icon = QIcon(pix)
        tab_widget.setTabIcon(index, icon)
        tab_widget.setIconSize(pix.size())



    def close_tab(self, index, tab_widget):
        container = tab_widget.widget(index)
        if not container:
            return

        plot = container.layout().itemAt(0).widget()
        if plot in self._plots:
            plot.close()
            self._plots.remove(plot)

        container.close()
        tab_widget.removeTab(index)

    def read_addr(self):
        return self.edit_save.text()

    def print_log(self, log_info):
        self.edit_log.setText(log_info)

    def any_is_running(self):
        return any(
            plot.stateui_manager.runstate is plot.stateui_manager.RunState.RUNNING
            for plot in self._plots
        )

    def stop_all(self):
        for plot in list(self._plots):
            plot.on_stop()

    def closeEvent(self, event):
        for plot in list(self._plots):
            plot.close()
        for showed_widget in list(self._showed_widgets):
            showed_widget.close()
        self._showed_widgets.clear()
        super().closeEvent(event)

    def showEvent(self, event):
        self._load_measurement_classes()
        super().showEvent(event)




