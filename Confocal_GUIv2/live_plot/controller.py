import threading
from PyQt5.QtCore import QTimer
from .plot_strategy import save_and_close_previous, change_to_widget
from Confocal_GUIv2.helper import log_error
import time
import inspect
from .data_figure import DataFigure

class PlotController:
    def __init__(self, measurement, live_plot, auto_save_and_close=True, qt_parent=None):
        self.measurement = measurement
        self.live_plot = live_plot
        self.auto_save_and_close = auto_save_and_close
        self.qt_parent = qt_parent
        # parent qt to set QTimer event
        self.mode = 'nb' if self.qt_parent is None else 'qt'
        self.update_time = self.live_plot.update_time
        self.is_interrupt = False
        self._start_event = threading.Event()
        self._done_event = threading.Event()
        self._finalized_event = threading.Event()
        self._after_tasks = []

    def register_after_task(self, func):
        self._after_tasks.append(func)
        sig = inspect.signature(func)
        params = sig.parameters
        if self._done_event.is_set():
            if 'controller' in params:
                func(controller=self)
            else:
                func()

    def _run_after_tasks(self):
        for func in self._after_tasks:
            sig = inspect.signature(func)
            params = sig.parameters
            if 'controller' in params:
                func(controller=self)
            else:
                func()

    def plot(self):
        # call plot in measurement
        if self._start_event.is_set():
            print("Measurement has been started once")
            return
        self._start_event.set()
        if self.auto_save_and_close:
            save_and_close_previous()
        if self.mode == 'nb':
            change_to_widget()
        self.live_plot.controller = self
        # don't move this line, otherwise controller may be GC by Qt and lost the control of plotter
        self.live_plot.init_figure_and_data()
        # make sure live_plot create fig before any possible fig logic in measurement
        # need clear in init_figure_and_data to clean up residual selector, callbacks in fig
        self.measurement.start()

        if self.mode == 'nb':
            try:
                while not self.measurement.is_done():
                    time.sleep(self.update_time)
                    if self.measurement.points_done == self.live_plot.points_done:
                        continue
                    else:
                        self.live_plot.update_figure(points_done=self.measurement.points_done, 
                            repeat_cur=self.measurement.repeat_cur)
            except KeyboardInterrupt:
                self.is_interrupt = True
            except Exception as e:
                log_error(e)
                self.is_interrupt = True
            finally:
                self.stop()
                return self.live_plot

        elif self.mode == 'qt':
            self._qt_timer = QTimer(self.qt_parent)
            self._qt_timer.timeout.connect(self._qt_loop)
            self._qt_timer.start(int(self.update_time * 1000))
            return self.live_plot

    def stop(self):
        if not self._start_event.is_set() or self._finalized_event.is_set():
            return
        self.measurement.stop()

        if self.mode == 'qt':
            self._qt_timer.stop()

        self._finalize_plot()

    def _finalize_plot(self):
        if self.measurement.points_done>0:
            self.live_plot.update_figure(
                points_done=self.measurement.points_done,
                repeat_cur=self.measurement.repeat_cur
            )

        self.live_plot.after_plot()
        data_figure = DataFigure(live_plot=self.live_plot)
        self.live_plot.data_figure = data_figure

        self._done_event.set()
        self._run_after_tasks()
        self._finalized_event.set()

    def _qt_loop(self):
        try:
            if self.measurement.is_done():
                self.stop()
            else:
                if self.measurement.points_done == self.live_plot.points_done:
                    return
                else:
                    self.live_plot.update_figure(points_done=self.measurement.points_done, 
                        repeat_cur=self.measurement.repeat_cur)
        except Exception as e:
            log_error(e)
            self._qt_timer.stop()
            self._done_event.set()
            self._finalized_event.set()
            # in case of error only do basic clean up





