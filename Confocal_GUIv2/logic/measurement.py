import threading, time, sys, os, inspect
from abc import ABC, abstractmethod, ABCMeta
import numpy as np
import numbers
from dataclasses import dataclass, field
from typing import Any, Optional, Dict, List
from matplotlib.path import Path


from Confocal_GUIv2.helper import fuzzy_search, log_error
from Confocal_GUIv2.device import get_devices, DeviceCanceled
from Confocal_GUIv2.live_plot import *
from Confocal_GUIv2.gui import run_fluent_window, LoadGUI, get_figs_dir, LivePlotMainGUI

def measurement_gui_meta(*, unit='1', name='', xlabel='X', ylabel='Y', zlabel='Z', 
    caller=None, plotter=None, device_names: List[str] = None, update_mode_valid: List[str] = None):
    """
    Decorator to attach GUI metadata and register a caller name.
    """
    def decorator(cls):
        # attach measurement metadata
        cls.unit    = unit
        cls.name    = name
        cls.xlabel  = xlabel
        cls.ylabel  = ylabel
        cls.zlabel  = zlabel
        cls.plotter = plotter
        cls.device_names = device_names or ['counter', 'pulse']
        # all devices used in this measurement
        # counter and pulse are two devices all measurements use
        cls.update_mode_valid = update_mode_valid or ['replace',]

        if caller:
            if 'caller' not in cls.__dict__:
                raise NotImplementedError(f"{cls.__name__}.caller() is not implemented")

            module = sys.modules[cls.__module__]        # get the module where the class is defined
            setattr(module, caller, cls.caller)

        """
        devide all args in groups here

        _base_caller: 
            1, 'is_GUI'

        controller: 
            2, 'auto_save_and_close', 'qt_parent'

        live_plot: 
            3, 'update_time', 'relim_mode' 
            4, 'fig'

        __init__: 
            5, 'parent' 
            6, (BaseMeasurementConfig - 'parent')  
            7, 'fig_sub' (optional, handled by GUI or in MEASUREMENT in code not configurable)
            8, extra_init

        caller:
            9, extra_caller
            10, device name in device_names


        KEYS_FOR_CONTEXT:
            1+2+4+5+7

        KEYS_FOR_SAVE(all keys affect data and plot display):
            all but not in 1+2+4+5+7

        KEYS_FOR_DEVICES:
            cls.device_names/10

        KEYS_FOR_EXTRA(for display in GUI):
            8+9

        KEYS_FOR_INIT_CONFIG:
            keys of BaseMeasurementConfig - 'device_overrides'

        KEYS_FOR_EXTRA_INIT:
            8
        
        """

        # 2) Inspect caller() signature to get all supported keys
        caller_sig = inspect.signature(cls.caller)
        all_keys = [p for p in caller_sig.parameters if p != 'cls']

        # 3) Compute each key‐set
        #   - Context keys: never affect measurement or saving
        context_keys = frozenset({
        'is_GUI', 'auto_save_and_close', 'qt_parent',
        'fig', 'parent', 'fig_sub'
        })
        # 1+2+4+5+7

        #   - Save keys: all keys that DO affect data/storage or plotting
        save_keys = frozenset(k for k in all_keys if k not in context_keys)

        #   - Device keys: names of device overrides
        device_keys = frozenset(cls.device_names)

        #   - Extra keys: parameters used for GUI display or other init/caller extras

        config_fields = set(BaseMeasurementConfig.__dataclass_fields__.keys())
        config_fields.discard('device_overrides')
        init_config_keys = frozenset(config_fields)

        init_sig = inspect.signature(cls.__init__)
        extra_keys = set(all_keys) - context_keys - config_fields - {'relim_mode', 'update_time'} - device_keys
        extra_keys_init = set(init_sig.parameters.keys()) - {'self', 'config'}

        # 5) Attach the computed key‐sets to the instance
        cls.KEYS_FOR_CONTEXT = [k for k in all_keys if k in context_keys]
        cls.KEYS_FOR_SAVE    = [k for k in all_keys if k in save_keys]
        cls.KEYS_FOR_DEVICES = [k for k in all_keys if k in device_keys]
        cls.KEYS_FOR_INIT_CONFIG  = [k for k in all_keys if k in init_config_keys]
        cls.KEYS_FOR_EXTRA   = [k for k in all_keys if k in extra_keys]
        cls.KEYS_FOR_EXTRA_INIT  = [k for k in all_keys if k in extra_keys_init]
        # order all keys, because set is unordered operation

        return cls
    return decorator


class MeasurementMeta(ABCMeta):
    """
    Metaclass that:
    validates that metadata fields are present at instantiation
    """

    def __init__(cls, name, bases, namespace, **kwargs):
        super().__init__(name, bases, namespace)

    def __call__(cls, *args, **kwargs):
        # Validate that metadata attributes exist before instantiation
        missing = [a for a in getattr(cls, 'required', ())
                   if not hasattr(cls, a)]
        if missing:
            raise TypeError(f"{cls.__name__} is missing property: {missing}")

        return super().__call__(*args, **kwargs)

@dataclass
class BaseMeasurementConfig:
    data_x: np.ndarray
    exposure: float = 0.1
    sample_num: int = 1000
    repeat: int = 1
    counter_mode: str = 'apd'
    data_mode: str = 'single'
    update_mode: str = 'replace'
    pulse_file: Optional[str] = None
    parent: Any = None
    # user-provided device overrides: key is device name, value is device instance
    device_overrides: Dict[str, Any] = field(default_factory=dict)
    # Each new config instance gets its own empty dict by default

Measurement_gui_window = {}
# keep all measurement gui window and not close
class BaseMeasurement(ABC, metaclass=MeasurementMeta):
    """
    Base class for all measurements. Handles threading,
    data arrays, and stop/done events.
    """
    required = ('unit','name','xlabel','ylabel','zlabel', 'caller', 'plotter', 'device_names', 'update_mode_valid',
                'KEYS_FOR_CONTEXT', 'KEYS_FOR_SAVE', 'KEYS_FOR_DEVICES', 'KEYS_FOR_INIT_CONFIG', 'KEYS_FOR_EXTRA',
                'KEYS_FOR_EXTRA_INIT')

    # if not use measurement_gui_meta decorator then must also satisfy all required properties

    @classmethod
    def caller(cls, *args, **kwargs):
        """
        Default stub: must be overridden or provided by decorator.
        """
        raise NotImplementedError(f"{cls.__name__}.caller() is not implemented")

    def __init__(self, config: BaseMeasurementConfig):
        super().__init__()
        self.config = config
        self.data_x = self.config.data_x
        self.update_mode = self.config.update_mode
        if self.update_mode not in self.update_mode_valid:
            raise ValueError(f"Invalid update_mode: {self.update_mode!r}, "
                 f"must be one of {self.update_mode_valid}")

        self.exposure = self.config.exposure
        self.sample_num = int(self.config.sample_num)
        self.repeat = int(self.config.repeat)
        self.parent = self.config.parent
        # to keep stop check with parent measurement
        self.data_mode = self.config.data_mode
        self.counter_mode = self.config.counter_mode
        self.points_done = 0
        self.points_total = len(self.data_x)
        self.repeat_cur = 0
        self._done_event = threading.Event()
        # internal set
        # allow external to check if done
        self._stop_event = threading.Event()
        # check if stop or keep running
        # allow external set
        self._thread = None

        self.unit = self.unit
        self.name = self.name
        self.info = {'unit':self.unit, 'class_name': self.__class__.__name__}
        # register info for save purpose, have data_figure save labels, and _init_kwargs is added in MeasurementMeta

    def load_devices_core(self):
        for device_name in self.KEYS_FOR_DEVICES:
            device_override = self.config.device_overrides.get(device_name, None)
            if device_override is not None:
                device = get_devices(device_override)
            else:
                device = get_devices(device_name)
            setattr(self, device_name, device)
        # load e.g. self.counter = get_devices('counter') or get_devices({name give in override})

    def load_devices(self):
        # all operations for init measurement but also device dependent
        self.load_devices_core()

        if (self.pulse is not None) and (self.config.pulse_file is not None):
            is_load = self.pulse.load_from_file(self.config.pulse_file)
            if is_load:
                self.pulse.on_pulse()
            # load pulse_file to pulse if any

        self.counter.data_mode = self.data_mode
        self.counter.counter_mode = self.counter_mode
        # counter is a must device so separate from other devices

        self.data_len = self.counter.data_len
        # normally one unless data_mode dual
        if self.update_mode == 'create':
            self.data_y = np.full((len(self.data_x), self.data_len*self.repeat), np.nan)
        else:
            self.data_y = np.full((len(self.data_x), self.data_len), np.nan)
        # determine data shape from data_len of counter, and keep this shape during lifetime


    @abstractmethod
    def to_initial_state(self): ...
    @abstractmethod
    def device_to_state(self, value): ...
    @abstractmethod
    def to_final_state(self): ...
    @abstractmethod
    def get_data_y(self): ...
    # below are for gui calling only, better be non-block method
    @abstractmethod
    def read_x(self): ...
    @abstractmethod
    def set_x(self): ...

    def _update_data_add(self, idx, value):
        if np.isnan(self.data_y[idx][0]):
            # if one of the data_y[idx] is not written then all not written
            self.data_y[idx] = value
        else:
            self.data_y[idx] += value

    def _update_data_replace(self, idx, value):
        self.data_y[idx] = value

    def _update_data_create(self, idx, value):
        self.data_y[idx][(self.repeat_cur-1)*self.data_len:(self.repeat_cur)*self.data_len] = value

    def _update_data_roll(self, idx, value):
        self.data_y[:] = np.roll(self.data_y, shift=1, axis=0)
        self.data_y[0] = value

    def update_data_y(self, idx):
        # or override this in subclass
        value = self.get_data_y()
        if self.should_stop():
            return False
        else:
            getattr(self, f'_update_data_{self.update_mode}')(idx, value)
            self.points_done += 1
            return True


    def devices_cancel(self):
        for device_name in self.KEYS_FOR_DEVICES:
            device = getattr(self, device_name, None)
            if device is not None:
                device.is_cancel = True

    def devices_clear_cancel(self):
        for device_name in self.KEYS_FOR_DEVICES:
            device = getattr(self, device_name, None)
            if device is not None:
                device.is_cancel = False

    def snapshot_devices(self):
        devices_state_dict = {}
        for device_name in self.KEYS_FOR_DEVICES:
            device = getattr(self, device_name, None)
            if device is not None:
                try:
                    device_state = device.snapshot()
                except Exception as e:
                    log_error(e)
                    device_state = 'error in snapshot'
            else:
                device_state = None
            devices_state_dict[device_name] = device_state
        self.info['devices_state'] = devices_state_dict

    def should_stop(self):
        if getattr(self, 'parent', None) is not None:
            return self.parent.should_stop() or self._stop_event.is_set()
        return self._stop_event.is_set()

    def is_done(self):
        return self._done_event.is_set()

    def _loop(self):
        try:
            if self.should_stop():
                self._done_event.set()
                return
            # for load exsiting data

            self.to_initial_state()
            self.snapshot_devices() # snapshot all devices states to info for data_figure.save
            for self.repeat_cur in range(1, self.repeat+1):
                for idx, value in enumerate(self.data_x):
                    self.device_to_state(value)
                    self.update_data_y(idx)
            self._done_event.set()
            self.to_final_state()
        except DeviceCanceled:
            print("[Measurement] Measurement canceled by user.")
            self._done_event.set()
            self.to_final_state()
            return
        except Exception as e:
            log_error(e)
            self._done_event.set()
            # in case of error only do basic clean up 
            return


    def start(self):
        if self._thread is not None:
            print('Measurement has already started')
            return
        else:
            self.devices_clear_cancel()
            th = threading.Thread(target=self._loop, daemon=True)
            th.start()
            self._thread = th

    def stop(self):
        self._stop_event.set()
        self.devices_cancel()
        if self._thread is not None:
            self._thread.join()
        self.devices_clear_cancel()
        # clear cancel event to make sure devices are ready


    @classmethod
    def _base_caller(cls, args: dict):
        # args contains all the parameters passed to caller(), including GUI flags,
        # BaseMeasurementConfig fields, device override slots, and any extras.
        try:
            is_GUI              = args['is_GUI']
            if not is_GUI:
                data_x = cls.generate_data_x(**args)
                args.update({'data_x':data_x})

                address = args.get('address', None)
                loaded = args.get('loaded', None)
                is_load = True if address is not None else False

                # 1) Read GUI-control parameters without removing them
                fig                 = args['fig']
                auto_save_and_close = args['auto_save_and_close']
                qt_parent           = args['qt_parent']

                relim_mode          = args['relim_mode']
                update_time         = args['update_time']


                config_kwargs = {k: args[k] for k in cls.KEYS_FOR_INIT_CONFIG if k in args}
                overrides = {k: args[k] for k in cls.KEYS_FOR_DEVICES if k in args}
                config = BaseMeasurementConfig(
                    **config_kwargs,
                    device_overrides=overrides
                )

                extra_init = {k: args[k] for k in cls.KEYS_FOR_EXTRA_INIT if k in args}
                caller_kwargs_for_save = {k: args[k] for k in cls.KEYS_FOR_SAVE if k in args}

                measurement = cls(config=config, **extra_init)
                plotter_cls = globals()[cls.plotter]
                measurement.info.update({'caller_kwargs_for_save':caller_kwargs_for_save})
                if is_load is True:
                    measurement.stop()
                    info = loaded['info'].item()
                    measurement.data_y = loaded.get('data_y')
                    measurement.points_done = info.get('points_done', measurement.points_total*measurement.repeat)
                    measurement.repeat_cur = info.get('repeat_cur', measurement.repeat)
                    if info.get('load_from', None) is None:
                        measurement.info.update({'load_from':address})
                    else:
                        measurement.info.update({'load_from':info.get('load_from')})
                    # make sure all link to the original copy
                    labels=info.get('labels', ['X', 'Y', 'Z'])
                    update_time=0.1
                else:
                    measurement.load_devices()
                    labels=measurement.labels

                live_plot = plotter_cls(
                    data_x=measurement.data_x,
                    data_y=measurement.data_y,
                    labels=labels,
                    update_time=update_time,
                    fig=fig,
                    relim_mode=relim_mode
                )

                ctr = PlotController(
                    measurement=measurement,
                    live_plot=live_plot,
                    auto_save_and_close=auto_save_and_close,
                    qt_parent=qt_parent
                )
                ctr.plot()
                return live_plot

            else:
                # GUI mode: open interactive Fluent window

                global Measurement_gui_window
                key = f'{cls.__module__}.{cls.__name__}'
                _gui_window = Measurement_gui_window.get(key)
                if _gui_window is None:
                    _gui_window = run_fluent_window(
                        use_fluent=True,
                        widget_class=LivePlotMainGUI,
                        widget_kwargs={'cls_name_left': cls, 'cls_name_right': LiveMeasurement},
                        title=f'{cls.name}-GUI@Wanglab, UOregon',
                        ui_path=None
                    )
                    Measurement_gui_window[key] = _gui_window
                else:
                    _gui_window = run_fluent_window(window_handle = _gui_window)

                return _gui_window
        except Exception as e:
            log_error(e)
            return None


    @staticmethod
    def generate_data_x(**kwargs):
        """
        Filters a regular (x_array, y_array) grid down to those cells whose area
        overlaps a convex polygon defined by bound_pts. Returns the (x, y) centers
        of those cells.

        Parameters:
        - x_array: 1D numpy array of x-coordinates (uniformly spaced)
        - y_array: 1D numpy array of y-coordinates (uniformly spaced)
        - bound_pts: list of (x, y) tuples defining a convex polygon

        Returns:
        - numpy array of shape (N, 2), where each row is (x_center, y_center)
          for a grid cell whose area overlaps the polygon.
        """
        # If no polygon is given, return the full grid of centers

        x_array = kwargs.get('x_array', None)
        y_array = kwargs.get('y_array', None)
        bound_pts = kwargs.get('bound_pts', None)
        data_x = kwargs.get('data_x', None)
        if data_x is not None:
            return data_x
        if y_array is None:
            if x_array is not None:
                return x_array.reshape(-1, 1)
            else:
                return None

        if not bound_pts:
            xx, yy = np.meshgrid(x_array, y_array)
            return np.vstack([xx.ravel(), yy.ravel()]).T

        # If either axis has fewer than 2 points, fall back to naive enumeration
        if len(x_array) < 2 or len(y_array) < 2:
            return np.array([(x, y) for y in y_array for x in x_array])

        # Compute grid spacing and half-spacing
        dx = x_array[1] - x_array[0]
        dy = y_array[1] - y_array[0]
        half_dx, half_dy = dx / 2.0, dy / 2.0

        # Compute axis-aligned bounding box of polygon, expanded by half a cell
        pts = np.array(bound_pts)
        xmin, ymin = pts.min(axis=0) - np.array([half_dx, half_dy])
        xmax, ymax = pts.max(axis=0) + np.array([half_dx, half_dy])

        # Find indices of grid points whose centers lie within that expanded box
        ix = np.where((x_array >= xmin) & (x_array <= xmax))[0]
        iy = np.where((y_array >= ymin) & (y_array <= ymax))[0]

        # If the box filter yields nothing, return the full grid
        if ix.size == 0 or iy.size == 0:
            xx, yy = np.meshgrid(x_array, y_array)
            return np.vstack([xx.ravel(), yy.ravel()]).T

        # Build the subset of grid centers to test
        xs_sub = x_array[ix]
        ys_sub = y_array[iy]
        pts_sub = np.array([(x, y) for y in ys_sub for x in xs_sub])

        # Create a Path object for the polygon
        poly = Path(bound_pts)

        # 1) Center-point test: which cell-centers lie inside the polygon?
        center_mask = poly.contains_points(pts_sub)

        # 2) Corner/vertex test for the remaining cells
        corner_mask = np.zeros(len(pts_sub), dtype=bool)
        for idx, (x, y) in enumerate(pts_sub):
            # Define the four corners of this cell
            corners = np.array([
                (x - half_dx, y - half_dy),
                (x - half_dx, y + half_dy),
                (x + half_dx, y - half_dy),
                (x + half_dx, y + half_dy),
            ])
            # Check if any corner lies inside the polygon
            if poly.contains_points(corners).any():
                corner_mask[idx] = True
                continue

            # Or if any polygon vertex lies inside the cell
            in_x = (pts[:, 0] > x - half_dx) & (pts[:, 0] < x + half_dx)
            in_y = (pts[:, 1] > y - half_dy) & (pts[:, 1] < y + half_dy)
            if np.any(in_x & in_y):
                corner_mask[idx] = True

        # Combine masks: keep cells whose center or corners overlap the polygon
        total_mask = center_mask | corner_mask
        selected = pts_sub[total_mask]

        # If nothing survived (unlikely), fall back to full grid
        if selected.size == 0:
            print("Warning: polygon did not overlap filtered grid; returning full grid.")
            xx, yy = np.meshgrid(x_array, y_array)
            return np.vstack([xx.ravel(), yy.ravel()]).T

        return selected

def load(addr='', fig=None, is_GUI=False, is_print_log=False, qt_parent=None):
    global filename
    filename = None
    qt_parent = None # only allow controller run as 'nb' mode to block the loading process
    def on_filename(addr):
        global filename
        filename = addr
    if is_GUI:
        run_fluent_window(use_fluent=True,widget_class=LoadGUI,
            widget_kwargs={'on_file_selected': on_filename},
        title=f'load-GUI@Wanglab, UOregon',ui_path=None)
        # have to use this way to have run_fluent_window call filedialog otherwise
        # cannot use qframelesswindow 

        if filename is not None:
            addr = filename[:-4] + '*'
        else:
            addr = ''

    address = fuzzy_search(addr, DIR=get_figs_dir(), file_type='.npz')
    if address is None:
        return None

    
    loaded = np.load(address, allow_pickle=True)
    keys = loaded.files
    if is_print_log is True:
        print("Keys in npz file:", keys)
        print(loaded['info'].item())

    info = loaded['info'].item()
    class_name = info.get('class_name', 'PLEMeasurement')
    caller_kwargs_for_save = info.get('caller_kwargs_for_save', {})
    cls_name = globals().get(class_name)

    caller_kwargs_to_load = {**caller_kwargs_for_save, 'is_GUI':False, 'fig':fig, 'auto_save_and_close':True, 'qt_parent':qt_parent, 
    'address':address, 'loaded':loaded}

    return cls_name._base_caller(caller_kwargs_to_load)

GUI_gui_window = None
def GUI():
    global GUI_gui_window
    if GUI_gui_window is None:
        GUI_gui_window = run_fluent_window(
            use_fluent=True,
            widget_class=LivePlotMainGUI,
            widget_kwargs={},
            title=f'GUI@Wanglab, UOregon',
            ui_path=None,
        )
    else:
        GUI_gui_window = run_fluent_window(window_handle = GUI_gui_window)
    return GUI_gui_window

def raw_plot(
    data_x=np.array([[i] for i in np.arange(100)]),
    data_y=np.array([[i] for i in np.arange(100)]),
    labels=['X', 'Y', 'Z'],
    relim_mode='tight',
    cls_name_str='ple'
):
    # Resolve the bound @classmethod (caller) and get its class
    caller_fn = globals()[cls_name_str]     # raises KeyError if not found
    cls = caller_fn.__self__                # the measurement class

    # Resolve plotter (string → class, or use it directly if it's already a class)
    plotter_cls = globals()[cls.plotter] if isinstance(cls.plotter, str) else cls.plotter

    save_and_close_previous()
    change_to_widget()
    # One-shot render
    live_plot = plotter_cls(
        data_x=data_x,
        data_y=data_y,
        labels=labels,
        update_time=0.1,
        fig=None,
        relim_mode=relim_mode
    )
    live_plot.init_figure_and_data()
    live_plot.update_figure()
    live_plot.after_plot()
    return live_plot

@measurement_gui_meta(
    unit='nm',
    name='ple',
    xlabel='Wavelength',
    ylabel='Counts',
    caller='ple',    # <— this wires it up so `ple(...)` exists
    plotter='Live1D',
    device_names=['counter','pulse','laser_stabilizer','wavemeter'],
    # all devices used in this measurement
    update_mode_valid=['add', 'replace', 'create'],
)
class PLEMeasurement(BaseMeasurement):
    def __init__(self, config:BaseMeasurementConfig):
        super().__init__(config=config)
        xlabel = f'{self.xlabel} ({self.unit})'
        if self.counter_mode=='apd_sample':
            ylabel = f'{self.ylabel}/{self.sample_num}pts'
        else:
            ylabel = f'{self.ylabel}/{self.exposure}s'
        zlabel = f'{self.zlabel}'
        self.labels = [xlabel, ylabel, zlabel]

    def to_initial_state(self):
        self.laser_stabilizer.on = True
    def device_to_state(self, value):
        self.laser_stabilizer.wait_to_wavelength(value[0])
    def to_final_state(self):
        self.laser_stabilizer.on = False
    def get_data_y(self):
        return self.counter.read_counts(exposure=self.exposure, sample_num=self.sample_num, parent=self)

    def read_x(self):
        return self.wavemeter.wavelength
    def set_x(self, value):
        self.laser_stabilizer.wavelength = value[0]


    @classmethod
    def caller(cls, 
        is_GUI=False,
        # if open gui
        data_x=None, x_array=np.linspace(737.095, 737.105, 251), exposure=0.1, sample_num=1000, repeat=1, 
        counter_mode='apd', data_mode='single', update_mode='add',
        pulse_file=None, parent=None, 
        # BaseMeasurementConfig
        counter='counter', pulse='pulse', laser_stabilizer='laser_stabilizer', wavemeter='wavemeter',
        # device_names
        # e.g. power=10
        # other params for measurement if any
        update_time=0.2, fig=None, relim_mode='tight',
        # live_plot params 
        auto_save_and_close=True, qt_parent=None):
        # controller params
        return cls._base_caller(locals())


@measurement_gui_meta(
    unit='us',
    name='pulsex',
    xlabel='Pulse X',
    ylabel='Counts',
    caller='pulsex',
    plotter='Live1D',
    device_names=['counter','pulse'],
    # all devices used in this measurement
    update_mode_valid=['add', 'replace', 'create'],
)
class PulseXMeasurement(BaseMeasurement):
    def __init__(self, config:BaseMeasurementConfig):
        super().__init__(config=config)
        xlabel = f'{self.xlabel} ({self.unit})'
        if self.counter_mode=='apd_sample':
            ylabel = f'{self.ylabel}/{self.sample_num}pts'
        else:
            ylabel = f'{self.ylabel}/{self.exposure}s'
        zlabel = f'{self.zlabel}'
        self.labels = [xlabel, ylabel, zlabel]

    def to_initial_state(self):
        pass
    def device_to_state(self, value):
        self.pulse.x = value[0]*1e3
        self.pulse.on_pulse()
    def to_final_state(self):
        pass
    def get_data_y(self):
        return self.counter.read_counts(exposure=self.exposure, sample_num=self.sample_num, parent=self)

    def read_x(self):
        return self.pulse.x/1e3
    def set_x(self, value):
        self.device_to_state(value=value)


    @classmethod
    def caller(cls, 
        is_GUI=False,
        # if open gui
        data_x=None, x_array=np.linspace(-10, 10, 201), exposure=0.1, sample_num=1000, repeat=1, 
        counter_mode='apd', data_mode='single', update_mode='add',
        pulse_file=None, parent=None, 
        # BaseMeasurementConfig
        counter='counter', pulse='pulse',
        # device_names
        # e.g. power=10
        # other params for measurement if any
        update_time=0.2, fig=None, relim_mode='tight',
        # live_plot params 
        auto_save_and_close=True, qt_parent=None):
        # controller params
        return cls._base_caller(locals())



@measurement_gui_meta(
    unit='1',
    name='ple_center',
    xlabel='Points',
    ylabel='Wavelength',
    caller='ple_center',    # <— this wires it up so `ple(...)` exists
    plotter='Live1D',
    device_names=['counter','pulse','laser_stabilizer','wavemeter'],
    # all devices used in this measurement
    update_mode_valid=['replace', 'create'],
)
class PLECENTERMeasurement(BaseMeasurement):
    def __init__(self, config:BaseMeasurementConfig, fig_sub=None):
        super().__init__(config=config)
        xlabel = f'{self.xlabel}'
        ylabel = f'{self.ylabel} (nm)'
        zlabel = f'{self.zlabel}'
        self.labels = [xlabel, ylabel, zlabel]
        self.fig_sub = fig_sub

    def to_initial_state(self):
        pass
    def device_to_state(self, value):
        pass
    def to_final_state(self):
        pass
    def get_data_y(self):
        caller_params = self.info.get('caller_kwargs_for_save')
        self.live_plot_sub = ple(parent=self, fig=getattr(self, 'fig_sub', None), auto_save_and_close=False,
            data_x = caller_params.get('data_x_sub'), x_array=caller_params.get('x_array_sub'),
            repeat=caller_params.get('repeat_sub'),
            exposure=caller_params.get('exposure_sub'), sample_num=caller_params.get('sample_num_sub'),
            counter_mode=caller_params.get('counter_mode_sub'),
            data_mode=caller_params.get('data_mode_sub'), update_mode=caller_params.get('update_mode_sub'), 
            update_time=caller_params.get('update_time_sub'), relim_mode=caller_params.get('relim_mode_sub'),
            counter=caller_params.get('counter'),
            pulse=caller_params.get('pulse'),
            laser_stabilizer=caller_params.get('laser_stabilizer'),
            wavemeter=caller_params.get('wavemeter'))
        self.fig_sub = self.live_plot_sub.fig
        _, popt = self.live_plot_sub.data_figure.lorent()
        return popt[0]

    def read_x(self):
        return 0
    def set_x(self, value):
        pass


    @classmethod
    def caller(cls, 
        is_GUI=False,
        # if open gui
        data_x=None, x_array=np.arange(30), exposure=0.1, sample_num=1000, repeat=1, 
        counter_mode='apd', data_mode='single', update_mode='replace',
        pulse_file=None, parent=None, 
        # BaseMeasurementConfig
        counter='counter', pulse='pulse', laser_stabilizer='laser_stabilizer', wavemeter='wavemeter',
        # device_names
        # e.g. power=10
        data_x_sub=None, x_array_sub=np.linspace(737.097, 737.103, 151), exposure_sub=0.2, sample_num_sub=1000, repeat_sub=1,
        counter_mode_sub='apd', data_mode_sub='single', update_mode_sub='add',
        update_time_sub=0.2, relim_mode_sub='tight', fig_sub=None,
        # other params for measurement if any
        update_time=0.2, fig=None, relim_mode='tight',
        # live_plot params 
        auto_save_and_close=True, qt_parent=None):
        # controller params
        return cls._base_caller(locals())


@measurement_gui_meta(
    unit='1',
    name='pl',
    xlabel='X',
    ylabel='Y',
    zlabel = 'Counts',
    caller='pl',
    plotter='Live2DDis',
    device_names=['counter', 'pulse', 'scanner'],
    update_mode_valid=['add', 'replace'],
)
class PLMeasurement(BaseMeasurement):
    def __init__(self, config:BaseMeasurementConfig):
        super().__init__(config=config)
        xlabel = f'{self.xlabel}'
        ylabel = f'{self.ylabel}'
        if self.counter_mode=='apd_sample':
            zlabel = f'{self.zlabel}/{self.sample_num}pts'
        else:
            zlabel = f'{self.zlabel}/{self.exposure}s'
        self.labels = [xlabel, ylabel, zlabel]
        self.cache_x = None
        self.cache_y = None

    def to_initial_state(self):
        pass
    def device_to_state(self, value):
        x = value[0]
        y = value[1]
        if (x is not None) and (x!=self.cache_x):
            self.scanner.x = x
            self.cache_x = x
        if (y is not None) and (y!=self.cache_y):
            self.scanner.y = y
            self.cache_y = y
        # use cache to avoid extra overhead in setting scanner x, y
    def to_final_state(self):
        pass
    def get_data_y(self):
        return self.counter.read_counts(exposure=self.exposure, sample_num=self.sample_num, parent=self)

    def read_x(self):
        return (self.scanner.x, self.scanner.y)
    def set_x(self, value):
        self.device_to_state(value=value)

    @classmethod
    def caller(cls, 
        is_GUI=False,
        # if open gui
        data_x=None, x_array=np.arange(-20, 21, 2), y_array=np.arange(-20, 21, 2), 
        bound_pts=[(-15, 10), (-15, -10), (20, -20), (20, 20)], 
        # in data_x OR x_array*y_array OR x_array*y_array& in (pt1, pt2, pt3, ...) quadrilateral
        exposure=0.05, sample_num=1000, repeat=1,
        counter_mode='apd', data_mode='single', update_mode='add',
        pulse_file=None, parent=None, 
        # BaseMeasurementConfig
        counter='counter', pulse='pulse', scanner='scanner',
        # device_names
        # e.g. power=10
        # other params for measurement if any
        update_time=1, fig=None, relim_mode='tight',
        # live_plot params 
        auto_save_and_close=True, qt_parent=None):
        # controller params
        return cls._base_caller(locals())


@measurement_gui_meta(
    unit='MHz',
    name='odmr',
    xlabel='RF Frequency',
    ylabel='Counts',
    caller='odmr',
    plotter='Live1D',
    device_names=['counter', 'pulse', 'rf'],
    update_mode_valid=['add', 'replace', 'create'],
)
class ODMRMeasurement(BaseMeasurement):
    def __init__(self, config:BaseMeasurementConfig, power:float):
        super().__init__(config=config)
        self.power = power
        xlabel = f'{self.xlabel} ({self.unit})'
        if self.counter_mode=='apd_sample':
            ylabel = f'{self.ylabel}/{self.sample_num}pts'
        else:
            ylabel = f'{self.ylabel}/{self.exposure}s'
        zlabel = f'{self.zlabel}'
        self.labels = [xlabel, ylabel, zlabel]

    def to_initial_state(self):
        self.rf.on = True
        self.rf.power = self.power
    def device_to_state(self, value):
        self.rf.frequency = value[0]*1e6
    def to_final_state(self):
        self.rf.on = False
    def get_data_y(self):
        return self.counter.read_counts(exposure=self.exposure, sample_num=self.sample_num, parent=self)

    def read_x(self):
        return self.rf.frequency/1e6
    def set_x(self, value):
        self.device_to_state(value=value)


    @classmethod
    def caller(cls, 
        is_GUI=False,
        # if open gui
        data_x=None, x_array=np.linspace(2870-50, 2870+50, 101), exposure=0.1, sample_num=1000, repeat=1,
        counter_mode='apd', data_mode='single', update_mode='add',
        pulse_file=None, parent=None, 
        # BaseMeasurementConfig
        counter='counter', pulse='pulse', rf='rf',
        # device_names
        power=10,
        # other params for measurement if any
        update_time=0.2, fig=None, relim_mode='tight',
        # live_plot params 
        auto_save_and_close=True, qt_parent=None):
        # controller params
        return cls._base_caller(locals())


@measurement_gui_meta(
    unit='1',
    name='live',
    xlabel='Points',
    ylabel='Counts',
    caller='live',
    plotter='LiveLiveDis',
    device_names=['counter', 'pulse'],
    update_mode_valid=['roll'],
)
class LiveMeasurement(BaseMeasurement):
    def __init__(self, config:BaseMeasurementConfig):
        super().__init__(config=config)
        xlabel = f'{self.xlabel}'
        if self.counter_mode=='apd_sample':
            ylabel = f'{self.ylabel}/{self.sample_num}pts'
        else:
            ylabel = f'{self.ylabel}/{self.exposure}s'
        zlabel = f'{self.zlabel}'
        self.labels = [xlabel, ylabel, zlabel]

    def to_initial_state(self):
        pass
    def device_to_state(self, value):
        pass
    def to_final_state(self):
        pass
    def get_data_y(self):
        return self.counter.read_counts(exposure=self.exposure, sample_num=self.sample_num, parent=self)

    def read_x(self):
        return 0
    def set_x(self, value):
        pass


    @classmethod
    def caller(cls, 
        is_GUI=False,
        # if open gui
        data_x=None, x_array=np.arange(100), exposure=0.1, sample_num=1000, repeat=1,
        counter_mode='apd', data_mode='single', update_mode='roll',
        pulse_file=None, parent=None, 
        # BaseMeasurementConfig
        counter='counter', pulse='pulse',
        # device_names
        is_finite=False,
        # other params for measurement if any
        update_time=0.05, fig=None, relim_mode='normal',
        # live_plot params 
        auto_save_and_close=True, qt_parent=None):
        # controller params
        if is_finite is False:
            repeat = int(10000)
        return cls._base_caller(locals())



@measurement_gui_meta(
    unit='MHz',
    name='mode',
    xlabel='RF Frequency',
    ylabel='Counts',
    caller='mode',
    plotter='Live1D',
    device_names=['counter', 'pulse', 'rf', 'laser', 'laser_stabilizer'],
    update_mode_valid=['replace', 'create'],
)
class ModeMeasurement(BaseMeasurement):
    def __init__(self, config:BaseMeasurementConfig, power, wavelength, is_adaptive, ref_freq, h10_ratio, beta, exposure_h1):
        super().__init__(config=config)
        self.power = power
        self.wavelength = wavelength
        self.is_adaptive = is_adaptive
        self.ref_freq = ref_freq
        self.h10_ratio = h10_ratio
        self.beta = beta
        self.exposure_h1 = exposure_h1
        self.h0_single_read = None
        xlabel = f'{self.xlabel} ({self.unit})'
        if self.counter_mode=='apd_sample':
            ylabel = f'{self.ylabel}/{self.sample_num}pts{" adaptive" if self.is_adaptive else ""}'
        else:
            ylabel = f'{self.ylabel}/{self.exposure}s{" adaptive" if self.is_adaptive else ""}'
        zlabel = f'{self.zlabel}'
        self.labels = [xlabel, ylabel, zlabel]

    def _cali_h0(self):
        prev_freq = self.read_x()
        self.device_to_state(value=[self.ref_freq,])
        # calibrate the average base counts per single read for h0 or lambda_0
        counts = self.counter.read_counts(exposure=self.exposure, sample_num=self.sample_num, parent=self)[0]
        t0 = time.time()
        n_read = 1
        while (time.time()-t0)<self.exposure_h1:
            counts += self.counter.read_counts(exposure=self.exposure, sample_num=self.sample_num, parent=self)[0]
            n_read += 1
        self.h0_single_read = counts/n_read
        estimated_n_read = np.log(self.beta)/(self.h0_single_read*(1+np.log(self.h10_ratio)-self.h10_ratio))
        print(f'[mode] estimated lambda_0:{self.h0_single_read}, n_read:{estimated_n_read}')
        self.device_to_state(value=[prev_freq,])

    def _counts_threshold(self, n):
        # calculate the counts threshold which indicates when to stop based on beta, n, lambda_0, h10_ratio
        # a = ln(beta/(1-alpha)) = ln(beta)
        # log-likelyhood ratio for poisson distribution is
        # llr = counts*ln(lambda_1/lambda_0) - (lambda_1-lambda_0)
        # sum_1_to_n(llr_i) < a --> counts_threshold
        return (np.log(self.beta) + n*self.h0_single_read*(self.h10_ratio-1))/np.log(self.h10_ratio)

    def to_initial_state(self):
        self.rf.on = True
        self.rf.power = self.power
        self.laser_stabilizer.on = True
        self.laser_stabilizer.wavelength = self.wavelength
    def device_to_state(self, value):
        self.rf.frequency = value[0]*1e6
    def to_final_state(self):
        self.rf.on = False
        self.laser_stabilizer.on = False
    def get_data_y(self):
        if (self.is_adaptive is True) and (self.h0_single_read is None):
            self._cali_h0()

        counts = self.counter.read_counts(exposure=self.exposure, sample_num=self.sample_num, parent=self)[0]
        t0 = time.time()
        n_read = 1

        while (self.is_adaptive is True) and (time.time()-t0)<self.exposure_h1:
            if counts < self._counts_threshold(n_read):
                # if meet h0 then return otherwise continue
                break
            counts += self.counter.read_counts(exposure=self.exposure, sample_num=self.sample_num, parent=self)[0]
            n_read += 1

        return [counts/n_read,]

    def read_x(self):
        return self.rf.frequency/1e6
    def set_x(self, value):
        self.device_to_state(value=value)


    @classmethod
    def caller(cls, 
        is_GUI=False,
        # if open gui
        data_x=None, x_array=np.arange(1000-1, 1000+1+1e-4, 1e-4), exposure=0.05, sample_num=1000, repeat=1,
        counter_mode='apd', data_mode='single', update_mode='replace',
        pulse_file=None, parent=None, 
        # BaseMeasurementConfig
        counter='counter', pulse='pulse', rf='rf',
        laser='laser', laser_stabilizer='laser_stabilizer',
        # device_names
        power=-20, wavelength=737.11,
        is_adaptive=False,
        ref_freq=500,
        # 500MHz is not resonate frequency
        h10_ratio=1.5, beta=0.10,
        # h1/h0 ratio, and beta the type II error of H1
        exposure_h1 = 1,
        # params for sequential probability ratio test (SPRT)
        # other params for measurement if any
        update_time=1, fig=None, relim_mode='tight',
        # live_plot params 
        auto_save_and_close=True, qt_parent=None):
        # controller params
        return cls._base_caller(locals())


