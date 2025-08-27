import numpy as np
import time, inspect, weakref
from numbers import Number
import os, glob, sys, json
from abc import ABC, abstractmethod, ABCMeta
import atexit
from functools import partial
import copy, traceback

import threading, importlib, functools, types
from PyQt5.QtCore import QEventLoop, QTimer
from PyQt5.QtWidgets import QApplication

import rpyc
from rpyc.utils.server import ThreadedServer
from rpyc.utils.authenticators import SSLAuthenticator, AuthenticationError
from rpyc.utils.classic import SlaveService

from Confocal_GUIv2.helper import fuzzy_search, log_error, align_to_resolution, python2plain, insert_mul_before_x
from Confocal_GUIv2.gui import DeviceGUI, PulseGUI, DeviceManagerGUI, run_fluent_window, get_pulses_dir, get_configs_dir
from .rsa import CERTIFICATE_DIR


global app_in_device
app = QApplication.instance()
if app is None:
    app_in_device = QApplication(sys.argv)
# make sure has QApplication before using time_sleep


_dm = None # global instances for DeviceManager
DeviceManager_gui_window = None
def init_devices(lookup_dict=None, addr=None, is_GUI=False):
    """initialize devices"""
    global _dm

    def action(config, lookup_dict):
        global _dm
        if isinstance(_dm, DeviceManager):
            _dm.reload(config=config, lookup_dict=lookup_dict)
        else:
            _dm = DeviceManager(config=config, lookup_dict=lookup_dict)

    # Decide which config to use
    cfg = {}
    if addr is not None:
        try:
            dir_to_use = get_configs_dir()
            path = fuzzy_search(addr, DIR=dir_to_use, file_type='.json')
            if path:
                with open(path, 'r', encoding='utf-8') as f:
                    cfg = json.load(f)
                if cfg:
                    print(f"[init_devices] Loaded config from {path}.")
                else:
                    print(f"[init_devices] Loaded empty config from {path}.")
            else:
                print(f"[init_devices] No config matched '{addr}' (dir={dir_to_use}).")
        except Exception as e:
            log_error(e)

    if is_GUI is False:
        action(cfg, lookup_dict)
        return _dm


    global DeviceManager_gui_window

    if DeviceManager_gui_window is None:
        DeviceManager_gui_window = run_fluent_window(
            use_fluent=True,
            widget_class=DeviceManagerGUI,
            widget_kwargs={
                'config': cfg,
                'lookup_dict': lookup_dict,
                'action': action,
            },
            title='DeviceManagerGUI@Wanglab, UOregon',
            ui_path=None
        )
    else:
        DeviceManager_gui_window = run_fluent_window(window_handle = DeviceManager_gui_window)

    return _dm


def get_devices(device_name:str = None):
    """
    e.g.
    dm = get_devices()
    or
    laser = get_devices('laser')
    """
    if _dm is None:
        return None
    if device_name is None:
        return _dm
    else:
        device = getattr(_dm, device_name, None)
        if device is None:
            raise KeyError(f"No device {device_name}")
        else:
            return device

class DeviceManager:
    def __init__(self, config: dict, lookup_dict: dict):
        """
        config: dict
        lookup_dict: globals() to share the classes imported to notebook

        e.g.
        config = 
          {
            'scanner': {
                'type':'VirtualScanner'
            },

            'laser'  : {
                'type':'VirtualLaser',
                'params':{'piezo_lb':60, 'piezo_ub':100}
            },

            'laser_stabilizer':{
                'type':'VirtualLaserStabilizer',
                'params':{
                    'wavemeter_handle':'$device:wavemeter',
                    'laser_handle'   :'$device:laser'
               },
            },

            'wavemeter': {
                'type':'VirtualWaveMeter'
            },
          }
        dm = DeviceManager(config=config, lookup_dict=globals())
        """
        self._config    = config
        self._lookup    = lookup_dict
        self._instances = {}
        self.loaded_config = {}

        for name in config:
            self._build(name)

        if self._instances:
            for k, v in self._instances.items():
                print(f"{k} => {v}")
            print(f'\nNow you can call devices using \ne.g. \n{k} \nor \ndm = get_devices()\ndm.{k}')
        else:
            print("(no devices loaded)")
        if self._lookup is not None:
            self._lookup.update(self._instances)

    def list_instances(self):
        return {k: (type(v).__name__, v) for k, v in self._instances.items()}

    def close_selected(self, name):
        inst = self._instances.get(name)
        try:
            still_referenced_elsewhere = any(
                (k != name) and (v is inst) for k, v in self._instances.items()
            )
            if self._lookup.get(name) is inst:
                self._lookup.pop(name, None)
            if not still_referenced_elsewhere:
                try:
                    inst.close()
                except Exception as e:
                    pass
            self._instances.pop(name, None)
            self.loaded_config.pop(name, None)
        except Exception as e:
            log_error(e)

    def close_all(self):
        for name, inst in list(self._instances.items()):
            self.close_selected(name)

    def _build(self, name: str):
        try:
            entry = self._config[name]
            type_name = entry['type']
            if type_name not in self._lookup:
                raise KeyError(f"Cannot find class '{type_name}' for device '{name}'")
            cls = self._lookup[type_name]

            raw_kwargs = dict(entry.get('params', {}))

            # 2. "$device:xxx" → self._build("xxx")
            for arg, val in list(raw_kwargs.items()):
                if isinstance(val, str) and val.startswith("$device:"):
                    dep = val.split(":", 1)[1]
                    if dep not in self._config:
                        raise KeyError(f"Dependency '{dep}' for '{name}' not found in config")
                    raw_kwargs[arg] = self._build(dep)

            sig = inspect.signature(cls.__init__)
            for param_name, param in sig.parameters.items():
                if param_name == "self":
                    continue
                if param_name not in raw_kwargs and param.default is inspect._empty:
                    raise TypeError(
                        f"Missing required arg '{param_name}' for device '{name}'"
                    )

            inst = cls(**raw_kwargs)
            self._instances[name] = inst
            self.loaded_config[name] = self._config.get(name, None)
            return inst
        except Exception as e:
            log_error(e)
            return None

    def reload(self, config: dict, lookup_dict=None):
        """
        Reload:

        """
        self._config = {**self.loaded_config, **config}
        # new config is config loaded last time + config this time
        self._lookup  = lookup_dict if lookup_dict is not None else self._lookup

        # Rebuild instance table strictly from the new config (reuse is handled elsewhere, e.g., metaclass)
        for name in config:
            old_inst = self._instances.get(name, None)
            self._build(name)
            new_inst = self._instances.get(name, None)
            if (old_inst is not None) and (old_inst is not new_inst):
                try:    
                    old_inst.close()
                except Exception as e:
                    pass

        print('Reload devices with config\n')
        if self._instances:
            for k, v in self._instances.items():
                print(f"{k} => {v}")
            print(f'\nNow you can call devices using \ne.g. \n{k} \nor \ndm = get_devices()\ndm.{k}')
        else:
            print("(no devices loaded)")
        if self._lookup is not None:
            self._lookup.update(self._instances)

    def __getattr__(self, name: str):
        if name in self._instances:
            return self._instances[name]
        raise AttributeError(f"No such device: {name!r}")



def simple_hashable_value(v):
    # if dict, converted to a tuple
    if isinstance(v, dict):
        return tuple(sorted(v.items()))
    return v

class SingletonAndCloseMeta(ABCMeta):
    # make sure all physical devices (same device class and same unique_id) only have one instance
    # mutiple initialization will get the same instance if params for initilization are not changed
    # and also register close() to atexit, and close will make sure next time is new init

    # Dictionary to store the instance and its initialization key for each class
    _instance_map = {}

    def __call__(cls, *args, **kwargs):
        # Convert args: if any arg is a dict, convert it to a sorted tuple of items
        hashable_args = tuple(simple_hashable_value(x) for x in args)
        # For kwargs, sort the items and convert dict values if needed
        hashable_kwargs = tuple(sorted((k, simple_hashable_value(v)) for k, v in kwargs.items()))
        # Use these to form a unique key
        device_key = (hashable_args, hashable_kwargs)
        if 'unique_id' not in kwargs:
            raise TypeError(f"{cls.__name__} need unique_id for init")
        unique_id = kwargs.get('unique_id', None)
        map_key = (cls, unique_id)
        
        if map_key in cls._instance_map:
            old_key, old_instance = cls._instance_map[map_key]
            if old_key == device_key:
                # If the initialization parameters match, return the existing instance
                return old_instance
            else:
                # If the parameters differ, and unique_id same means same physical device
                # then closing the existing instance
                try:
                    old_instance.close()
                except Exception as e:
                    pass
        else:
            pass
            # means a new instance with different physical device
        
        # Create a new instance and register its close() method for program exit
        instance = super().__call__(*args, **kwargs)
        instance._thread_lock = threading.Lock()
        # add thread_lock in case need to use

        close_meth = getattr(instance, "close", None)
        if callable(close_meth):
            orig_close = close_meth
            inst_ref = weakref.ref(instance)

            @functools.wraps(orig_close)
            def _wrapped_close(*a, **kw):
                try:
                    return orig_close(*a, **kw)
                finally:
                    inst = inst_ref()
                    if inst is not None:
                        cur = cls._instance_map.get(map_key)
                        if cur and cur[1] is inst:
                            cls._instance_map.pop(map_key, None)


            instance.close = _wrapped_close
            atexit.register(_wrapped_close)
        # make sure reinit the class if previous instances was closed, even with same keys to init

        cls._instance_map[map_key] = (device_key, instance)
        return instance

class RemoteDevice(metaclass=SingletonAndCloseMeta):
    # not singleton to initiate mutiple devices if needed, but need to close
    _LOCAL_ATTRS = {"remote_device", "_device_cls", '_conn', '_thread_lock', '_gui_window', 'close'}

    def __init__(self, unique_id, ip='127.0.0.1', port=50000):
        conn = rpyc.utils.classic.ssl_connect(
            host=ip,
            port=port,
            ca_certs=os.path.join(CERTIFICATE_DIR, "server.crt"),
        )

        remote = conn.root.get_device()
        object.__setattr__(self, "_conn", conn)
        object.__setattr__(self, "remote_device", remote)
        mod_name = remote.__class__.__module__
        cls_name = remote.__class__.__name__
        mod = importlib.import_module(mod_name)
        device_cls = getattr(mod, cls_name)
        object.__setattr__(self, "_device_cls", device_cls)
        # looking for desired gui class at local
        # assuming HOST and Client share the same package
        object.__setattr__(self, "_gui_window", None)

        print(f"Connected to remote device at {ip}:{port} -> {mod_name}.{cls_name}")

    def gui(self, in_GUI=False):
        return self._device_cls.gui(self, in_GUI=in_GUI)

    def close(self):
        win = getattr(self, "_gui_window", None)
        if win is not None:
            try:
                win.close()
            except Exception:
                pass
            self._gui_window = None

        conn = getattr(self, "_conn", None)
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass
            self._conn = None

    def __getattr__(self, name):
        if name == 'gui_dict':
            gd = self.remote_device.gui_dict.copy()
            gd.pop('allow_remote', None)
            return gd
            # disable remotedevice access allow_remote method
        else:
            return getattr(self.remote_device, name)

    def __setattr__(self, name, value):
        if name in RemoteDevice._LOCAL_ATTRS:
            object.__setattr__(self, name, value)
        else:
            value_for_rpyc = python2plain(value)
            # to pass copy not ref otherwise affected by connection
            setattr(self.remote_device, name, value_for_rpyc)

    # calling to class method/property will look for __getattribute__, __setattribute__
    # first and if fail then try __getattr__, __setattr__


class BaseDevice(ABC, metaclass=SingletonAndCloseMeta):
    """
    define lb, ub, valid if needed

    self.frequency_lb = 0.1e9 #GHz
    self.frequency_ub = 3.5e9 #GHz
    self.power_lb = -50 #dbm
    self.power_ub = 10 #dbm
    self.letter_valid = ['a', 'b', 'c']

    @BaseDevice.ManagedProperty(gui_type='float')
    def frequency(self):
        return self._frequency
    
    @frequency.setter
    def frequency(self, value):
        self._frequency = value

    ManagedProperty will use lb, ub, valid to determine if input is valid
    """

    class ManagedProperty:
        """
        Descriptor for instrument attributes:

        Metadata:
          - gui_type: one of 'float', 'bool', 'str', 'func' (for GUI rendering or binding)
          - monitor: bool, if True starts a background thread to poll the getter
          - interval: polling interval in seconds when monitoring

        Behavior:
          1. Registers metadata for automatic GUI generation.
          2. Validates inputs in the setter:
             - Floats are clamped to [lb, ub] if defined
             - Bools must be True or False
             - Strings must be in a predefined valid list
          3. Caches values when monitoring to avoid blocking hardware on each access.
          4. Supports method binding when gui_type='func'.
        """
        def __init__(self, gui_type, monitor=False, interval=0.2, thread_safe=False):
            self.meta = {
                'gui_type': gui_type, 'monitor': monitor, 'interval': interval, 'thread_safe': thread_safe
            }
            self.fget = None
            self.fset = None

        def __call__(self, fget):
            self.fget = fget
            return self

        def __set_name__(self, owner, name):
            """
            one time run after class initiate
            """
            self.name = name
            # merge gui_dict from all bases (right-to-left to respect MRO)
            merged = {}
            for base in reversed(owner.__mro__[1:]):  # skip owner itself
                if hasattr(base, 'gui_dict'):
                    merged.update(getattr(base, 'gui_dict'))
            if 'gui_dict' not in owner.__dict__:
                owner.gui_dict = dict(merged)

            owner.gui_dict[name] = {
                'gui_type': self.meta['gui_type'],
                'monitor': self.meta['monitor'],
                'interval': self.meta['interval'],
                'thread_safe': self.meta['thread_safe'],
            }

        def getter(self, fget):
            self.fget = fget
            return self

        def setter(self, fset):
            @functools.wraps(fset)
            def wrapper(instance, value):

                lock = getattr(instance, '_thread_lock', None) if self.meta.get('thread_safe') else None

                v = python2plain(value)
                def _validate_and_set(value):
                    name = self.fget.__name__
                    lb    = python2plain(getattr(instance, f"{name}_lb", None))
                    ub    = python2plain(getattr(instance, f"{name}_ub", None))
                    valid = python2plain(getattr(instance, f"{name}_valid", None))

                    if self.meta['gui_type']=='float':
                        if lb is not None and value < lb:
                            #print(f"{name} too low, clipped to {lb}")
                            value = lb
                        if ub is not None and value > ub:
                            #print(f"{name} too high, clipped to {ub}")
                            value = ub
                    elif self.meta['gui_type']=='bool':
                        if not isinstance(value, bool):
                            raise ValueError(f"{name} must be a bool")
                    elif self.meta['gui_type']=='str':
                        if valid is not None and value not in valid:
                            raise ValueError(f"{name} must be one of {valid}")

                    return fset(instance, value)

                if lock:
                    with lock:
                        return _validate_and_set(v)
                else:
                    return _validate_and_set(v)

            self.fset = wrapper
            return self

        def __get__(self, instance, owner):
            if instance is None:
                return self

            lock = getattr(instance, '_thread_lock', None) if self.meta.get('thread_safe') else None

            if self.meta['gui_type'] == 'func':
                # Build a "bound" signature that hides 'self'
                orig_sig = inspect.signature(self.fget)
                params = list(orig_sig.parameters.values())
                if params and params[0].name in ('self',):  # drop self
                    params = params[1:]
                bound_sig = orig_sig.replace(parameters=params)

                if lock:
                    @functools.wraps(self.fget)                       # <-- keep name/doc/__wrapped__
                    def locked_method(*args, **kwargs):
                        with lock:
                            return self.fget(instance, *args, **kwargs)
                    locked_method.__signature__ = bound_sig           # <-- show real params in GUI
                    return locked_method
                else:
                    method = functools.partial(self.fget, instance)   # bound to instance
                    try:
                        method.__signature__ = bound_sig              # make sure inspect sees it
                    except Exception:
                        pass
                    return method

            if self.meta['monitor']:
                if getattr(instance, f"_{self.name}_monitor_thread", None) is None:
                    self._start_monitor(instance)
                return getattr(instance, f"_{self.name}")

            if lock:
                with lock:
                    return self.fget(instance)
            else:
                return self.fget(instance)

        def __set__(self, instance, value):
            if not self.fset:
                raise AttributeError("No setter defined")
            return self.fset(instance, value)


        def _start_monitor(self, instance):
            stop_evt = threading.Event()

            lock = getattr(instance, '_thread_lock', None) if self.meta.get('thread_safe') else None

            def _poll():
                while not stop_evt.is_set():
                    try:
                        if lock:
                            with lock:
                                val = self.fget(instance)
                        else:
                            val = self.fget(instance)

                        setattr(instance, f"_{self.name}", val)
                    except Exception as e:
                        log_error(e)
                        self._stop_monitor(instance)
                        # if error then stop monitor
                    time.sleep(self.meta['interval'])

            t = threading.Thread(target=_poll, daemon=True)

            if lock:
                with lock:
                    val = self.fget(instance)
            else:
                val = self.fget(instance)
            setattr(instance, f"_{self.name}", val)
            setattr(instance, f"_{self.name}_monitor_thread", t)
            setattr(instance, f"_{self.name}_monitor_stop", stop_evt)

            t.start()

        def _stop_monitor(self, instance):
            stop_evt = getattr(instance, f"_{self.name}_monitor_stop", None)
            if stop_evt:
                stop_evt.set()

    # no @BaseDevice.ManagedProperty cause inside BaseDevice
    @ManagedProperty(gui_type='func')
    def allow_remote(self, enable=True, host="0.0.0.0", port=None):

        import socket, struct
        def fast_close(conn):
            try:
                sock = conn._channel.stream.sock
                # SO_LINGER
                sock.setsockopt(
                    socket.SOL_SOCKET, socket.SO_LINGER,
                    struct.pack('ii', 1, 0)
                )
                sock.close()
            except Exception as e:
                log_error(e)

        if not hasattr(self, "_active_conns"):
            self._active_conns = set()
        if not enable:
            for conn in list(self._active_conns):
                fast_close(conn)

            self._active_conns.clear()
            if hasattr(self, "_rpc_server") and self._rpc_server:
                self._rpc_server.close()
            if hasattr(self, "_rpc_thread") and self._rpc_thread:
                self._rpc_thread.join(timeout=1)
            self._rpc_server = None
            self._rpc_thread = None
            print(
                f"[allow_remote] disabled: "
                f"closed connections, "
                f"RPC server and thread stopped."
            )
            return

        if not hasattr(self, "_active_conns"):
            self._active_conns = set()
        thr = getattr(self, '_rpc_thread', None)
        if thr and thr.is_alive():
            return

        if port is None:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.bind((host, 0))
            chosen_port = sock.getsockname()[1]
            sock.close()
        else:
            chosen_port = port
        parent = self
        class RFService(SlaveService):
            def on_connect(self, conn):
                parent._active_conns.add(conn)
            def on_disconnect(self, conn):
                parent._active_conns.discard(conn)
            def exposed_get_device(self):
                return _make_plain_proxy(parent)  # only getattr is sanitized

        def _make_plain_proxy(target):
            """
            Server-side lightweight proxy:
            - __getattr__: non-callable -> python2plain(value) (by-value, numpy-safe)
                           callable     -> return original callable (Rpyc will netref it)
            - __setattr__: forward to target untouched (client RemoteDevice sanitizes on setattr)
            - Mimic target class module/name so client import-by-name keeps working.
            """
            tgt_cls = type(target)
            ProxyCls = type(tgt_cls.__name__, (), {})
            ProxyCls.__module__ = tgt_cls.__module__

            def __init__(self, _t):
                object.__setattr__(self, "_target", _t)

            def __getattr__(self, name):
                val = getattr(self._target, name)
                if callable(val):
                    return val  # let RPyC create a netref
                return python2plain(val)  # plainify non-callables

            def __setattr__(self, name, value):
                if name == "_target":
                    object.__setattr__(self, name, value)
                else:
                    setattr(self._target, name, value)  # client already sanitized

            ProxyCls.__init__ = __init__
            ProxyCls.__getattr__ = __getattr__
            ProxyCls.__setattr__ = __setattr__
            return ProxyCls(target)

        class QuietSSLAuthenticator(SSLAuthenticator):
            def __call__(self, sock):
                import ssl
                try:
                    return super().__call__(sock)
                except (ConnectionResetError, ssl.SSLError, OSError) as e:
                    try:
                        sock.close()
                    except Exception:
                        pass
                    raise AuthenticationError(str(e))
            # avoid weird WinError 10054 error

        key_path = os.path.join(CERTIFICATE_DIR, "server.key")
        certfile_path = os.path.join(CERTIFICATE_DIR, "server.crt")
        auth = QuietSSLAuthenticator(keyfile=key_path, certfile=certfile_path)
        server = ThreadedServer(
            RFService,
            hostname=host,
            port=chosen_port,
            authenticator=auth,
            protocol_config={
                "allow_public_attrs": True,
                "allow_all_attrs":    True,
                "allow_setattr":      True,
                "allow_pickle":       True,
            }
        )
        t = threading.Thread(target=server.start, daemon=True)
        t.start()
        self._rpc_server = server
        self._rpc_thread = t

        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        external_ip = s.getsockname()[0]
        s.close()
        print(f"Remote service listening on {host}:{chosen_port}, external reachable at {external_ip}:{chosen_port}")

    def close(self):
        pass

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        orig_close = getattr(cls, 'close', None)

        @functools.wraps(orig_close)
        def close_and_cleanup(self, *args, **kw):
            for prop_name, cfg in type(self).gui_dict.items():
                if cfg.get('monitor'):
                    descriptor = getattr(type(self), prop_name)
                    try:
                        descriptor._stop_monitor(self)
                    except Exception as e:
                        log_error(e)
            return orig_close(self, *args, **kw) if callable(orig_close) else None

        cls.close = close_and_cleanup

    def snapshot(self):
        """
        Take a deep-copied, point-in-time snapshot of ManagedProperty values.

        - Skips entries whose gui_type == 'func'.
        - Uses an internal exclude set (default excludes 'is_cancel').
        - Accesses values via getattr(self, name), which may start monitor threads
          for monitored properties by design.
        - Deep-copies each value to freeze the current state.
        """
        snap = {'__class__': type(self).__name__,}
        gui = getattr(type(self), 'gui_dict', {})
        exclude = {'is_cancel'}  # subclasses can override snapshot() to change this

        for name, meta in gui.items():
            if meta.get('gui_type') == 'func' or name in exclude:
                continue
            try:
                val = getattr(self, name)          # OK to trigger monitor
                snap[name] = copy.deepcopy(val)    # freeze current state
            except Exception as e:
                log_error(e)  # don't break whole snapshot on single failure
        return snap


    def gui(self, in_GUI=False):
        """
        Use self.gui_dict to determine how to display configurable parameters
        self.gui_dict is generated by ManagedProperty
        """
        if getattr(self, '_gui_window', None) is None:
            self._gui_window = run_fluent_window(use_fluent=True, widget_class=DeviceGUI, widget_kwargs={'device':self}, 
            title='DeviceGUI@Wanglab, UOregon', ui_path=None, in_GUI=in_GUI)
        else:
            self._gui_window = run_fluent_window(in_GUI=in_GUI, window_handle = self._gui_window)

        return self._gui_window


    @ManagedProperty(gui_type='bool')
    def is_cancel(self):
        ev = getattr(self, "_cancel_event", None)
        return ev.is_set() if ev else False

    @is_cancel.setter
    def is_cancel(self, value):
        ev = getattr(self, "_cancel_event", None)
        if ev is None:
            ev = threading.Event()
            setattr(self, "_cancel_event", ev)
        if value is True:
            ev.set()
        else:
            ev.clear()

    # ---- helper for readable cancel logs ----
    def _device_label(self):
        """Return a stable label for this device: ClassName<unique_id or hex(id)>"""
        uid = getattr(self, "unique_id", None)
        if uid is None:
            # Try to discover unique_id from the metaclass registry
            try:
                for (klass, uid1), (_key, inst) in SingletonAndCloseMeta._instance_map.items():
                    if inst is self:
                        uid = uid1
                        break
            except Exception:
                pass
        if uid is not None:
            return f"{type(self).__name__}<{uid}>"
        return f"{type(self).__name__}<{hex(id(self))}>"

    def _cancel_log(self, where, **extras):
        """Print a unified cancel message with optional extra fields."""
        extra_str = ", ".join(f"{k}={v}" for k, v in extras.items() if v is not None)
        if extra_str:
            print(f"[CANCEL] {self._device_label()} @ {where}: {extra_str}")
        else:
            print(f"[CANCEL] {self._device_label()} @ {where}")

    def time_sleep(self, exposure):
        # performance good and stable version of time.sleep
        # but not used in e.g. laser_stabilizer _loop
        self.chunk_s = 0.2 # check abort event every chunk_s second
        s_left = exposure
        while s_left > 0:
            if self.is_cancel:
                self._cancel_log("time_sleep", remaining=f"{s_left:.3f}s")
                return False
            sleep_time = int(np.ceil(min(s_left, self.chunk_s) * 1000))
            loop = QEventLoop()
            QTimer.singleShot(sleep_time, loop.quit)
            loop.exec_()
            s_left -= self.chunk_s
        return True


class BaseLaser(BaseDevice):

    @property
    @abstractmethod
    def wavelength(self):
        pass
    
    @wavelength.setter
    @abstractmethod
    def wavelength(self, value):
        pass

    @property
    @abstractmethod
    def piezo(self):
        pass
    
    @piezo.setter
    @abstractmethod
    def piezo(self, value):
        pass            

class BaseRF(BaseDevice):

    @property
    @abstractmethod
    def frequency(self):
        pass
    
    @frequency.setter
    @abstractmethod
    def frequency(self, value):
        pass

    @property
    @abstractmethod
    def power(self):
        pass
    
    @power.setter
    @abstractmethod
    def power(self, value):
        pass

    @property
    @abstractmethod
    def on(self):
        pass
    
    @on.setter
    @abstractmethod
    def on(self, value):
        pass


class BaseWavemeter(BaseDevice):

    @property
    @abstractmethod
    def wavelength(self):
        pass
    
class BaseCounter(BaseDevice):

    @property
    @abstractmethod
    def counter_mode_valid(self):
        pass

    @property
    @abstractmethod
    def data_mode_valid(self):
        pass

    @property
    @abstractmethod
    def data_mode(self):
        pass

    @property
    @abstractmethod
    def counter_mode(self):
        pass

    @data_mode.setter
    def data_mode(self, value):
        pass

    @counter_mode.setter
    def counter_mode(self, value):
        pass

    @property
    @abstractmethod
    def data_len(self):
        pass

    @abstractmethod
    def read_counts(self, exposure, parent=None):
        pass
    # parent param to get class name of caller for test


class BaseScanner(BaseDevice):

    @property
    @abstractmethod
    def x(self):
        pass
    
    @x.setter
    @abstractmethod
    def x(self, value):
        pass

    @property
    @abstractmethod
    def y(self):
        pass
    
    @y.setter
    @abstractmethod
    def y(self, value):
        pass

class BaseLaserStabilizer(BaseDevice):
    """
    Base class to 
    
    """
    def __init__(self, unique_id, wavemeter_handle:BaseDevice, laser_handle:BaseDevice, freq_thre=0.015, freq_deadzone=0.005,
        wavelength_lb=None, wavelength_ub=None):
        self.is_ready = False
        self._wavelength = None # wavelength that user inputs
        self.desired_wavelength = None # used for feedback
        self.is_running = True
        self._on = False
        # indicate if wavelnegth has changed
        self.wavemeter = wavemeter_handle
        self.laser = laser_handle

        self.spl = 299792458
        self.freq_thre = freq_thre # default 0.015=15MHz threshold defines when to return is_ready
        self.freq_deadzone = 0.015 # if error less than freq_deadzone then no actions
        self.wavelength_lb = wavelength_lb
        self.wavelength_ub = wavelength_ub

        self._ready_event = threading.Event()
        self._evt_request = threading.Event()  # request from wait_to_wavelength
        self._evt_ack = threading.Event()      # loop paused and ready to apply
        self._evt_init = threading.Event()     # caller finished init (on + wavelength)

        self.dt = self._get_wavemeter_dt()
        print(f'Stabilizer using delta t at {self.dt}s')
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        # will be killed if main thread is killed


    @BaseDevice.ManagedProperty('float')
    def wavelength(self):
        return self._wavelength
    
    @wavelength.setter
    def wavelength(self, value):
        self._wavelength = value
        self.desired_wavelength = self._wavelength
        self.is_ready = False
        self._ready_event.clear()

    @BaseDevice.ManagedProperty('bool')
    def on(self):
        return self._on
    
    @on.setter
    def on(self, value):
        if value is False:
            self.is_ready = False
            self._ready_event.clear()
        self._on = value

    def _get_wavemeter_dt(self):
        gd = getattr(self.wavemeter, 'gui_dict', {})
        meta = gd.get('wavelength', {})
        monitor_on = bool(meta.get('monitor', False))
        if monitor_on:
            dt = float(meta.get('interval', 0.01))
        else:
            dt = 0.01
        return max(0.03, 3*dt)
        # minimum 30ms
                
    
    def _run(self):
        while self.is_running:    
            time.sleep(self.dt)
            wave_cache = self.wavemeter.wavelength #wait
            if (self.desired_wavelength is None) or (not self.on) or (wave_cache==0):
                pass
            else:
                freq_desired = self.spl/self.desired_wavelength
                freq_diff = freq_desired - self.spl/wave_cache
                if np.abs(freq_diff) <= self.freq_thre:
                    self.is_ready = True
                    self._ready_event.set()
                else:
                    self.is_ready = False
                if np.abs(freq_diff)>self.freq_deadzone:
                    self._stabilizer_core(freq_diff)

            if self._evt_request.is_set():
                self._evt_ack.set()          # ready_to_go
                self._evt_init.wait()        # wait until caller sets on/wavelength
                self._evt_init.clear()
                self._evt_request.clear()

    @BaseDevice.ManagedProperty('func', thread_safe=True)
    def wait_to_wavelength(self, wavelength):
        self._evt_request.set()
        self._evt_ack.wait()
        self.on = True
        self.wavelength = wavelength
        self._evt_init.set()
        while True:
            self.time_sleep(0.01)
            if self._ready_event.is_set():
                self._evt_ack.clear()
                self._evt_init.clear()
                self._evt_request.clear()
                break
            elif self.is_cancel:
                self._cancel_log("wait_to_wavelength", target=wavelength)
                self._evt_ack.clear()
                self._evt_init.clear()
                self._evt_request.clear()
                self.on = False
                break

    @abstractmethod
    def _stabilizer_core(self, freq_diff):
        pass


    def close(self):
        if self.thread.is_alive(): #use () cause it's a method not property
            self.is_running = False
            self.thread.join()


class BasePulse(BaseDevice):
    """
    Base class for pulse control
    """
    def __init__(self, t_resolution=(1,1)):
        self.t_resolution = max(t_resolution) 
        # minumum allowed pulse (width, resolution), (12, 2) for spin core, will round all time durations beased on this
        self._valid_str = ['+', '-', 'x', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'e', 'E', ' ']
        self.x = self.t_resolution # postive int in ns, a timing varible used to replace all 'x' in timing array and matrix, effective only in read_data()
        self.delay_array = np.array([0,0,0,0,0,0,0,0])
        self.data_matrix = np.array([[1000, 1,0,0,0,0,0,0,0], [1000, 1,0,0,0,0,0,0,0]])
        # example of data_matrix [[duration in ns, on or off for channel i, ...], ...]
        # self._data_matrix can also be np.array([['x', 1,1,1,1,1,1,1,1], ['1000-x', 1,1,1,1,1,1,1,1]])
        self.repeat_info = [0, -1, 1] # start_index, end_index(include), repeat_times

        self.ref_info = {"is_ref":False, "signal":None, "DAQ":None, "DAQ_ref":None, 'clock':None} #

        # _data_matrix for on_pulse(), data_matrix_tmp for reopen gui display
        self.channel_names = ['', '', '', '', '', '', '', '']
        # names of all channels, such as 'RF', 'Green', 'DAQ', 'DAQ_ref'
        self.total_duration = None
        self.on = False
        self.last_on_state = None


    def gui(self, in_GUI=False):
        """
        Use self.gui_dict to determine how to display configurable parameters
        self.gui_dict is generated by ManagedProperty
        """
        if getattr(self, '_gui_window', None) is None:
            self._gui_window = run_fluent_window(use_fluent=True, widget_class=PulseGUI, widget_kwargs={'device_handle':self}, 
            title='PulseGUI@Wanglab, UOregon', ui_path=None, in_GUI=in_GUI)
        else:
            self._gui_window = run_fluent_window(in_GUI=in_GUI, window_handle = self._gui_window)

        return self._gui_window

    @BaseDevice.ManagedProperty(gui_type='func')
    def allow_remote(self, enable=True, host="0.0.0.0", port=None):
        print('[allow_remote] Pulse is not allowed to be exposed remotely.')
        return


    gui.__doc__ = PulseGUI.__doc__

    def round_up(self, t, allow_any=False):
        # round up t into multiple of self.t_resolution
        return align_to_resolution(value=t, resolution=self.t_resolution, allow_any=allow_any)

    def set_x_bound(self):
        """
        Compute feasible x range so that every period duration >= t_resolution.
        Support integer multipliers on x (±k x).

        For each duration row:
            Let d(x) = A + B*x, where
                A = d(0)
                B = d(1) - d(0)
            Constraint: A + B*x >= t_resolution
              - if B > 0: x >= (t_resolution - A)/B
              - if B < 0: x <= (A - t_resolution)/(-B)
              - if B = 0: (no x in this row) skip; but we only enter when 'x' in string.
        Aggregate:
            x_lb = max(all lower bounds), x_ub = min(all upper bounds)
        """
        min_x_candidate = []  # lower bounds
        max_x_candidate = []  # upper bounds

        for period in self.data_matrix:
            duration = period[0]
            if isinstance(duration, str) and ('x' in duration):
                # Make '2x' -> '2*x' before eval
                s = insert_mul_before_x(str(duration))
                try:
                    A = eval(s.replace('x', '0'))
                    B = eval(s.replace('x', '1')) - A
                except Exception:
                    # If unparsable, skip this row conservatively
                    continue

                # Only linear forms with a single x-term are allowed by your regex;
                # still, guard B == 0 just in case.
                if abs(B) < 1e-15:
                    # effectively constant; ignore (it doesn't constrain x)
                    continue

                # A + B*x >= t_resolution
                if B > 0:
                    lb = (self.t_resolution - A) / B
                    min_x_candidate.append(lb)
                else:  # B < 0
                    ub = (A - self.t_resolution) / (-B)
                    max_x_candidate.append(ub)

        min_x = max(min_x_candidate) if min_x_candidate else None
        max_x = min(max_x_candidate) if max_x_candidate else None

        self.x_lb = min_x
        self.x_ub = max_x

        if (min_x is not None) or (max_x is not None):
            print(f"Set x_lb, x_ub to {self.x_lb}ns, {self.x_ub}ns due to every period in data_matrix must be >= {self.t_resolution}ns")

        self.x = self.x
        # update x in case bound changes


    @property
    def delay_array(self):
        return self._delay_array
    
    @delay_array.setter
    def delay_array(self, value):
        if len(value)!= 8:
            print('invalid delay array length')
            return
        if not all(isinstance(item, (Number, str)) for item in value):
            print('Invalid delay array content. Must only contain int numbers in ns or str contains x for time variable.')
            return
        if not all(isinstance(item, Number) or all(elem in self._valid_str for elem in item) for item in value):
            print(f"Invalid input. Can only be one of {self._valid_str}.")
            return
        self._delay_array = [self.round_up(delay, allow_any=True) for delay in value]

    @property
    def data_matrix(self):
        return self._data_matrix
    
    @data_matrix.setter
    def data_matrix(self, value):
        if len(value)< 2:
            print('invalid data_matrix length')
            return
        if not all(
            len(item) == 9 and 
            (isinstance(item[0], (Number, str))) and 
            all(elem in (0, 1) for elem in item[1:])
            for item in value
        ):
            # must be length 9, item[0] must be int in ns or 'x' str, item in item[1:] must be 1 or 0
            print("Invalid input. Each item must meet the conditions.")
            return
        for period in value:
            if not isinstance(period[0], Number):
                if not all(letter in self._valid_str for letter in period[0]):
                    print(f"Invalid input. Can only be one of {self._valid_str}.") 
                    return

        self._data_matrix = [[self.round_up(item, allow_any=False) if i==0 else int(item) 
            for i, item in enumerate(period)] for period in value]
        self.set_x_bound()

    @property
    def repeat_info(self):
        return self._repeat_info
    
    @repeat_info.setter
    def repeat_info(self, value):
        if len(value)!= 3:
            print('invalid repeat_info length')
            return
        if not all(isinstance(item, (Number, str)) for item in value):
            print('Invalid repeat_info content.')
            return
        if not ((0<=value[0]<=(len(self.data_matrix)-2)) and ((value[1]-value[0])>=1 or (value[1]==-1))):
            print('Invalid repeat_info content.')
        self._repeat_info = [int(item) for item in value]
        # must all be integers

    @property
    def ref_info(self):
        return self._ref_info
    
    @ref_info.setter
    def ref_info(self, value):
        if not isinstance(value, dict):
            print('invalid ref_info, example {"is_ref":False, "signal":None, "DAQ":None, "DAQ_ref":None, "clock":None}')
            return
        if value.get('is_ref', None) not in [True, False]:
            print('Invalid ref_info["is_ref"].')
            return
        if not all(value.get(key, True) in [None, 0, 1, 2, 3, 4, 5, 6, 7] for key in ['signal', 'DAQ', 'DAQ_ref', 'clock']):
            print('Invalid ref_info["signal"] or ref_info["DAQ"] or ref_info["DAQ_ref"] or ref_info["clock"]')

        self._ref_info = {}
        for key, channel in value.items():
            self._ref_info[key] = channel
        # must all be integers

    @property
    def channel_names(self):
        return self._channel_names

    @channel_names.setter
    def channel_names(self, value):
        if len(value) != 8:
            print('invalid channel_names length')
            return
        self._channel_names = [str(x) for x in list(value)]

    def snapshot(self):
        return {
            'data_matrix':  copy.deepcopy(self.data_matrix),
            'delay_array':  copy.deepcopy(self.delay_array),
            'channel_names':copy.deepcopy(self.channel_names),
            'ref_info':     copy.deepcopy(self.ref_info),
            'repeat_info':  copy.deepcopy(self.repeat_info),
        }

    @BaseDevice.ManagedProperty(gui_type='float')
    def x(self):
        return self._x
    
    @x.setter
    def x(self, value):
        self._x = self.round_up(value, allow_any=True)
        # bound is also at resolution grid so will not round out of bounds

    @property
    def on(self):
        return self._on
    
    @on.setter
    def on(self, value):
        if value in (True, False):
            self._on = value

    def off_pulse(self):
        self.off_pulse_core()
        self.on = False
   
    def on_pulse(self):
        self.off_pulse()
        self.on_pulse_core()
        self.on = True
        self.last_on_state = self.snapshot()

    @abstractmethod    
    def off_pulse_core(self):
        # rewrite this method for real pulse
        pass

    @abstractmethod    
    def on_pulse_core(self):
        # rewrite this method for real pulse
        pass

    def set_timing_simple(self, timing_matrix):
        # set_timing_simple([[duration0, [channels0, ]], [], [], []])
        # eg. 
        # set_timing_simple([[100, (3,)], [100, (3,5)]])
        # channel0 - channel7
        n_sequence = len(timing_matrix)
        if n_sequence <= 2:
            print('pulse length must larger than 2')
            return 
        data_matrix = [[0]*9 for _ in range(n_sequence)]

        for i in range(n_sequence):
            if isinstance(timing_matrix[i][0], str):
                data_matrix[i][0] = timing_matrix[i][0]
            else:
                data_matrix[i][0] = int(timing_matrix[i][0]) # avoid possible float number ns duration input

            for channel in timing_matrix[i][1]:
                data_matrix[i][channel+1] = 1


        self.data_matrix = data_matrix


    def load_x_to_str(self, timing, time_type):
        if time_type == 'delay':
            if isinstance(timing, Number):
                return int(timing)
            if isinstance(timing, str):
                s = insert_mul_before_x(f'{timing}')
                return eval(s.replace('x', str(self.x)))
        elif time_type == 'duration':
            # must larger than 0
            if isinstance(timing, Number):
                if int(timing) > 0:
                    return int(timing)
                else:
                    return self.t_resolution
            if isinstance(timing, str):
                s = insert_mul_before_x(f'{timing}')
                v = eval(s.replace('x', str(self.x)))
                return v if v > 0 else self.t_resolution

    def total_duration_str(self, data_matrix=None, repeat_info=None, ref_info=None):
        """
        Return a simplified symbolic total duration as 'A±Bx' (in ns), considering
        repeat and ref. No numeric evaluation of x; we only combine constants and
        the coefficient of 'x'.

        The duration strings are assumed to be sums of numbers and +/- 'x'
        (matching your existing allowed chars: digits, '+', '-', 'x', 'e', 'E', spaces).
        """

        dm = data_matrix
        ri = repeat_info
        rf = ref_info

        if any(x is None for x in (dm, ri, rf)):
            return ''

        # Normalize repeat indices
        start = int(ri[0]) if ri is not None else 0
        end_excl = (int(ri[1]) + 1) if (ri is not None and ri[1] != -1) else len(dm)
        rpt = int(ri[2]) if ri is not None else 1

        def _accum(section):
            c = 0; a = 0
            for row in section:
                d = row[0]
                if isinstance(d, Number):
                    c += int(d)
                else:
                    s = insert_mul_before_x(str(d))
                    c0 = eval(s.replace('x', '0'))
                    c1 = eval(s.replace('x', '1'))
                    c += int(c0)
                    a += int(c1 - c0)
            return c, a

        c_pre, a_pre   = _accum(dm[:start])
        c_mid, a_mid   = _accum(dm[start:end_excl])
        c_post, a_post = _accum(dm[end_excl:])

        c_tot = c_pre + rpt * c_mid + c_post
        a_tot = a_pre + rpt * a_mid + a_post

        # If reference is enabled, the timeline doubles
        if isinstance(rf, dict) and rf.get('is_ref', False):
            c_tot *= 2
            a_tot *= 2

        # Build compact string: "A", "Bx", "A+Bx", "A-Bx" without spaces
        if a_tot == 0:
            return str(int(c_tot))
        coef = "x" if abs(a_tot) == 1 else f"{abs(a_tot)}x"
        if c_tot == 0:
            return (coef if a_tot > 0 else f"-{coef}")
        return f"{int(c_tot)}+{coef}" if a_tot > 0 else f"{int(c_tot)}-{coef}"
                    
    def read_data(self, type='time_slices'):
        # type ='time_slices' or 'data_matrix'
        valid_type = ['time_slices', 'data_matrix']
        if type not in valid_type:
            print(f'type must be one of {valid_type}')
            return

        # return delayed time_slices [[[t0, 1], [t1, 0], ...], [], [],...] for all channels
        
        data_matrix = self.data_matrix
        # data_matrix is [[1e3, 1, 1, 1, 1, 1, 1, 1, 1], ...]
        start_index = self.repeat_info[0]
        end_index = len(self.data_matrix) if (self.repeat_info[1]==-1) else (self.repeat_info[1]+1)
        time_slices = []
        for channel in range(8):
            time_slice = []

            for period in data_matrix[:start_index]:
                time_slice.append([self.load_x_to_str(period[0], 'duration'), period[1:][channel]])
            # before repeat sequence
            for repeat in range(int(self.repeat_info[2])):
                for period in data_matrix[start_index:end_index]:
                    time_slice.append([self.load_x_to_str(period[0], 'duration'), period[1:][channel]])
            # repeat sequence
            for period in data_matrix[end_index:]:
                time_slice.append([self.load_x_to_str(period[0], 'duration'), period[1:][channel]])
            # after repeat sequence

            if self.ref_info['is_ref']:
                # repeat one more time with disabling 'signal' and replace 'DAQ' with 'DAQ_ref' channel
                def apply_ref(channel, period):
                    if channel==self.ref_info['signal']:
                        return 0
                    if channel==self.ref_info['DAQ']:
                        return 0
                    if channel==self.ref_info['clock']:
                        return 0
                    if channel==self.ref_info['DAQ_ref']:
                        return period[self.ref_info['DAQ']]
                    return period[channel]

                for period in data_matrix[:start_index]:
                    time_slice.append([self.load_x_to_str(period[0], 'duration'), apply_ref(channel, period[1:])])
                # before repeat sequence
                for repeat in range(int(self.repeat_info[2])):
                    for period in data_matrix[start_index:end_index]:
                        time_slice.append([self.load_x_to_str(period[0], 'duration'), apply_ref(channel, period[1:])])
                # repeat sequence
                for period in data_matrix[end_index:]:
                    time_slice.append([self.load_x_to_str(period[0], 'duration'), apply_ref(channel, period[1:])])
                # after repeat sequence


            time_slice_delayed = self.delay(self.load_x_to_str(self.delay_array[channel], 'delay'), time_slice)
            time_slices.append(time_slice_delayed)

        if type == 'time_slices':
        
            return [[[period[0], period[1]] for period in time_slice] for time_slice in time_slices]

        elif type == 'data_matrix':
            # process, convert time_slices to data_matrix_delayed
            data_matrix_delayed = self._time_slices_to_data_matrix(time_slices)
            return [[period[i] if i==0 else period[i] for i in range(len(period))] for period in data_matrix_delayed]

    def _time_slices_to_data_matrix(self, time_slices):
        data_matrix = []
        while len(time_slices[0])!=0:
            t_cur = np.min([time_slice[0][0] for time_slice in time_slices])
            # find the minimum time of first period of all channels
            period_enable = [time_slice[0][1] for time_slice in time_slices]
            data_matrix.append([t_cur, ] + period_enable)
            for i in range(len(time_slices)):
                time_slices[i][0][0] -= t_cur
                if time_slices[i][0][0]==0:
                    time_slices[i] = time_slices[i][1:]

        return data_matrix


    def save_to_file(self, addr=''):

        current_time = time.localtime()
        current_date = time.strftime("%Y-%m-%d", current_time)
        current_time_formatted = time.strftime("%H:%M:%S", current_time)
        time_str = current_date.replace('-', '_') + '_' + current_time_formatted.replace(':', '_')

        if addr=='' or ('/' not in addr):
            pass
        else:
            directory = os.path.dirname(addr)
            if not os.path.exists(directory):
                os.makedirs(directory)

        
        np.savez(addr + '.npz', data_matrix = np.array(self.data_matrix, dtype=object),
            delay_array = np.array(self.delay_array, dtype=object),
            channel_names = np.array(self.channel_names, dtype=object), 
            repeat_info = np.array(self.repeat_info, dtype=object),
            ref_info = np.array(self.ref_info, dtype=object)
        )

    def load_from_file(self, addr: str) -> bool:
        address = fuzzy_search(addr, DIR=get_pulses_dir(), file_type='.npz')

        loaded = np.load(address, allow_pickle=True)
        self.data_matrix = loaded['data_matrix']
        self.delay_array = loaded['delay_array']
        self.channel_names = loaded['channel_names']
        self.repeat_info = loaded['repeat_info']
        self.ref_info = loaded['ref_info'].item()

        return True

    
    def delay(self, delay, time_slice):
        # accept time slice
        # example of time slice [[duration in ns, on or off], ...]
        # [[1e3, 1], [1e3, 0], ...] 
        # add delay to time slice (mod by total duration)

        total_duration = 0
        for period in time_slice:
            total_duration += period[0]

        self.total_duration = total_duration

        delay = delay%total_duration

        if delay == 0:
            return time_slice


        # below assumes delay > 0
        cur_time = 0
        for ii, period in enumerate(time_slice[::-1]):
            # count from end of pulse for delay > 0
            cur_time += period[0]
            if delay == cur_time:
                return time_slice[-(ii+1):] + time_slice[:-(ii+1)]
                # cycle roll the time slice to right (ii+1) elements
            if delay < cur_time:
                duration_lhs = cur_time - delay
                # duration left on the left hand side of pulse
                duration_rhs = period[0] - duration_lhs

                time_slice_lhs = time_slice[:-(ii+1)] + [[duration_lhs, period[1]], ]
                time_slice_rhs = [[duration_rhs, period[1]], ] + time_slice[-(ii+1):][1:] # skip the old [t_ii, enable_ii] period
                return time_slice_rhs + time_slice_lhs

            # else will be delay > cur_time and should continue



class BaseCounterNI(BaseCounter):
    # Basecounter class for NI-DAQ board/card
    # defines how to setup counter tasks and analogin tasks

    def __init__(self, port_config):
        super().__init__()

        import nidaqmx
        from nidaqmx.constants import AcquisitionType
        from nidaqmx.constants import TerminalConfiguration
        from nidaqmx.stream_readers import AnalogMultiChannelReader

        defaults = {
            'dev_num':         'Dev2',
            'apd_signal':      'PFI3',
            'apd_gate':        'PFI4',
            'apd_gate_ref':    'PFI1',
            'apd_clock':       'PFI12',
            'analog_signal':   'ai0',
            'analog_gate':     'ai1',
            'analog_gate_ref': 'ai2',
        }

        cfg = {**defaults, **(port_config or {})}
        # if some in port_config will then override

        def last(seg: str) -> str:
            s = str(seg).strip().strip('/')
            return s.split('/')[-1] if s else ''
        dev = last(cfg['dev_num'])
        self.dev_num = f'/{dev}/'

        for key in (set(defaults) - {'dev_num'}):
            tok = last(cfg[key])
            setattr(self, key, f'{self.dev_num}{tok}')

        self.nidaqmx = nidaqmx
        self.exposure = 0.1
        self.sample_num_single_default = 1000
        self.sample_num = 1000
        self.exposure_single = 0.2 # defines the sample number counter reads from hardware every time
        self.exposure_single_lb = 0.1
        self.exposure_single_ub = 0.5
        self.analog_threshold = 2.7 # 2.7V as the threshold to identify gate signal
        self.apd_sample_check_interval = 0.005 # maximum 5ms overhead
        self.tasks_to_close = [] # tasks need to be closed after swicthing counter mode 
        self.base_counter_mode_valid = ['apd', 'analog', 'apd_sample']
        self.base_data_mode_valid = ['single', 'ref_div', 'ref_sub', 'dual']
        # base_xxx_valid defines the all availabel mode for base class of BaseCounterNI

        self.data_mode_valid = ['single', 'ref_div', 'ref_sub', 'dual']
        self.counter_mode_valid = ['apd', 'analog', 'apd_sample']
        self.counter_mode = 'apd'
        self.data_mode = 'single'

        dev = self.nidaqmx.system.System.local().devices[self.dev_num[1:-1]] # remove two /
        print(f'Max channel rate: {dev.ai_max_single_chan_rate}, {dev.ai_max_multi_chan_rate}')
        print(f'Support simultaneous sampling: {dev.ai_simultaneous_sampling_supported}')
        self.analog_clock_max = dev.ai_max_multi_chan_rate/3 if not dev.ai_simultaneous_sampling_supported else dev.ai_max_multi_chan_rate
        self.analog_clock_max = int(self.analog_clock_max//1e3)*1e3
        print(f'Set analog clock to: {self.analog_clock_max}')

    @BaseDevice.ManagedProperty('float')
    def exposure_single(self):
        return self._exposure_single

    @exposure_single.setter
    def exposure_single(self, value):
        self._exposure_single = value

    @property
    def counter_mode_valid(self):
        return self._counter_mode_valid

    @counter_mode_valid.setter
    def counter_mode_valid(self, value):
        if not all(mode in self.base_counter_mode_valid for mode in value):
            print(f'Can only be subset of the {self.base_counter_mode_valid}')
        else:
            self._counter_mode_valid = value

    @property
    def data_mode_valid(self):
        return self._data_mode_valid

    @data_mode_valid.setter
    def data_mode_valid(self, value):
        if not all(mode in self.base_data_mode_valid for mode in value):
            print(f'Can only be subset of the {self.base_data_mode_valid}')
        else:
            self._data_mode_valid = value

    @BaseDevice.ManagedProperty('str')
    def data_mode(self):
        return self._data_mode

    @BaseDevice.ManagedProperty('str')
    def counter_mode(self):
        return self._counter_mode

    @data_mode.setter
    def data_mode(self, value):
        self._data_mode = value

    @counter_mode.setter
    def counter_mode(self, value):
        if value != getattr(self, '_counter_mode', None):
            self._counter_mode = value # before call set_counter
            self.set_counter()
            self.set_timing(self.exposure, self.sample_num)

    @property
    def data_len(self):
        return 2 if self.data_mode=='dual' else 1
    # the return data type 


    def set_timing(self, exposure, sample_num):
        if self.counter_mode == 'apd':
            self.clock = 1e4 
            # sampling rate for edge counting, defines when transfer counts from DAQ to PC, should be not too large to accomdate long exposure
            self.buffer_size = int(1e6)
            self.task_counter_clock = self.nidaqmx.Task()
            self.task_counter_clock.co_channels.add_co_pulse_chan_freq(counter=self.dev_num+'ctr2', freq=self.clock, duty_cycle=0.5)
            # ctr2 clock for buffered edge counting ctr0 and ctr1
            self.task_counter_clock.timing.cfg_implicit_timing(sample_mode=self.nidaqmx.constants.AcquisitionType.CONTINUOUS)

            self.tasks_to_close += [self.task_counter_clock,]

            self.sample_num = int(round(self.clock*exposure))+1
            self.sample_num_single = min(self.sample_num, int(round(self.clock*self.exposure_single)))
            self.task_counter_ctr.timing.cfg_samp_clk_timing(self.clock, source = self.dev_num+'Ctr2InternalOutput',
                sample_mode=self.nidaqmx.constants.AcquisitionType.CONTINUOUS, samps_per_chan=self.buffer_size
            )
            self.task_counter_ctr_ref.timing.cfg_samp_clk_timing(self.clock, source = self.dev_num+'Ctr2InternalOutput',
                sample_mode=self.nidaqmx.constants.AcquisitionType.CONTINUOUS, samps_per_chan=self.buffer_size
            )

            self.exposure = exposure
            self.counts_main_array = np.zeros(self.sample_num_single, dtype=np.uint32)
            self.counts_ref_array = np.zeros(self.sample_num_single, dtype=np.uint32)
            self.reader_ctr = self.nidaqmx.stream_readers.CounterReader(self.task_counter_ctr.in_stream)
            self.reader_ctr_ref = self.nidaqmx.stream_readers.CounterReader(self.task_counter_ctr_ref.in_stream)

            self.task_counter_ctr.start()
            self.task_counter_ctr_ref.start()
            # start clock after counter tasks
            self.task_counter_clock.start()

        if self.counter_mode == 'apd_sample':
            self.clock = 1e4 
            # sampling rate for edge counting, defines when transfer counts from DAQ to PC, should be not too large to accomdate long exposure
            self.buffer_size = int(1e6)

            self.sample_num = sample_num + 1
            self.sample_num_single = min(self.sample_num, self.sample_num_single_default)
            self.task_counter_ctr.timing.cfg_samp_clk_timing(self.clock, source = self.apd_clock,
                sample_mode=self.nidaqmx.constants.AcquisitionType.CONTINUOUS, samps_per_chan=self.buffer_size
            )
            self.task_counter_ctr_ref.timing.cfg_samp_clk_timing(self.clock, source = self.apd_clock,
                sample_mode=self.nidaqmx.constants.AcquisitionType.CONTINUOUS, samps_per_chan=self.buffer_size
            )

            self.exposure = exposure
            self.counts_main_array = np.zeros(self.sample_num_single, dtype=np.uint32)
            self.counts_ref_array = np.zeros(self.sample_num_single, dtype=np.uint32)
            self.reader_ctr = self.nidaqmx.stream_readers.CounterReader(self.task_counter_ctr.in_stream)
            self.reader_ctr_ref = self.nidaqmx.stream_readers.CounterReader(self.task_counter_ctr_ref.in_stream)

            # configure arm trigger to start both tasks at same time
            for t in (self.task_counter_ctr, self.task_counter_ctr_ref):
                trig = t.triggers.arm_start_trigger
                trig.trig_type = self.nidaqmx.constants.TriggerType.DIGITAL_EDGE
                trig.dig_edge_src = self.dev_num + 'Ctr2InternalOutput'
                trig.dig_edge_edge = self.nidaqmx.constants.Edge.RISING
                t.control(self.nidaqmx.constants.TaskMode.TASK_COMMIT)

            self.task_sync_pulse = self.nidaqmx.Task()
            self.task_sync_pulse.co_channels.add_co_pulse_chan_time(
                counter=self.dev_num + 'ctr2', low_time=5e-6, high_time=5e-6
            )
            self.task_sync_pulse.timing.cfg_implicit_timing(
                sample_mode=self.nidaqmx.constants.AcquisitionType.FINITE, samps_per_chan=1
            )
            self.task_sync_pulse.control(self.nidaqmx.constants.TaskMode.TASK_COMMIT)
            self.tasks_to_close += [self.task_sync_pulse]


            self.task_counter_ctr.start()
            self.task_counter_ctr_ref.start()
            self.task_sync_pulse.start()

        elif self.counter_mode == 'analog':
            self.clock = self.analog_clock_max 
            # sampling rate for analog input, should be fast enough to capture gate signal for postprocessing
            self.sample_num = int(round(self.clock*exposure))+1
            self.sample_num_single = min(self.sample_num, int(round(self.clock*self.exposure_single)))
            self.buffer_size = int(1e6)
            self.task_counter_ai.timing.cfg_samp_clk_timing(self.clock, sample_mode=self.nidaqmx.constants.AcquisitionType.CONTINUOUS
                , samps_per_chan=self.buffer_size)
            self.exposure = exposure
            self.counts_array = np.zeros((3, self.sample_num_single), dtype=np.float64)
            self.counts_array_remain = np.zeros((3, self.sample_num%self.sample_num_single), dtype=np.float64)
            self.reader_analog = self.nidaqmx.stream_readers.AnalogMultiChannelReader(self.task_counter_ai.in_stream)

            self.task_counter_ai.start()


    def close_old_tasks(self):
        for task in self.tasks_to_close:
            task.stop()
            task.close()
        self.tasks_to_close = []

    def close(self):
        self.close_old_tasks()

    def set_counter(self):
        if self.counter_mode in ('apd', 'apd_sample'):
            self.close_old_tasks()

            self.task_counter_ctr = self.nidaqmx.Task()
            self.task_counter_ctr.ci_channels.add_ci_count_edges_chan(self.dev_num+'ctr0')
            self.task_counter_ctr.triggers.pause_trigger.dig_lvl_src = self.apd_gate
            self.task_counter_ctr.ci_channels.all.ci_count_edges_term = self.apd_signal
            self.task_counter_ctr.triggers.pause_trigger.trig_type = self.nidaqmx.constants.TriggerType.DIGITAL_LEVEL
            self.task_counter_ctr.triggers.pause_trigger.dig_lvl_when = self.nidaqmx.constants.Level.LOW

            self.task_counter_ctr_ref = self.nidaqmx.Task()
            self.task_counter_ctr_ref.ci_channels.add_ci_count_edges_chan(self.dev_num+'ctr1')
            self.task_counter_ctr_ref.triggers.pause_trigger.dig_lvl_src = self.apd_gate_ref
            self.task_counter_ctr_ref.ci_channels.all.ci_count_edges_term = self.apd_signal
            self.task_counter_ctr_ref.triggers.pause_trigger.trig_type = self.nidaqmx.constants.TriggerType.DIGITAL_LEVEL
            self.task_counter_ctr_ref.triggers.pause_trigger.dig_lvl_when = self.nidaqmx.constants.Level.LOW

            self.task_counter_ctr.in_stream.relative_to = self.nidaqmx.constants.ReadRelativeTo.FIRST_SAMPLE
            self.task_counter_ctr_ref.in_stream.relative_to = self.nidaqmx.constants.ReadRelativeTo.FIRST_SAMPLE
            # relative to beginning of buffer, change offset instead
            self.tasks_to_close += [self.task_counter_ctr, self.task_counter_ctr_ref]

        elif self.counter_mode == 'analog':
            self.close_old_tasks()

            self.task_counter_ai = self.nidaqmx.Task()
            self.task_counter_ai.ai_channels.add_ai_voltage_chan(self.analog_signal)
            self.task_counter_ai.ai_channels.add_ai_voltage_chan(self.analog_gate)
            self.task_counter_ai.ai_channels.add_ai_voltage_chan(self.analog_gate_ref)
            self.task_counter_ai.in_stream.relative_to = self.nidaqmx.constants.ReadRelativeTo.FIRST_SAMPLE
            # for analog counter
            self.tasks_to_close += [self.task_counter_ai,]


    @BaseDevice.ManagedProperty('func', thread_safe=True)
    def read_counts(self, exposure=0.1, sample_num=1000, parent=None):
        if self.counter_mode == 'apd_sample':
            if (sample_num+1)!=self.sample_num:
                self.set_counter()
                self.set_timing(exposure, sample_num)
        else:
            if exposure!=self.exposure:
                self.set_counter()
                self.set_timing(exposure, sample_num)

        if self.counter_mode == 'apd':
            total_sample = self.task_counter_ctr.in_stream.total_samp_per_chan_acquired
            sample_remain = self.sample_num
            data_main_0 = None
            data_ref_0 = None
            current_sample = total_sample
            while sample_remain>0:
                self.task_counter_ctr.in_stream.offset = current_sample
                self.task_counter_ctr_ref.in_stream.offset = current_sample
                # update read pos accrodingly to keep reading most recent self.sample_num samples
                read_sample_num = np.min([self.sample_num_single, sample_remain])
                try:
                    self.reader_ctr.read_many_sample_uint32(self.counts_main_array[:read_sample_num]
                        , number_of_samples_per_channel = read_sample_num, timeout=5*self.exposure_single
                    )
                    self.reader_ctr_ref.read_many_sample_uint32(self.counts_ref_array[:read_sample_num]
                        , number_of_samples_per_channel = read_sample_num, timeout=5*self.exposure_single
                    )
                except Exception as e:
                    log_error(e)
                    return None
                if self.is_cancel:
                    return None

                if (data_main_0 is None) and (data_ref_0 is None):
                    data_main_0 = float(self.counts_main_array[0])
                    data_ref_0 = float(self.counts_ref_array[0])
                # get the first counts when sample_num larger than buffer_size and need loop
                sample_remain -= read_sample_num
                current_sample += read_sample_num
            data_main = float(self.counts_main_array[read_sample_num-1] - data_main_0)
            data_ref = float(self.counts_ref_array[read_sample_num-1] - data_ref_0)

        if self.counter_mode == 'apd_sample':
            total_sample = self.task_counter_ctr.in_stream.total_samp_per_chan_acquired
            sample_remain = self.sample_num
            data_main_0 = None
            data_ref_0 = None
            current_sample = total_sample
            while sample_remain>0:
                self.task_counter_ctr.in_stream.offset = current_sample
                self.task_counter_ctr_ref.in_stream.offset = current_sample
                # update read pos accrodingly to keep reading most recent self.sample_num samples
                read_sample_num = np.min([self.sample_num_single, sample_remain])

                while True:
                    avail = self.task_counter_ctr.in_stream.avail_samp_per_chan
                    if avail<read_sample_num:
                        if not self.time_sleep(self.apd_sample_check_interval):
                            return None
                    else:
                        break

                try:
                    self.reader_ctr.read_many_sample_uint32(self.counts_main_array[:read_sample_num]
                        , number_of_samples_per_channel = read_sample_num, timeout=5*self.exposure_single
                    )
                    self.reader_ctr_ref.read_many_sample_uint32(self.counts_ref_array[:read_sample_num]
                        , number_of_samples_per_channel = read_sample_num, timeout=5*self.exposure_single
                    )
                except Exception as e:
                    log_error(e)
                    return None

                if self.is_cancel:
                    return None

                if (data_main_0 is None) and (data_ref_0 is None):
                    data_main_0 = float(self.counts_main_array[0])
                    data_ref_0 = float(self.counts_ref_array[0])
                # get the first counts when sample_num larger than buffer_size and need loop
                sample_remain -= read_sample_num
                current_sample += read_sample_num

            data_main = float(self.counts_main_array[read_sample_num-1] - data_main_0)
            data_ref = float(self.counts_ref_array[read_sample_num-1] - data_ref_0)

        elif self.counter_mode == 'analog':
            total_sample = self.task_counter_ai.in_stream.total_samp_per_chan_acquired
            sample_remain = self.sample_num
            data_main = 0
            data_ref = 0
            current_sample = total_sample
            while sample_remain>0:
                self.task_counter_ai.in_stream.offset = current_sample
                # update read pos accrodingly to keep reading most recent self.sample_num samples
                read_sample_num = np.min([self.sample_num_single, sample_remain])

                if read_sample_num==self.sample_num_single:
                    read_array = self.counts_array
                else:
                    read_array = self.counts_array_remain

                try:
                    self.reader_analog.read_many_sample(read_array,
                        number_of_samples_per_channel = read_sample_num, timeout=5*self.exposure_single
                    )
                except Exception as e:
                    log_error(e)
                    return None

                if self.is_cancel:
                    return None

                data = read_array[0, :read_sample_num]
                gate1 = read_array[1, :read_sample_num]
                gate2 = read_array[2, :read_sample_num]

                gate1_index = np.where(gate1 > self.analog_threshold)[0]
                gate2_index = np.where(gate2 > self.analog_threshold)[0]

                data_main += float(np.sum(data[gate1_index]))
                data_ref += float(np.sum(data[gate2_index]))

                sample_remain -= read_sample_num
                current_sample += read_sample_num


            data_main = data_main/self.sample_num
            data_ref = data_ref/self.sample_num


        if self.data_mode == 'single':
            return [data_main,]
            
        elif self.data_mode == 'ref_div':
            if data_main==0 or data_ref==0:
                return [0,]
            else:
                return [data_main/data_ref,]

        elif self.data_mode == 'ref_sub':
            return [(data_main - data_ref),]

        elif self.data_mode == 'dual':
            return [data_main, data_ref]
