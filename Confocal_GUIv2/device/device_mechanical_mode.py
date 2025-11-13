from .base import *


class TLB6700(BaseLaser):
    """
    laser = TLB6700()
    
    >>> laser1.wavelength
    >>> 737.11
    >>> laser1.wavelength = 737.12
    >>> 737.12
    # return or set wavelngth
    
    >>> laser1.piezo
    >>> 0
    >>> laser1.piezo = 10
    >>> 10
    # return or set piezo voltage
        
    """
    
    def __tlb_open(self):
        self.tlb.OpenDevices(self.ProductID, True)

    def __tlb_close(self):
        self.tlb.CloseDevices()

    def __tlb_query(self, msg):
        self.answer.Clear()
        self.tlb.Query(self.DeviceKey, msg, self.answer)
        return self.answer.ToString()    
    
    def __init__(self, unique_id, piezo_lb=0, piezo_ub=100):

        import clr
        from System.Text import StringBuilder
        from System import Int32
        from System.Reflection import Assembly
        import Newport

        clr.AddReference(r'mscorlib')
        sys.path.append('C:\\Program Files\\New Focus\\New Focus Tunable Laser Application\\')
        # location of new focus laser driver file
        clr.AddReference('UsbDllWrap')

        self.tlb = Newport.USBComm.USB()
        self.answer = StringBuilder(64)

        self.ProductID = 4106
        self.DeviceKey = '6700 SN37711'
        # your own devicekey/ID, check this using newport software

        self.connect()
        
        self.piezo_lb = piezo_lb
        self.piezo_ub = piezo_ub
        
    def connect(self):
        self.__tlb_open()
        
    def close(self):
        self.__tlb_close()
        
    @BaseDevice.ManagedProperty('float', thread_safe=True)
    def wavelength(self):
        self._wavelength = float(self.__tlb_query('SOURCE:WAVELENGTH?'))
        return self._wavelength
    
    @wavelength.setter
    def wavelength(self, value):
        self._wavelength = value
        self.__tlb_query(f'SOURce:WAVElength {self._wavelength:.2f}')
        self.__tlb_query('OUTPut:TRACK 1')
        
    @BaseDevice.ManagedProperty('float', thread_safe=True)
    def piezo(self):
        self._piezo = float(self.__tlb_query('SOURce:VOLTage:PIEZO?'))
        return self._piezo
    
    @piezo.setter
    def piezo(self, value):
        self._piezo = value
        self.__tlb_query(f'SOURce:VOLTage:PIEZO {self._piezo:.2f}')



class DLCpro(BaseLaser):
    """
    laser = DLCpro()
    
    >>> laser1.wavelength
    >>> 737.11
    >>> laser1.wavelength = 737.12
    >>> 737.12
    # return or set wavelngth
    
    >>> laser1.piezo
    >>> 0
    >>> laser1.piezo = 10
    >>> 10
    # return or set piezo voltage
        
    """     
    def __init__(self, unique_id, ip=None, piezo_lb=None, piezo_ub=None):

        from toptica.lasersdk.dlcpro.v2_2_0 import DLCpro, NetworkConnection        
        if ip is None:
            ip = '128.223.23.108'
        self.ip = ip
        self.DLCpro = DLCpro
        self.NetworkConnection = NetworkConnection
        self.dlc = self.DLCpro(self.NetworkConnection(self.ip))
        self.dlc.__enter__()

        self.piezo = self.dlc.laser1.dl.pc.voltage_set.get()
        self.piezo_lb = piezo_lb
        self.piezo_ub = piezo_ub
        # piezo range where DLCpro is mode-hop free

    @property
    def wavelength(self):
        pass
    
    @wavelength.setter
    def wavelength(self, value):
        pass

        
    @BaseDevice.ManagedProperty('float', thread_safe=True)
    def piezo(self):
        self._piezo = self.dlc.laser1.dl.pc.voltage_set.get()
        return self._piezo
    
    @piezo.setter
    def piezo(self, value):
        self.dlc.laser1.dl.pc.voltage_set.set(value)
        self._piezo = value

    def close(self):
        self.dlc.__exit__()



class LaserStabilizerDLCpro(BaseLaserStabilizer):
    """
    core logic for stabilizer
    """
    
    def __init__(self, unique_id, wavemeter_handle:BaseDevice, laser_handle:BaseDevice, 
        wavelength_lb=None, wavelength_ub=None, freq_thre=0.015, freq_deadzone=0.005):
        super().__init__(unique_id=unique_id, wavemeter_handle=wavemeter_handle, laser_handle=laser_handle,
            wavelength_lb=wavelength_lb, wavelength_ub=wavelength_ub, freq_thre=freq_thre, freq_deadzone=freq_deadzone)
        self.v_mid = 0.5*(self.laser.piezo_ub + self.laser.piezo_lb)
        self.v_min = self.laser.piezo_lb + 0.01*(self.laser.piezo_ub - self.laser.piezo_lb)
        self.v_max = self.laser.piezo_lb + 0.99*(self.laser.piezo_ub - self.laser.piezo_lb)
        self.P = 1/0.56 # +1V piezo -> +0.56GHz freq, scaling factor of PID control
        self.v_step = 1 # maximum v change during single step, less than inf to prevent mode hop

        
    def _stabilizer_core(self, freq_diff):
        v_diff = np.clip(self.P*freq_diff, -self.v_step, self.v_step) # limit range of v_diff
        v_0 = self.laser.piezo
        if (v_0+v_diff)<self.v_min:
            self.laser.piezo = self.v_min
        elif (v_0+v_diff)>self.v_max:
            self.laser.piezo = self.v_max
        else:
            self.laser.piezo = v_0+v_diff
        return




class PulseSpinCore(BasePulse):
    """
    Pulse class to spincore control,
    call pulse.gui() to access to all functions and pulse sequence editing

    notes:
    pulse duration, delay can be a int in ns or str contains 'x' which will be later replaced by self.x,
    therefore enabling fast pulse control by self.x = x and self.on_pulse() without reconfiguring self.data_matrix
    and sel.delay_array

    Save Pulse: save pulse sequence to self.data_matrix, self.delay_array but not in file
    Save to file: save pulse sequence to a .npz file


    all pulse duration, delay array are round to mutiple time of 2ns


    """

    def __init__(self, unique_id):
        super().__init__(t_resolution=(12, 2)) #12ns minimum width and 2ns resolution
        import spinapi 
        self.spinapi = spinapi


    def _init(self):
        #from spinapi import pb_set_debug, pb_get_version, pb_count_boards, pb_get_error, pb_core_clock, pb_init
        self.spinapi.pb_set_debug(0)

        if self.spinapi.pb_init() != 0:
            print("Error initializing board: %s" % self.spinapi.pb_get_error())
            input("Please press a key to continue.")
            exit(-1)

        # Configure the core clock
        self.spinapi.pb_core_clock(500)


    def off_pulse_core(self):
        #from spinapi import pb_stop,pb_close
        try:
            self._init()
        except:
            pass
        self.spinapi.pb_stop()
        self.spinapi.pb_close()
        
    def on_pulse_core(self):
        #from spinapi import pb_start_programming, pb_inst_pbonly, CONTINUE, BRANCH, pb_reset, pb_start\
        #, pb_stop_programming, PULSE_PROGRAM

        ch1 = 0b000000000000000000000001
        ch2 = 0b000000000000000000000010
        ch3 = 0b000000000000000000000100
        ch4 = 0b000000000000000000001000
        ch5 = 0b000000000000000000010000
        ch6 = 0b000000000000000000100000
        ch7 = 0b000000000000000001000000
        ch8 = 0b000000000000000010000000

        channels = (ch1,
                  ch2,
                  ch3,
                  ch4,
                  ch5,
                  ch6,
                  ch7,
                  ch8)
        all_ch = 0b0
        for i in range(len(channels)):
            all_ch += channels[i]
        disable = 0b111000000000000000000000
        
        try:
            self._init()
        except:
            pass
        
        def check_chs(array):
            chs = 0b0
            for ii, i in enumerate(array[1:]):
                chs += channels[ii]*int(i)
            return chs
        
        data_matrix = self.read_data(type='data_matrix')
        count = len(data_matrix)
        # Program the pulse program

        self.spinapi.pb_start_programming(self.spinapi.PULSE_PROGRAM)


        start = self.spinapi.pb_inst_pbonly(check_chs(data_matrix[0])+disable, self.spinapi.CONTINUE, 0, data_matrix[0][0])
        for i in range(count-2):
            self.spinapi.pb_inst_pbonly(check_chs(data_matrix[i+1])+disable, self.spinapi.CONTINUE, 0, data_matrix[i+1][0])
        self.spinapi.pb_inst_pbonly(check_chs(data_matrix[-1])+disable, self.spinapi.BRANCH, start, data_matrix[-1][0])

        self.spinapi.pb_stop_programming()

        # Trigger the board
        self.spinapi.pb_reset() 
        self.spinapi.pb_start()

    def close(self):
        pass


class AFG31152(BaseScanner):
    """
    class for scanner AFG31152
    
    example:
    >>> afg3052c = AFG3052C()
    >>> afg3052c.x
    10
    >>> afg3052c.x = 20
    
    """
    def __init__(self, unique_id, visa_address='GPIB0::1::INSTR'):    
        import pyvisa   
        rm = pyvisa.ResourceManager()
        self.handle = rm.open_resource(visa_address)

        self.x_lb = -5000 #mV
        self.x_ub = 5000 #mV
        self.y_lb = -5000 #mV
        self.y_ub = 5000 #mV

    
    @BaseDevice.ManagedProperty('float', monitor=True, thread_safe=True)
    def x(self):
        result_str = self.handle.query('SOURce1:VOLTage:LEVel:IMMediate:OFFSet?')
        self._x = int(1000*eval(result_str[:-1]))
        return self._x
    
    @x.setter
    def x(self, value):
        self.handle.write(f'SOURce1:VOLTage:LEVel:IMMediate:OFFSet {value}mV')
        
    @BaseDevice.ManagedProperty('float', monitor=True, thread_safe=True)
    def y(self):
        result_str = self.handle.query('SOURce2:VOLTage:LEVel:IMMediate:OFFSet?')
        self._y = int(1000*eval(result_str[:-1]))
        return self._y
    
    @y.setter
    def y(self, value):
        self.handle.write(f'SOURce2:VOLTage:LEVel:IMMediate:OFFSet {value}mV')

    def close(self):
        pass

class WaveMeter871(BaseWavemeter):
    """
    Control code for 871 Wavelength Meter
    
    wavemeter871 = WaveMeter671()
    
    >>> wavemeter871.wavelength
    >>> 737.105033
    # read wavelength from wavemeter
    
    """
    

    def __init__(self, unique_id, ip=None):
        if ip is None:
            ip = '10.199.199.1'
        import pyvisa
        self.HOST = ip
        self.rm = pyvisa.ResourceManager('@py')
        visa_str = f'TCPIP0::{self.HOST}::23::SOCKET'
        self.handle = self.rm.open_resource(visa_str, timeout=5000, write_termination='\r\n', read_termination='\r\n')
        time.sleep(0.5)
        self.handle.clear()
        self.handle.write('*CLS')

    @BaseDevice.ManagedProperty('float', monitor=True, thread_safe=True, interval=0.01)
    def wavelength(self):
        try:
            data_str = self.handle.query(':FETC:WAV?')
            self._wavelength = float(eval(data_str))
        except Exception as e:
            pass
        finally:
            return self._wavelength
         
    def close(self):
        self.handle.close()
        self.rm.close()


class DSG836(BaseRF):
    """
    Class for RF generator DSG836
    
    power in dbm
    
    frequency for frequency
    
    on for if output is on
    """
    
    def __init__(self, unique_id, visa_address='USB0::0x1AB1::0x099C::DSG8M223900103::INSTR', power_ub=-5, power_lb=None):
        import pyvisa
        rm = pyvisa.ResourceManager()
        self.handle = rm.open_resource(visa_address)
        self.power_ub = power_ub
        self.power_lb = power_lb

        
    @BaseDevice.ManagedProperty('float', thread_safe=True)
    def power(self):
        self._power = eval(self.handle.query('SOURce:Power?')[:-1])
        return self._power
    
    @power.setter
    def power(self, value):
        self._power = value
        self.handle.write(f'SOURce:Power {self._power}')
    
    @BaseDevice.ManagedProperty('float', monitor=True, thread_safe=True)
    def frequency(self):
        self._frequency = eval(self.handle.query('SOURce:Frequency?')[:-1])
        return self._frequency
    
    @frequency.setter
    def frequency(self, value):
        self._frequency = value
        self.handle.write(f'SOURce:Frequency {self._frequency}')
           
    @BaseDevice.ManagedProperty('bool', thread_safe=True)
    def on(self):
        self._on = True if (eval(self.handle.query('OUTPut:STATe?')[:-1]) == True) else False
        # will return 0, 1
        return self._on
    
    @on.setter
    def on(self, value):
        self._on = value
        if self._on is True:
            self.handle.write('OUTPut:STATe ON')
        else:
            self.handle.write('OUTPut:STATe OFF')


    def close(self):
        pass


class USB2120(BaseCounterNI):
    """
    Class for NI DAQ USB-2120
    """
    
    def __init__(self, unique_id, port_config=None):

        if port_config is None:
            port_config = {'apd_signal':'PFI3', 'apd_gate':'PFI4', 'apd_gate_ref':'PFI1', 'apd_clock':'PFI12', 'dev_num':'Dev2'}
        super().__init__(port_config=port_config)
        self.counter_mode_valid = ['apd', 'analog', 'apd_sample']
        self.data_mode_valid = ['single', 'ref_div', 'ref_sub', 'dual']


class Camera(BaseCounter):
    def __init__(self, unique_id):
        from IPython.display import display
        import PIL.Image
        import cv2
        self.cv2 = cv2
        self.display = display
        self.PILImage = PIL.Image
        self.cap = self.cv2.VideoCapture(0, self.cv2.CAP_DSHOW)
        self.x_l = 0
        self.x_u = -1
        self.y_l = 0
        self.y_u = -1
        self.overhead = 0.04 # 25FPS

        self.exposure = None
        self.counter_mode_valid = ['apd']
        self.data_mode_valid = ['single']
        self.data_mode = 'single'
        self.counter_mode = 'apd'

    def close(self):
        self.cap.release()
        self.cv2.destroyAllWindows()

    def read_frame(self, counts=1):
        frame_ = None
        for _ in range(counts):
            ret, frame = self.cap.read()
            gray = self.cv2.cvtColor(frame, self.cv2.COLOR_BGR2GRAY)
            frame3 = self.cv2.merge([gray, gray, gray])
            frame_large = frame3.astype(np.float32)
            if frame_ is None:
                frame_ = frame_large
            else:
                frame_ += frame_large

        return (frame_/counts).astype(np.uint8)[self.x_l:self.x_u, self.y_l:self.y_u]

    def monitor(self):
        # display image until keyboard interrupt
        try:
            self.handle = None
            while True:
                self.frame_ = self.read_frame(counts=5)
                image = self.PILImage.fromarray(self.frame_)
                if self.handle is None:
                    self.handle = self.display(image, display_id = True)
                else:
                    self.handle.update(image)
        except KeyboardInterrupt:
            return
        except Exception as e:
            log_error(e)
            return

    def image_sum(self, counts=5):
        return np.sum(self.read_frame(counts=counts))

    @property
    def counter_mode_valid(self):
        return self._counter_mode_valid

    @counter_mode_valid.setter
    def counter_mode_valid(self, value):
        self._counter_mode_valid = value

    @property
    def data_mode_valid(self):
        return self._data_mode_valid

    @data_mode_valid.setter
    def data_mode_valid(self, value):
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
        self._counter_mode = value

    @BaseDevice.ManagedProperty('func', thread_safe=True)
    def read_counts(self, exposure=0.1, sample_num=1000, parent=None):
        _ = int(self.image_sum(counts=1))
        # skip the incomplete first frame
        self.exposure = exposure
        counts = int(self.image_sum(counts=int(np.ceil(self.exposure/self.overhead))))
        return [counts,]

    @property
    def data_len(self):
        return 1