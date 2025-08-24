from .base import *

class VirtualLaser(BaseLaser):
    """
    VirtualLaser class to simulate laser 
        
    """
      
    
    def __init__(self, unique_id, piezo_lb, piezo_ub):
        self.wavelength = 0
        self.piezo = 0
        self.piezo_lb = piezo_lb
        self.piezo_ub = piezo_ub

    @BaseDevice.ManagedProperty('float', thread_safe=True)
    def wavelength(self):
        return self._wavelength
    
    @wavelength.setter
    def wavelength(self, value):
        self._wavelength = value

    @BaseDevice.ManagedProperty('float', thread_safe=True)
    def piezo(self):
        return self._piezo
    
    @piezo.setter
    def piezo(self, value):
        self._piezo = value

    def close(self):
        pass



class VirtualRF(BaseRF):
    """
    VirtualRF class to simulate rf source,
    call rf.gui() to see all configurable parameters 

    """

    def __init__(self, unique_id, frequency_lb=0.1e9, frequency_ub=3.5e9):
        self.frequency = 0
        self.power = 0
        self.on = False
        self.frequency_lb = frequency_lb
        self.frequency_ub = frequency_ub
        self.power_lb = -50 #dbm
        self.power_ub = 20 #dbm


    @BaseDevice.ManagedProperty(gui_type='float', monitor=True, interval=0.2, thread_safe=True)
    def frequency(self):
        time.sleep(0.05)
        return self._frequency
    
    @frequency.setter
    def frequency(self, value):
        self._frequency = value

    @BaseDevice.ManagedProperty(gui_type='float', thread_safe=True)
    def power(self):
        return self._power
    
    @power.setter
    def power(self, value):
        self._power = value

    @BaseDevice.ManagedProperty(gui_type='bool', thread_safe=True)
    def on(self):
        return self._on
    
    @on.setter
    def on(self, value):
        self._on = value

    def close(self):
        pass


    
        
class VirtualWavemeter(BaseWavemeter):
    """
    VirtualWavemeter class to simulate wavemeter
    
    """
    

    def __init__(self, unique_id):
        self.wavelength = 1


    @BaseDevice.ManagedProperty('float', monitor=True, thread_safe=True, interval=0.01)
    def wavelength(self):
        return self._wavelength


    @wavelength.setter
    def wavelength(self, value):
        self._wavelength = value

    def close(self):
        pass

class VirtualLaserStabilizer(BaseLaserStabilizer):
    """
    VirtualLaserStabilizer class to simulate laserstabilizer which stabilize laser wavelength using feedback,
    call laserstabilizer.gui() to see all configurable parameters
    
    """

    def _stabilizer_core(self, freq_diff):
        # defines the core logic of feedback stabilization
        self.wavemeter.wavelength = self.desired_wavelength


class VirtualScanner(BaseScanner):
    """
    VirtualScanner class to scanner,
    call scanner.gui() to see all configurable parameters
    
    """
    def __init__(self, unique_id):
        self.x = 0
        self.y = 0

        self.x_lb = -5000 #mV
        self.x_ub = 5000 #mV
        self.y_lb = -5000 #mV
        self.y_ub = 5000 #mV

    
    @BaseDevice.ManagedProperty('float', monitor=True, thread_safe=True)
    def x(self):
        return self._x
    
    @x.setter
    def x(self, value):
        self._x = int(round(value))
        
    @BaseDevice.ManagedProperty('float', monitor=True, thread_safe=True)
    def y(self):
        return self._y
    
    @y.setter
    def y(self, value):
        self._y = int(round(value))

    def close(self):
        pass


class VirtualCombo(BaseScanner, VirtualRF):
    """
    VirtualScanner class to scanner,
    call scanner.gui() to see all configurable parameters
    
    """
    def __init__(self, unique_id):
        super().__init__(unique_id = unique_id)
        self.x = 0
        self.y = 0

        self.x_lb = -5000 #mV
        self.x_ub = 5000 #mV
        self.y_lb = -5000 #mV
        self.y_ub = 5000 #mV

    
    @BaseDevice.ManagedProperty('float', monitor=True, thread_safe=True)
    def x(self):
        return self._x
    
    @x.setter
    def x(self, value):
        self._x = int(round(value))
        
    @BaseDevice.ManagedProperty('float', monitor=True, thread_safe=True)
    def y(self):
        return self._y
    
    @y.setter
    def y(self, value):
        self._y = int(round(value))

    def close(self):
        pass

class VirtualPulse(BasePulse):
    """
    VirtualPulse class to simulate pulse control (e.g. pulse streamer),
    call pulse.gui() to access to all functions and pulse sequence editing

    notes:
    pulse duration, delay can be a int in ns or str contains 'x' which will be later replaced by self.x,
    therefore enabling fast pulse control by self.x = x and self.on_pulse() without reconfiguring self.data_matrix
    and sel.delay_array

    Save Pulse: save pulse sequence to self.data_matrix, self.delay_array but not in file
    Save to file: save pulse sequence to a .npz file


    """

    def __init__(self, unique_id):
        super().__init__(t_resolution=(2,2))


    def off_pulse_core(self):
        # rewrite this method for real pulse
        pass
        
    def on_pulse_core(self):
        # rewrite this method for real pulse
        time_slices = self.read_data(type='time_slices')
        for ii, time_slice in enumerate(time_slices):
            print(f'Ch{ii}', time_slice)
        return time_slices

    def close(self):
        pass



class VirtualCounter(BaseCounter):
    """
    VirtualCounter class to simulate counter,
    defines how counts are changing depends on rf.frequency, laser.wavelength, etc.
    
    """

    def __init__(self, unique_id):
        self.counter_mode = 'apd'
        self.data_mode = 'single'


    @property
    def counter_mode_valid(self):
        return ['analog', 'apd', 'apd_sample']

    @property
    def data_mode_valid(self):
        return ['single', 'ref_div', 'ref_sub', 'dual']

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

    @property
    def data_len(self):
        return 2 if self.data_mode=='dual' else 1
    # the return data type 


    @BaseDevice.ManagedProperty('func', thread_safe=True)
    def read_counts(self, exposure=None, sample_num=None, parent=None):
        """
        simulated counter for test
        """

        if parent is None:
            self.time_sleep(exposure)
            return [np.random.poisson(exposure*10000),]

        if self.counter_mode == 'apd_sample':
            if sample_num is None:
                return None
            total_duration = get_devices('pulse').total_duration
            if total_duration is None:
                exposure = 10
            else:
                exposure = sample_num*total_duration/1e9
        
        _class = parent.__class__.__name__
        if _class == 'PLEMeasurement':
            return self._PLEMeasurement(exposure)
        elif _class == 'ODMRMeasurement':
            return self._ODMRMeasurement(exposure)
        elif _class == 'PLMeasurement':
            return self._PLMeasurement(exposure)
        else:
            self.time_sleep(exposure)
            return [np.random.poisson(exposure*10000),]

    def _PLEMeasurement(self, exposure):
        ple_height = 3000
        ple_width = 0.0004
        ple_center = 737.1
        ple_bg = 500

        self.time_sleep(exposure)
        wavelength = get_devices('wavemeter').wavelength
        lambda_counts = exposure*(ple_height*(ple_width/2)**2
                                 /((wavelength-ple_center)**2 + (ple_width/2)**2) + ple_bg
        )
        lambda_ref = exposure*ple_bg
        ref = np.random.poisson(lambda_ref)

        if self.data_mode=='dual':
            return [np.random.poisson(lambda_counts), ref]
        elif self.data_mode == 'ref_sub':
            return [np.random.poisson(lambda_counts)-ref,]
        elif self.data_mode == 'ref_div':
            return [np.random.poisson(lambda_counts)/ref if ref!=0 else 0,]
        elif self.data_mode == 'single':
            return [np.random.poisson(lambda_counts),]

    def _ODMRMeasurement(self, exposure):
        odmr_bg = 3000
        bg = 500
        odmr_width = 1e6 # intrinsic half width
        odmr_center = 2.87e9
        odmr_power = 3 # then Omega is gamma*10**((Power-3)/10)

        self.time_sleep(exposure)
        frequency = get_devices('rf').frequency
        power = get_devices('rf').power
        omega = odmr_width*10**((power-odmr_power)/10)
        cr_bg = 3000 + bg
        delta = frequency - odmr_center
        width = np.sqrt(odmr_width**2 + 2*omega**2)
        cr_dip = 3000*(1-(2*omega**2/width**2)*1/(1+(delta/width)**2)) + bg
        lambda_counts = exposure*cr_dip
        lambda_ref = exposure*cr_bg
        ref = np.random.poisson(lambda_ref)

        if self.data_mode=='dual':
            return [np.random.poisson(lambda_counts), ref]
        elif self.data_mode == 'ref_sub':
            return [np.random.poisson(lambda_counts)-ref,]
        elif self.data_mode == 'ref_div':
            return [np.random.poisson(lambda_counts)/ref if ref!=0 else 0,]
        elif self.data_mode == 'single':
            return [np.random.poisson(lambda_counts),]

    def _PLMeasurement(self, exposure):
        pl_bg = 300
        pl_height = 5000
        pl_width = 2
        pl_center = (10, 3)

        self.time_sleep(exposure)
        x = get_devices('scanner').x
        y = get_devices('scanner').y
        dis = np.sqrt((x-pl_center[0])**2 + (y-pl_center[1])**2)
        pl_counts = pl_height*np.exp(-dis**2/(2*pl_width**2)) + pl_bg
        lambda_counts = exposure*pl_counts
        lambda_ref = exposure*pl_bg
        ref = np.random.poisson(lambda_ref)

        if self.data_mode=='dual':
            return [np.random.poisson(lambda_counts), ref]
        elif self.data_mode == 'ref_sub':
            return [np.random.poisson(lambda_counts)-ref,]
        elif self.data_mode == 'ref_div':
            return [np.random.poisson(lambda_counts)/ref if ref!=0 else 0,]
        elif self.data_mode == 'single':
            return [np.random.poisson(lambda_counts),]

    def close(self):
        pass

class VirtualCounter2(BaseCounter):
    """
    VirtualCounter class to simulate counter,
    defines how counts are changing depends on rf.frequency, laser.wavelength, etc.
    
    """

    def __init__(self, unique_id):
        self.counter_mode = 'apd'
        self.data_mode = 'single'


    @property
    def counter_mode_valid(self):
        return ['analog', 'apd']

    @property
    def data_mode_valid(self):
        return ['single', 'ref_div', 'ref_sub', 'dual']

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

    @property
    def data_len(self):
        return 2 if self.data_mode=='dual' else 1
    # the return data type 


    @BaseDevice.ManagedProperty('func')
    def read_counts(self, exposure, parent=None):
        """
        simulated counter for test
        """
        self.time_sleep(exposure)
        return [0,]

    def close(self):
        pass
        
        