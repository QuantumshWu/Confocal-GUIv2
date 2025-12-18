from .base import *
  


class USB6346(BaseCounterNI, BaseScanner):
    """
    Class for NI DAQ USB-6346
    will be used for scanner: ao0, ao1 for X and Y of Galvo
    and for counter, 
    CTR1 uses PFI3, PFI4 for src and gate, 
    CTR2 (ref) uses PFI3, PFI5 for src and gate
    exit_handler method defines how to close task when exit
    """

    def __init__(self, unique_id, port_config=None):

        if port_config is None:
            port_config = {'analog_signal':'ai0', 'analog_gate':'ai1', 'analog_gate_ref':'ai2',
            'apd_signal':'PFI3', 'apd_gate':'PFI4', 'apd_gate_ref':'PFI5', 'dev_num':'Dev1'}
        super().__init__(port_config=port_config)
        self.counter_mode_valid = ['apd', 'analog']
        self.data_mode_valid = ['single', 'ref_div', 'ref_sub', 'dual']
 

        self.task = self.nidaqmx.Task()
        self.task.ao_channels.add_ao_voltage_chan(self.dev_num + 'ao0', min_val=-5, max_val=5)
        self.task.ao_channels.add_ao_voltage_chan(self.dev_num + 'ao1', min_val=-5, max_val=5)
        self.task.start()

        self.x_lb = -5000
        self.x_ub = 5000
        self.y_lb = -5000
        self.y_ub = 5000

        self._x = 0
        self._y = 0


        
    @BaseDevice.ManagedProperty('float', monitor=True, thread_safe=True)
    def x(self):
        return self._x
    
    @x.setter
    def x(self, value):
        self._x = int(value) # in mV 
        self.task.write([self._x/1000, self._y/1000], auto_start=True) # in V
        
    @BaseDevice.ManagedProperty('float', monitor=True, thread_safe=True)
    def y(self):
        return self._y
    
    @y.setter
    def y(self, value):
        self._y = int(value) # in mV 
        self.task.write([self._x/1000, self._y/1000], auto_start=True) # in V


class USB6120(BaseCounterNIM):
    """
    Class for NI DAQ USB-6120
    """
    
    def __init__(self, unique_id, port_config=None):

        if port_config is None:
            port_config = {'analog_signal':'ai0', 'analog_gate':'ai1', 'analog_gate_ref':'ai2',
            'apd_signal':'PFI3', 'apd_gate':'PFI4', 'apd_gate_ref':'PFI5', 'dev_num':'Dev2'}
        self.counter_mode_valid = ['apd', 'analog']
        self.data_mode_valid = ['single', 'ref_div', 'ref_sub', 'dual']

class AFG31152(BaseScanner):
    """
    class for scanner AFG31152
    
    example:
    >>> afg3052c = AFG3052C()
    >>> afg3052c.x
    10
    >>> afg3052c.x = 20
    
    """
    def __init__(self, unique_id, visa_address=None):    
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

class SynthUSB3(BaseRF):
    """
    Class for SynthUSB3
    
    power in dbm
    
    frequency for frequency
    
    always on, seems no commands to set on/off
    """
    
    def __init__(self, unique_id, visa_address='ASRL12::INSTR', power_ub=-5, power_lb=None):
        import pyvisa
        rm = pyvisa.ResourceManager()
        self.handle = rm.open_resource(visa_address)
        self.power_ub = power_ub
        self.power_lb = power_lb
        self.on = True

        
    @BaseDevice.ManagedProperty('float', thread_safe=True)
    def power(self):
        self._power = eval(self.handle.query('W?')[:-1])
        return self._power
    
    @power.setter
    def power(self, value):
        self._power = value
        self.handle.write(f'W{self._power}')
    
    @BaseDevice.ManagedProperty('float', monitor=True, thread_safe=True)
    def frequency(self):
        self._frequency = eval(self.handle.query('f?')[:-1])
        return self._frequency
    
    @frequency.setter
    def frequency(self, value):
        self._frequency = value
        self.handle.write(f'f{self._frequency}')
           
    @BaseDevice.ManagedProperty('bool', thread_safe=True)
    def on(self):
        return self._on
    
    @on.setter
    def on(self, value):
        self._on = value


    def close(self):
        pass
            
class Pulse(BasePulse):
    """
    Pulse class to pulse streamer control,
    call pulse.gui() to access to all functions and pulse sequence editing

    notes:
    pulse duration, delay can be a int in ns or str contains 'x' which will be later replaced by self.x,
    therefore enabling fast pulse control by self.x = x and self.on_pulse() without reconfiguring self.data_matrix
    and sel.delay_array

    Save Pulse: save pulse sequence to self.data_matrix, self.delay_array but not in file
    Save to file: save pulse sequence to a .npz file


    """

    def __init__(self, unique_id, ip=None):
        super().__init__(t_resolution=(1, 1)) #1ns minimum width and 1ns resolution
        from pulsestreamer import PulseStreamer, Sequence 
        self.PulseStreamer = PulseStreamer
        self.Sequence = Sequence
        if ip is None:
            self.ip = '169.254.8.2'
            # default ip address of pulse streamer
        else:
            self.ip = ip
        self.ps = PulseStreamer(self.ip)

    def off_pulse_core(self):


        # Create a sequence object
        sequence = self.ps.createSequence()
        pattern_off = [(1e3, 0), (1e3, 0)]
        for channel in range(0, 8):
            sequence.setDigital(channel, pattern_off)
        # Stream the sequence and repeat it indefinitely
        n_runs = self.PulseStreamer.REPEAT_INFINITELY
        self.ps.stream(self.Sequence.repeat(sequence, 8), n_runs)
        # need to repeat 8 times because pulse streamer will pad sequence to multiple of 8ns, otherwise unexpected changes of pulse
        
    def on_pulse_core(self):
        
        self.off_pulse()
        
        def check_chs(array): 
            # return a bool(0, 1) list for channels
            # defines the truth table of channels at a given period of pulse
            return array[1:]
        
        time_slices = self.read_data()
        sequence = self.ps.createSequence()

        for channel in range(0, 8):
            time_slice = time_slices[channel]
            count = len(time_slice)
            pattern = []
            # pattern is [(duration in ns, 1 for on or 0 for off), ...]
            pattern.append((time_slice[0][0], time_slice[0][1]))
            for i in range(count-2):
                pattern.append((time_slice[i+1][0], time_slice[i+1][1]))
            pattern.append((time_slice[-1][0], time_slice[-1][1]))

            sequence.setDigital(channel, pattern)


        # Stream the sequence and repeat it indefinitely
        n_runs = self.PulseStreamer.REPEAT_INFINITELY
        self.ps.stream(self.Sequence.repeat(sequence, 8), n_runs)






    




