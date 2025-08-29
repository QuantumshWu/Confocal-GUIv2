from Confocal_GUIv2.device import get_devices
from .measurement import *

def find_bright_sivs_task(wavelength_array, x_array, y_array, bound_pts, exposure, addr):

    laser_stabilizer = get_devices('laser_stabilizer')
    live_plot = None
    for wavelength in wavelength_array:
        laser_stabilizer.wait_to_wavelength(wavelength)
        live_plot = pl(x_array=x_array, y_array=y_array, bound_pts=bound_pts, fig=live_plot.fig if live_plot else None, 
        	exposure=exposure, auto_save_and_close=False)

        live_plot.data_figure.save(f'{addr}wavelength_at_{wavelength}')
        # in case a keyboardinterrupt/or any exceptions
        if live_plot.controller.is_interrupt:
            break

    laser_stabilizer.on = False


def mode_search_task(freq_array=np.arange(976.95-1, 976.95+1e-4, 1e-4), 
    exposure=0.01, sample_num=1000, counter_mode='apd_sample', 
    power=-20, ref_freq=500, h10_ratio=1.5, beta=0.05, exposure_h1=5,
    wavelength=737.11935, ple_span=0.003, ple_step=0.0001, ple_exposure=0.5,
    center=[0,0], pl_span=10, pl_step=2, pl_exposure=0.5,
    addr='mode_search/', pts_save=3000):
    n = np.ceil(len(freq_array)/pts_save)
    for i in range(n):
        live_plot = ple(x_array = np.arange(wavelength-ple_span, wavelength+ple_span+ple_step, ple_step), 
            exposure=ple_exposure)
        _, p0 = live_plot.data_figure.lorent()
        live_plot.data_figure.save(addr)
        wavelength = p0[0]
        laser_stabilizer = get_devices('laser_stabilizer')
        laser_stabilizer.wait_to_wavelength(wavelength)
        live_plot = pl(x_array = np.arange(center[0]-pl_span, center[0]+ple_span+pl_step, pl_step),
                        y_array = np.arange(center[1]-pl_span, center[1]+ple_span+pl_step, pl_step),
                        exposure=pl_exposure)
        laser_stabilizer.on = False
        _, p0 = live_plot.data_figure.center()
        live_plot.data_figure.save(addr)
        x = p0[-2]
        y = p0[-1]
        center = [x, y]
        if i==(n-1):
            x_array = freq_array[i*pts_save:]
        else:
            x_array = freq_array[i*pts_save:(i+1)*pts_save]
        live_plot = mode(x_array=x_array, exposure=exposure, sample_num=sample_num, counter_mode=counter_mode,
            power=power, ref_freq=ref_freq, h10_ratio=h10_ratio, beta=beta, exposure_h1=exposure_h1,
            is_adaptive=True)
        live_plot.data_figure.save(addr)




def test_task():

    live_plot = pl()
    live_plot = live()
    live_plot = ple(x_array = np.linspace(737.12, 737.13, 101))

