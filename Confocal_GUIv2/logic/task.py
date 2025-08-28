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
    power=-20, ref_freq=500, h10_ratio=1.5, beta=0.05, exposure_h1=1,
    wavelength=737.11935, ple_span=0.003, ple_step=0.0001, ple_exposure=0.5,
    center=[0,0], pl_span=10, pl_step=2, pl_exposure=0.5,
    addr='mode_search/'):
    live_plot = ple(x_array = np.arange(wavelength-ple_span, wavelength+ple_span+ple_step, ple_step), exposure=ple_exposure)
    _, p0 = live_plot.data_figure.lorent()
    live_plot.data_figure.save(addr)
    wavelength = p0[0]
    laser_stabilizer = get_devices('laser_stabilizer')
    laser_stabilizer.wait_to_wavelength(wavelength)
    live_plot = pl(x_array = None)
    laser_stabilizer.on = False
    pass
    # to be continued



def test_task():

    live_plot = pl()
    live_plot = live()
    live_plot = ple(x_array = np.linspace(737.12, 737.13, 101))

