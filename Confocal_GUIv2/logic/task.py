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


def test_task():

    live_plot = pl()
    live_plot = live()
    live_plot = ple(x_array = np.linspace(737.12, 737.13, 101))

