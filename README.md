# Confocal_GUI

A Python package for realizing GUI and live plot in Jupyter notebook (or jupyterlab) for confocal related experiment. 

## Basic Usage
Refer to 'jupyter notebook examples/examples with virtual devices.ipynb' for a complete guide on basic uses

### GUI for setting up pulses
![pulse_GUIv2](https://github.com/user-attachments/assets/535f3279-ea84-4468-8c93-2a475e09d572)

Can be configure in src\Confocal_GUI\device\base.py and src\Confocal_GUI\device\device.py to connect to Pulsestreamer or SpinCore

### GUI for setting up devices
![device_guiv2](https://github.com/user-attachments/assets/93a9e1aa-083f-4fc4-a593-9a5b85c81d83)

Can be configure in src\Confocal_GUI\device\device.py for all accessible parameters from GUI


### Inline live plot for 1D measurement, PLE for exmaple
![plev2](https://github.com/user-attachments/assets/ccbe72f4-721b-4fab-9c88-f2e62f814d70)

Similar live plot is also realized for 2D measurements, refer to src\Confocal_GUI\live_plot\live_plot.py


### GUI for combined measurments
![GUI](https://github.com/user-attachments/assets/2c5e8856-7fe2-4b36-a1a6-7f3f09ea9bb6)

Read the help(GUI), can be connected to any 1D/2D measurements when call GUI()


## Configure Your Own Experiment
Refer to 'jupyter notebook examples/examples for UO optics lab.ipynb'

Add your device classes to src\Confocal_GUI\device\

Add your logic for measurement to src\Confocal_GUI\logic\





