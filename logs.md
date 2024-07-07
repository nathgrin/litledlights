# Logs

## Initial wiring

### 2023-01-29

- reinstall bunch of python and install adafruit-circuitpython following <https://www.tomshardware.com/how-to/use-circuitpython-raspberry-pi>
- follow <https://learn.adafruit.com/neopixels-on-raspberry-pi/python-usage#> and .. works

### 2023-12-31

- try to install venv...
- failed, is not installing packages, is not using venv, git hates .pyc files that come out of nowhere and doesnt want to sync anymore

### 2024-01-03

- port to rpi-4B
- try venv now


### 2024-07-07

- install venv
- circuitpython <https://learn.adafruit.com/circuitpython-on-raspberrypi-linux/installing-circuitpython-on-raspberry-pi> kills the GUI
- reinstall
- circuitpython <https://learn.adafruit.com/circuitpython-on-raspberrypi-linux/installing-circuitpython-on-raspberry-pi>
- adafruit https://learn.adafruit.com/neopixels-on-raspberry-pi/python-usage and things worked but not yet the lights
  - in the venv with (.venv) $ sudo -E env PATH=$PATH ./myscript.py
  - apparently sudo PYTHON3_VENV_PATH=$(which python3) $PYTHON3_VENV_PATH path/to/your/script.py is a good thing too