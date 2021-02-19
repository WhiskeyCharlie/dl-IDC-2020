# Deep Learning Final Project - Webcam Background Removal

## System Requirements

* Any OS to train model (tested on Ubuntu 20.04 only), Linux (only tested on Ubuntu 20.04)
  for application (must install v4l2loopback-utils).
* Some webcam

## Installation

* Create a virtual environment with Python3.+ and activate it (tested with Python 3.8).
* Run `pip install -r requirements.txt`
* Install `v4l2loopback-utils`, on a debian system you can run `apt-get install v4l2loopback-utils`
* Run `modprobe v4l2loopback devices=2 ` (may require root privileges).

## Usage

### Training a model

* _Note_: If you would prefer not to train, a complete model is provided under `saved_models/`
* Run `training.py` with a python interpreter as configured in the installation stage.
* A progress bar will display the estimated time until completion of training.
    * As configured, training time should take no more than 30 minutes on a reasonably modern system.
    * Parameters are available towards the top of the `training.py` file. It may be worth investigating them, some
      functionality (such as resizing images) is available there.
* A trained model will be stored in the `saved_models/`.

### Running the Application

* Again, a Linux system with v4l2loopback-utils is required.
* Copy the name of a trained model stored in `saved_models/` and place it as the string in the
  `MODEL_PATH` variable in `application.py`. As it is, an example model is provided and configured to run.
* Assuming you have followed the installation steps correctly, and have exactly one webcam plugged in, you should now be
  able to run `application.py` without arguments.
* This will populate the feed of the newly created virtual webcam with background-removed images from your actual
  webcam. If you run into problems, you may need to modify the `FAKE_WEBCAM_PATH` variable in application.py to reflect
  the newly added virtual webcam's path.

### Additional Notes

* The background image can be changed by modifying the variable `BACKGROUND_IMAGE` in `application.py`
  to point to some other background image. One is provided by default in `backgrounds/`


