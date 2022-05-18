# Deep_Learning_Denoiser
50.039 Theory and Practice of Deep Learning Project

Implementation of a denoiser that takes in a noisy audio file and returns a .wav file of the denoised audio. Noise refers to background urban noises such as idling engines or air conditioner humming. The model follows a encoder-decoder architecture with 1D convolutional layers, along with multi-headed attention layers as the bottleneck module between encoder and decoder.

## Preparing Datset
Read `./data/INSTRUCTIONS.txt` for steps to download datasets

To load tensors of waveforms from the clean and noisy audio .wav files, in notebooks/Audio_conversion.ipynb, run the notebook cells from header 'Converting WAV to Tensors' onwards.

## Training the model
Run all cells in Train.ipynb.

## Directory structure
* **Utils**:  
utility files, including helper functions and dataset.py, which implements a helper class that generates noisy audio files by adding noise samples to clean audio files

* **notebooks**:  
jupyter notebooks for various purposes

* **static**:  
styling for GUI

* **templates**:  
HTML which defines structure for GUI

* **trained_weights**:  
saved weights which can be loaded to a denoiser model

* **data (NOT IN REPOSITORY)**:  
folder to hold all relevant data for training of denoiser model

* app.py:  
Flask server

* denoiser_app.py:  
Wrapper class for denoiser model which interfaces with the Flask server

* Train.ipynb:  
jupyter notebook for training of denoiser model

* audio_dataset.py:  
dataloader for training

* loss.py:  
loss function for training

* model.py:  
denoiser model architecture class

* trainer.py:  
trainer class for denoiser model

## Running the application
It is recommended that you carry out these steps in Visual Studio Code. In the command line terminal, ensure that you are in the Deep_Learning_Denoiser folder. Then, execute the following steps:

1. Create a virtual environment
* POSIX: `python3 -m venv venv`
* Windows: `py -3 -m venv venv`

2. Activate the virtual environment
* POSIX: `. venv/bin/activate`
* Windows: `venv\Scripts\activate`

3. Install dependencies in the virtual environment  
`pip install -r requirements.txt`

4. Set the Flask app to be app.py  
`set FLASK_APP=app.py`
  
5. Run the app  
`python -m flask run`

6. Follow the link generated in your terminal (to your localhost) to load the application

Following the instructions on the web application should allow you to upload an audio file to denoise and subsequently download the denoised file.
