# Deep_Learning_Denoiser
50.039 Theory and Practice of Deep Learning Project

Implementation of a denoiser that takes in a noisy audio file and returns a .wav file of the denoised audio. Noise refers to background urban noises such as idling engines or air conditioner humming. The model follows a encoder-decoder architecture with 1D convolutional layers, along with multi-headed attention layers as the bottleneck module between encoder and decoder.

## Preparing Datset
To load tensors of waveforms from the clean and noisy audio .wav files, in notebooks/Audio_conversion.ipynb, run the notebook cells from header 'Converting WAV to Tensors' onwards.

## Training the model
Run Train.ipynb

## Directory structure
* **Utils**: utility files, including helper functions and dataset.py, which implements a helper class that generates noisy audio files by adding noise samples to clean audio files
* **notebooks**: jupyter notebooks for various purposes
* **static**: styling for GUI
* **templates**: HTML which defines structure for GUI
* **trained_weights**: saved weights which can be loaded to a denoiser model
* app.py: Flask server
* denoiser_app.py: Wrapper class for denoiser model which interfaces with the Flask server
* Train.ipynb: jupyter notebook for training of denoiser model
* audio_dataset.py: dataloader for training
* loss.py: loss function for training
* model.py: denoiser model architecture class
* trainer.py: trainer class for denoiser model
