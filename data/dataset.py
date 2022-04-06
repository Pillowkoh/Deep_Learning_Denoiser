import os
import math
import random
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import librosa
import torch
import random
import warnings
from IPython.display import Audio
from tqdm import tqdm
from scipy.io.wavfile import write

# suppress 'UserWarning: PySoundFile failed. Trying audioread instead.' warning from Librosa when processing .mp3 files
warnings.filterwarnings('ignore')

class NoisyAudioDataset:
    def __init__(self, audio_paths, noise_paths, size=10000, sr=22050):
        self.audio_paths = audio_paths[:10000]
        self.noise_paths = noise_paths
        self.size = size
        self.sr = sr

    # helper method to get a random segment of the audio
    def _get_random_segment(self, signal):
        random_idx = np.random.randint(0, signal.size)
        random_position = np.random.random() > 0.5
        return signal[random_idx:] if random_position else signal[:random_idx]

    # adds single random noise to audio
    def _add_single_noise_to_audio(self, audio_signal, noise_signal):
        if len(audio_signal) >= len(noise_signal):
            while len(audio_signal) >= len(noise_signal):
                rand = np.random.random() < 0.75
                extra = noise_signal if rand else self._get_random_segment(noise_signal)
                noise_signal = np.append(noise_signal, extra)
        
        idx = np.random.randint(0, noise_signal.size - audio_signal.size)
        noise_segment = noise_signal[idx: idx + audio_signal.size]

        speech_power = np.sum(audio_signal ** 2)
        noise_power = np.sum(noise_segment ** 2)
        random_noise_scaler = np.random.uniform(0.2, 1)
        noisy_audio = audio_signal + random_noise_scaler * np.sqrt(speech_power / noise_power) * noise_segment

        return noisy_audio

    # add multiple noise to audio
    def _add_multiple_noise_to_audio(self, audio_signal, *noise_gen):
        noise = list(noise_gen)
        min_noise_length = min([len(n) for n in noise])
        ratio = 1 / len(noise)
        
        noise_signal = np.zeros(min_noise_length, dtype="float32")
        for n in noise:
            noise_signal = np.add(noise_signal, ratio * n[:min_noise_length])

        return self._add_single_noise_to_audio(audio_signal, noise_signal)

    def _read_audio(self, filepath):
        audio, _ = librosa.load(filepath, sr=self.sr)
        return audio

    def _write_wav_file(self, filepath, audio):
        write(filepath, 22050, audio)

    # creates a directory of noisy audio files. Also generates a .csv file matching the filename of each noisy audio clip to its target
    def create_noisy_dataset(self):
        # open csv file to link oisy audio filename and clean filename
        f = open('clean_noisy_fn.csv', 'w')
        writer = csv.writer(f)
        writer.writerow(['clean', 'noisy'])

        os.mkdir('noisy_audio')

        for i in tqdm(range(self.size)):
            # fetch both clean audio file and noise file
            audio_path = self.audio_paths[i]
            audio_fn = os.path.basename(audio_path)
            audio_signal = self._read_audio(audio_path)
            rand_idx = np.random.randint(0, len(self.noise_paths))
            noise_signal = self._read_audio(self.noise_paths[rand_idx])


            # randomly decide if writing single or multiple noise audio to clean audio, and produce the noisy audio signal
            rand_is_multiple = np.random.random() < 0.25
            if rand_is_multiple:
                rand_idx2 = np.random.randint(0, len(self.noise_paths))
                noise_signal2 = self._read_audio(self.noise_paths[rand_idx2])
                noisy_audio = self._add_multiple_noise_to_audio(audio_signal, noise_signal, noise_signal2)
            else:
                noisy_audio = self._add_multiple_noise_to_audio(audio_signal, noise_signal)

            # write new file to noisy_audio directory
            folder_path = 'noisy_audio'
            noisy_fn = 'noisy_' + str(i) + '.wav'
            noisy_fp = os.path.join(folder_path, noisy_fn)
            self._write_wav_file(noisy_fp, noisy_audio)

            # write new entry to csv
            writer.writerow([audio_fn, noisy_fn])
        f.close()

# get file paths for both noise audio and clean data (USE THIS OVER PREVIOUS)
urban_sounds_path = os.path.join('Data', 'UrbanSound8k', 'audio')
urban_sounds_filenames = []
for i in range(1, 11):
    fold_path = 'fold' + str(i)
    filenames = os.listdir(os.path.join(urban_sounds_path, fold_path))
    sound_filesnames = filter(lambda fn : fn.endswith('.wav'), filenames)
    sound_filesnames = map(lambda fn : os.path.join(urban_sounds_path, fold_path, fn), sound_filesnames)
    urban_sounds_filenames.extend(sound_filesnames)

common_voice_path = os.path.join('Data', 'common-voice', 'en/clips')
common_voice_filenames = os.listdir(common_voice_path)
common_voice_filenames = filter(lambda fn : fn.endswith('.mp3'), common_voice_filenames)
common_voice_filenames = list(map(lambda fn: os.path.join(common_voice_path, fn), common_voice_filenames))

dataset = NoisyAudioDataset(common_voice_filenames, urban_sounds_filenames, size=10000)
dataset.create_noisy_dataset()
