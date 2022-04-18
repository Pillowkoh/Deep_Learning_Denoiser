import os
import torch
import numpy as np
from torch.utils.data import Dataset


class Audio_Dataset(Dataset):
    def __init__(self, config, set_type="train"):
        super(Audio_Dataset, self).__init__() 
        self.clean_dir = os.path.join(config["clean_dir"], "clean_audio_WAV")
        self.noisy_dir = os.path.join(config["noisy_dir"], "noisy_audio_WAV")