import os
import torch
import numpy as np
from torch.utils.data import Dataset


class Audio_Dataset(Dataset):
    def __init__(self, config, set_type="train"):
        super(Audio_Dataset, self).__init__() 
        self.device = config["device"]
        self.clean_dir = os.path.join(config["clean_dir"], "clean_audio_tensor")
        self.noisy_dir = os.path.join(config["noisy_dir"], "noisy_audio_tensor")

        self.clean_names = np.sort(os.listdir(self.clean_dir))
        self.noisy_names = np.sort(os.listdir(self.noisy_dir))

        if set_type=="train":
            n_start = 0
            n_end = 7000

        elif set_type == "val":
            n_start = 7000
            n_end = 9000

        else:
            n_start = 9000
            n_end = 10000

        self.clean_names = self.clean_names[n_start:n_end]
        self.noisy_names = self.noisy_names[n_start:n_end]


    def __len__(self):
        return len(self.clean_names)


    def __getitem__(self, idx):
        clean_name = self.clean_names[idx]
        noisy_name = self.noisy_names[idx]

        audio_input = torch.load(noisy_name, map_location=self.device)
        ground_truth = torch.load(clean_name, map_location=self.device)

        return {
            "noisy":audio_input,
            "clean":ground_truth,
        }

