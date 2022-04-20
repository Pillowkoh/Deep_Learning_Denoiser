import os
import torch
import numpy as np
from torch.utils.data import Dataset


class Audio_Dataset(Dataset):
    """
    Class to load audio tensor dataset. 
    
    Args: 
        - config (dict): Dictionary that contains config settings for model 
        - set_type (Optional[str]): Choose which data to load

    """

    def __init__(self, config, set_type="train"):
        super(Audio_Dataset, self).__init__() 
        self.device = config["device"]
        self.clean_dir = os.path.join(config["data_dir"], "clean_audio_tensors_v2")
        self.noisy_dir = os.path.join(config["data_dir"], "noisy_audio_tensors_v2")

        self.clean_names = np.sort(os.listdir(self.clean_dir))
        self.noisy_names = np.sort(os.listdir(self.noisy_dir))

        self.train_end = int(config["train"] * len(os.listdir(self.clean_dir)))
        self.val_end = int((config["train"]+config["val"]) * len(os.listdir(self.clean_dir)))
        self.test_end = len(os.listdir(self.clean_dir))

        if set_type=="train":
            n_start = 0
            n_end = self.train_end

        elif set_type == "val":
            n_start = self.train_end
            n_end = self.val_end

        else:
            n_start = self.val_end
            n_end = self.test_end

        self.clean_names = self.clean_names[n_start:n_end]
        self.noisy_names = self.noisy_names[n_start:n_end]


    def __len__(self):
        return len(self.clean_names)


    def __getitem__(self, idx):
        clean_name = self.clean_names[idx]
        noisy_name = self.noisy_names[idx]

        clean_tensor_file = os.path.join(self.clean_dir, clean_name)
        noisy_tensor_file = os.path.join(self.noisy_dir, noisy_name)

        audio_input = torch.load(noisy_tensor_file, map_location=self.device)
        ground_truth = torch.load(clean_tensor_file, map_location=self.device)

        return {
            "noisy":audio_input,
            "clean":ground_truth,
        }
