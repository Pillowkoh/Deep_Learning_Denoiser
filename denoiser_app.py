from locale import normalize
import torchaudio
import torch
import torch.nn.functional as F
import os
from scipy.io import wavfile
from model import Denoiser

if os.name == 'nt':
    BEST_WEIGHT_PATH = 'model_final'

elif os.name == 'posix':
    BEST_WEIGHT_PATH = 'model_final'

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class AudioDenoiser:
    """
        Wrapper class for the denoiser model. Processes for a single input audio file and returns a denoised audio file using the denoise() function
        
        Args:
            - filename (str): input filepath of audio file to be denoised
            - sr (int): sample rate of denoied file
            - weight_path (str): path of weights to be used for denoiser model
    """
    def __init__(self, filename, sr=48_000, weight_path = BEST_WEIGHT_PATH):
        self.fn = filename
        self.sr = sr
        self.waveform = self._get_waveform(filename, sr)
        self.model = Denoiser(depth=5, N_attention=1)
        self.model.load_state_dict(torch.load(weight_path, map_location=DEVICE))
    

    def denoise(self, out_fp):
        denoised_waveform = self._denoise_waveform().detach()
        torchaudio.save(out_fp, denoised_waveform, self.sr)


    def _denoise_waveform(self):
        denoised_waveform = self.model(self.waveform.unsqueeze(0)).squeeze(0)
        return denoised_waveform


    def _get_waveform(self, fn, new_sr):
        waveform, sr = torchaudio.load(fn, normalize=True)

        if sr != new_sr:
            resampler = torchaudio.transforms.Resample(sr, new_sr)
            resampled_waveform = resampler(waveform)
        else:
            return waveform

        return resampled_waveform

    ### Deprecated. Do not use. ###

    # def _split_chunks(self, waveform, chunk_size=22050):
    #     chunks = list(torch.split(waveform, chunk_size, dim=1))
    #     chunks[-1], padded_extra = self._pad(chunks[-1], 22050)

    #     return chunks, padded_extra
    
    # def _pad(self, data, size):
    #     assert data.size(dim=1) <= size
    #     if data.size(dim=1) == size:
    #         return data
    #     padded_extra = size - data.shape[1]
    #     padded_data  = F.pad(data, pad=(0, padded_extra))
    #     return padded_data, padded_extra