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
    BEST_WEIGHT_PATH = 'trained_weights/model_030'

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
        # self.chunks, self.padding = self._split_chunks(self.waveform, 22050)

        self.max_val = 1

        # TO DO: load model here
        self.model = Denoiser(depth=5, N_attention=1)
        self.model.load_state_dict(torch.load(weight_path, map_location=DEVICE))
    

    def denoise(self, out_fp):
        denoised_waveform = self._denoise_waveform().detach()
        torchaudio.save(out_fp, denoised_waveform, self.sr)

    def _denoise_waveform(self):
        denoised_chunks = []
        # for chunk in self.chunks:
        #     chunk = torch.unsqueeze(chunk, 0)
        #     denoised_chunk = self.model(chunk)
        #     denoised_chunk = torch.squeeze(denoised_chunk, 0)
        #     denoised_chunks.append(denoised_chunk)
        
        # denoised_waveform = torch.cat(denoised_chunks, dim=1)
        # denoised_waveform = denoised_waveform[..., :-self.padding]
        # print("DENOISED:", denoised_waveform.shape)
        print(self.waveform.dtype)
        denoised_waveform = self.model(self.waveform.unsqueeze(0)).squeeze(0)
        print(torch.max(denoised_waveform))
        
        # return torch.mul(denoised_waveform, 10)
        return denoised_waveform

    def _get_waveform(self, fn, new_sr):
        waveform, sr = torchaudio.load(fn, normalize=True)
        print("INPUT SHAPE:", waveform.shape)

        self.max_val = torch.max(waveform)
        print("MAX VAL:", self.max_val)

        if sr != new_sr:
            resampler = torchaudio.transforms.Resample(sr, new_sr)
            resampled_waveform = resampler(waveform)
        else:
            return waveform

        return resampled_waveform

    def _split_chunks(self, waveform, chunk_size=22050):
        chunks = list(torch.split(waveform, chunk_size, dim=1))
        chunks[-1], padded_extra = self._pad(chunks[-1], 22050)

        return chunks, padded_extra
    
    def _pad(self, data, size):
        assert data.size(dim=1) <= size
        if data.size(dim=1) == size:
            return data
        padded_extra = size - data.shape[1]
        padded_data  = F.pad(data, pad=(0, padded_extra))
        return padded_data, padded_extra