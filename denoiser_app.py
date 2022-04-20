import torchaudio
import torch
import torch.nn.functional as F
import os
from scipy.io import wavfile
from model import Denoiser

if os.name == 'nt':
    BEST_WEIGHT_PATH = '.\\trained_weights\\model_030.pt'

elif os.name == 'posix':
    BEST_WEIGHT_PATH = './trained_weights/model_030.pt'

class AudioDenoiser:
    def __init__(self, filename, sr=22050, weight_path = BEST_WEIGHT_PATH):
        self.fn = filename
        self.sr = sr

        self.waveform = self._get_waveform(filename, sr)
        self.chunks, self.padding = self._split_chunks(self.waveform, 22050)

        # TO DO: load model here
        self.model = Denoiser()
        self.model.load_state_dict(torch.load(weight_path))
    

    def denoise(self, out_fp):
        denoised_waveform = self._denoise_waveform() 
        torchaudio.save(out_fp, denoised_waveform, self.sr)

    def _denoise_waveform(self):
        denoised_chunks = []
        for chunk in self.chunks:
            denoised_chunk = self.model(chunk)
            denoised_chunks.append(denoised_chunk)
        
        denoised_waveform = torch.cat(denoised_chunks, dim=1)
        denoised_waveform = denoised_waveform[:-self.padding]
        
        return denoised_waveform

    def _get_waveform(self, fn, new_sr):
        waveform, sr = torchaudio.load(fn)
    
        if sr != new_sr:
            resampler = torchaudio.transforms.Resample(sr, new_sr)
            resampled_waveform = resampler(waveform)

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