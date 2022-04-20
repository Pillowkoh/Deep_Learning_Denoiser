import torchaudio
import torch
import torch.functional as F
from scipy.io import wavfile
from model import Denoiser

class AudioDenoiser:
    def __init__(self, filename, sr=22050):
        self.fn = filename
        self.sr = sr

        self.waveform = self._get_waveform(filename, sr)
        self.chunks, self.padding = self._split_chunks(self.waveform, 22050)

        # TO DO: load model here
        self.model = Denoiser()
    

    def denoise(self, out_fp):
        # TO DO: add model
        # denoised_waveform = self._denoise_waveform() 
        # wavfile.write(out_fp, self.sr, denoised_waveform)
        wavfile.write(out_fp, self.sr, self.waveform)

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