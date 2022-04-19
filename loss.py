from scipy.signal import stft
import torch
import torch.nn.functional as F


class DenoiserLoss(torch.nn.Module):
    def __init__(self):
        super(DenoiserLoss, self).__init__()

    def forward(self, clean, denoised):
        l1_loss = F.l1_loss(clean, denoised)
        multi_resolution_stft_loss = MutltiResolutionSTFTLoss()
        sc_loss, mag_loss = multi_resolution_stft_loss(clean, denoised)

        loss = l1_loss + sc_loss + mag_loss
        return loss


class MutltiResolutionSTFTLoss(torch.nn.Module):
    def __init__(self, 
                 fft_sizes = [512, 1024, 2048], 
                 hop_sizes = [60, 120, 240], 
                 win_lengths = [240, 600, 1200],
                 sc_factor = 0.1,
                 mag_factor = 0.1):
        super(MutltiResolutionSTFTLoss, self).__init__()
        assert len(fft_sizes) == len(hop_sizes) == len(win_lengths)

        self.sc_factor = sc_factor
        self.mag_factor = mag_factor

        self.stft_losses = torch.nn.ModuleList()
        for fft_size, hop_size, win_length in zip(fft_sizes, hop_sizes, win_lengths):
            self.stft_losses.append(STFTLoss(fft_size, hop_size, win_length))
        

    def forward(self, clean, denoised):
        total_sc_loss, total_mag_loss = 0,0
        for stft_loss in self.stft_losses:
            sc_loss, mag_loss = stft_loss(clean, denoised)
            total_sc_loss += sc_loss
            total_mag_loss += mag_loss
        
        avg_sc_loss = self.sc_factor * total_sc_loss / len(self.stft_losses)
        avg_mag_loss = self.mag_factor * total_mag_loss / len(self.stft_losses)

        return avg_sc_loss, avg_mag_loss


class STFTLoss(torch.nn.Module):
    def __init__(self, fft_size, hop_size, win_length, sc_factor = 0.1, mag_factor = 0.1):
        super(STFTLoss, self).__init__()
        self.fft_size = fft_size
        self.hop_size = hop_size
        self.win_length = win_length
        self.sc_factor = sc_factor
        self.mag_factor = mag_factor
    
    def forward(self, clean, denoised):
        stft_clean = self._stft(clean)
        stft_denoised = self._stft(denoised)
        sc_loss = self._spectral_conversion_loss(stft_clean, stft_denoised)
        mag_loss = self._magnitude_loss(stft_clean, stft_denoised)

        return sc_loss, mag_loss

    # def _stft(self, x, window="hann_window"):
    def _stft(self, x):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        window = torch.hann_window(self.win_length).to(device)
        x_stft = torch.stft(x, self.fft_size, self.hop_size, self.win_length, window, return_complex=False)
        real = x_stft[..., 0]
        imag = x_stft[..., 1]

        return torch.sqrt(torch.clamp(real ** 2 + imag ** 2, min=1e-7)).transpose(2,1)
    
    def _spectral_conversion_loss(self, stft_clean, stft_denoised):
        num = torch.norm(stft_clean - stft_denoised)
        denom = torch.norm(stft_clean)

        return num / denom
    
    def _magnitude_loss(self, stft_clean, stft_denoised):
        return F.l1_loss(torch.log(stft_clean), torch.log(stft_denoised))