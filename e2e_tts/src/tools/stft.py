import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

import librosa
from .utils import *


class TorchSTFT(nn.Module):
    def __init__(self,
                 filter_length=1024,
                 hop_length=256,
                 win_length=1024,
                 n_mel_channels=80,
                 sampling_rate=22050,
                 mel_fmin=0.0,
                 mel_fmax=8000.0,
                 device=None):
        super(TorchSTFT, self).__init__()
        self.device = "cpu" if device is None else device

        self.sampling_rate = sampling_rate
        self.n_mel_channels = n_mel_channels

        self.filter_length = filter_length
        self.hop_length = hop_length
        self.win_length = win_length
        self.fmin = mel_fmin
        self.fmax = mel_fmax

        self.stft_pad = (int((self.filter_length - self.hop_length) / 2), int((self.filter_length - self.hop_length) / 2))
        mel_basis = librosa.filters.mel(
            sr=self.sampling_rate, 
            n_fft=self.filter_length, 
            n_mels=self.n_mel_channels, 
            fmin=self.fmin, 
            fmax=self.fmax
        )
        mel_basis = torch.from_numpy(mel_basis).float()

        self.register_buffer("mel_basis", mel_basis)
        self.window = torch.hann_window(self.win_length).to(self.device)

    def mel_spectrogram(self, input_data, center=False, return_energy=False):
        """Computes mel-spectrograms from a batch of waves
        PARAMS
        ------
        y: Variable(torch.FloatTensor) with shape (B, T) in range [-1, 1]

        RETURNS
        -------
        melspec: torch.FloatTensor of shape (B, n_mel_channels, T)
        """
        assert (torch.min(input_data.data) >= -1)
        assert (torch.max(input_data.data) <= 1)

        # padding for torch.stft
        input_data = F.pad(
            input=input_data.unsqueeze(1), 
            pad=self.stft_pad, 
            mode="reflect"
        ).squeeze(1)
        magnitudes = torch.stft(
            input_data, 
            n_fft=self.filter_length, 
            hop_length=self.hop_length, 
            win_length=self.win_length, 
            window=self.window,
            center=center, 
            pad_mode="reflect", 
            normalized=False, 
            onesided=True, 
            return_complex=False
        )
        magnitudes = torch.sqrt(magnitudes.pow(2).sum(-1) + (1e-9))
        magnitudes = magnitudes.data

        melspec = torch.matmul(self.mel_basis, magnitudes)
        melspec = dynamic_range_compression(melspec)

        if return_energy is True:
            energy = torch.norm(magnitudes, dim=1)

            return melspec, energy
        else:

            return melspec
    
    def inverse_tranform(self, magnitude, phase):
        input_data = magnitude * torch.exp(phase * 1j)
        inverse_transform = torch.istft(
            input_data,
            n_fft=self.filter_length, 
            hop_length=self.hop_length, 
            win_length=self.win_length, 
            window=self.window
        )

        return inverse_transform.unsqueeze(-2)  # unsqueeze to stay consistent with conv_transpose1d implementation
                                                                                                                                                                                                                                                                                    

# this is building up for training hifigan 
mel_basis = {}
hann_window = {}
def generate_melspecs(y, n_fft=1024, num_mels=80, sampling_rate=22050, hop_size=256, win_size=1024, fmin=0.0, fmax=8000.0, center=False) -> torch.Tensor:
    if torch.min(y) < -1.:
        warnings.warn(f"min value is {torch.min(y).item()}")
    if torch.max(y) > 1.:
        warnings.warn(f"max value is {torch.max(y).item()}")

    global mel_basis, hann_window
    if fmax not in mel_basis:
        mel = librosa.filters.mel(
            sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax
        )
        mel_basis[str(fmax) + "_" + str(y.device)] = torch.from_numpy(mel).float().to(y.device)
        hann_window[str(y.device)] = torch.hann_window(win_size).to(y.device)

    y = F.pad(
        input=y.unsqueeze(1), 
        pad=(int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)), 
        mode="reflect"
    )
    y = y.squeeze(1)

    spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window[str(y.device)],
                      center=center, pad_mode="reflect", normalized=False, onesided=True, return_complex=False)
    spec = torch.sqrt(spec.pow(2).sum(-1) + (1e-9))

    spec = torch.matmul(mel_basis[str(fmax) + "_" + str(y.device)], spec)
    spec = dynamic_range_compression(spec)

    return spec


def inverse_stft(magnitude, phase, n_fft=1024, hop_size=256, win_size=1024):
    hann_window = torch.hann_window(win_size).to(magnitude.device)
    inverse_transform = torch.istft(
        magnitude * torch.exp(phase * 1j),
        n_fft=n_fft, 
        hop_length=hop_size, 
        win_length=win_size, 
        window=hann_window
    )

    return inverse_transform.unsqueeze(-2)  # unsqueeze to stay consistent with conv_transpose1d implementation
