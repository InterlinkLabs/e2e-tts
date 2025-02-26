import os
import scipy
import requests
import parselmouth
import numpy as np
import pyworld as pw

from scipy.io import wavfile
from scipy.stats import betabinom

import torch
from models.g2p import _symbols_to_sequence 

# some global static config
f0_bin = 256
f0_min = 50.0
f0_max = 1100.0
f0_mel_min = 1127 * np.log(1 + f0_min / 700)
f0_mel_max = 1127 * np.log(1 + f0_max / 700)


def dynamic_range_compression(x, C=1, clip_val=1e-5):
    """
    PARAMS
    ------
    C: compression factor
    """
    return torch.log(torch.clamp(x, min=clip_val) * C)


def dynamic_range_decompression(x, C=1):
    """
    PARAMS
    ------
    C: compression factor used to compress
    """
    return torch.exp(x) / C


def load_wav_to_torch(path: str) -> torch.Tensor:
    sampling_rate, data = wavfile.read(path)

    return torch.FloatTensor(data.astype(np.float32)), sampling_rate


def extract_f0(wav_data: np.array, mel_len: int, sample_rate: int, hop_length: int, with_pitch: bool=False) -> np.array:
    # initilize param config
    assert hop_length in [128, 256]
    pad_size = 4 if hop_length == 128 else 2

    f0 = parselmouth.Sound(wav_data, sample_rate).to_pitch_ac(
        time_step=hop_length / sample_rate,
        voicing_threshold=0.5,
        pitch_floor=80,
        pitch_ceiling=750
    ).selected_array["frequency"]
    f0 = f0[:mel_len - 8] # to avoid negative rpad
    lpad = pad_size - 2
    rpad = mel_len - len(f0) - lpad
    f0 = np.pad(f0, [[lpad, rpad]], mode="constant")
    """
    Some problems here: 
        + mel and f0 are extracted by 2 different libraries => force to same length
        + new version of some libraries could cause "rpad" to be a negative values
    """
    delta = mel_len - len(f0)
    assert np.abs(delta) <= 8
    if delta > 0:
        f0 = np.concatenate([f0, [f0[-1]] * delta], 0)
    f0 = f0[: mel_len]

    if with_pitch is True:
        pitch_coarse = f0_to_coarse(f0)
        
        return f0, pitch_coarse
    else:
        
        return f0


def f0_to_coarse(f0: np.array):
    f0_mel = 1127 * np.log(1 + f0 / 700)
    f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - f0_mel_min) * (f0_bin - 2) / (f0_mel_max - f0_mel_min) + 1

    f0_mel[f0_mel <= 1] = 1
    f0_mel[f0_mel > f0_bin - 1] = f0_bin - 1
    f0_coarse = np.rint(f0_mel).astype(np.int)
    assert f0_coarse.max() <= 255 and f0_coarse.min() >= 1, (f0_coarse.max(), f0_coarse.min())

    return f0_coarse


def extract_pitch(wav_data: np.array, sample_rate: int, hop_length: int) -> np.array:
    """
    Extract pitch directly from raw audio waveform 
    f0_floor : float
        Lower F0 limit in Hz.
        Default: 71.0
    f0_ceil : float
        Upper F0 limit in Hz.
        Default: 800.0
    """
    # Extract Pitch/f0 from raw waveform using PyWORLD
    pitch, t = pw.dio(wav_data.astype(np.float64), sample_rate, frame_period=hop_length / sample_rate * 1000)
    pitch = pw.stonemask(wav_data.astype(np.float64), pitch, t, sample_rate)

    # perform linear interpolation
    nonzero_ids = np.where(pitch != 0)[0]
    interp_fn = scipy.interpolate.interp1d(
        x=nonzero_ids, 
        y=pitch[nonzero_ids],
        fill_value=(pitch[nonzero_ids[0]], pitch[nonzero_ids[-1]]), 
        bounds_error=False
    )
    pitch = interp_fn(np.arange(0, len(pitch)))

    return pitch


def extract_energy(energy: torch.Tensor) -> np.array:
    """
    Extract energy by calculate L1 norm in each frame / each window in STFT
    """
    energy = torch.squeeze(energy, 0).cpu().numpy().astype(np.float32)
    
    return energy


def beta_binomial_prior_distribution(phoneme_count, mel_count, scaling_factor=1.0):
    P, M = phoneme_count, mel_count
    x = np.arange(0, P)
    mel_text_probs = []
    for i in range(1, M + 1):
        a, b = scaling_factor * i, scaling_factor * (M + 1 - i)
        rv = betabinom(P, a, b)
        mel_i_prob = rv.pmf(x)
        mel_text_probs.append(mel_i_prob)

    return np.array(mel_text_probs)


def remove_outlier(in_dir):
    values = np.load(in_dir)
    p25 = np.percentile(values, 25)
    p75 = np.percentile(values, 75)
    lower = p25 - 1.5 * (p75 - p25)
    upper = p75 + 1.5 * (p75 - p25)
    normal_indices = np.logical_and(values > lower, values < upper)

    return values[normal_indices]


def parse_input(inputs: list, speakers: dict, stats: dict, use_uv: bool, device: torch.device):
    """
    Parse input to forward in models
    """
    file_name = [os.path.basename(x[0]) for x in inputs]
    wavs_path = [os.path.dirname(x[0]) for x in inputs]
    mels_path = [os.path.join(os.path.dirname(wavs_path[i]), "mels", file_name[i].split(".")[0] + ".npy") for i in range(len(file_name))]

    mel2ph_path = [os.path.join(os.path.dirname(wavs_path[i]), "mel2ph", file_name[i].split(".")[0] + ".npy") for i in range(len(file_name))]   
    pitches_path = [os.path.join(os.path.dirname(wavs_path[i]), "pitch", file_name[i].split(".")[0] + ".npy") for i in range(len(file_name))]
    f0s_path = [os.path.join(os.path.dirname(wavs_path[i]), "f0", file_name[i].split(".")[0] + ".npy") for i in range(len(file_name))]
    energies_path = [os.path.join(os.path.dirname(wavs_path[i]), "energy", file_name[i].split(".")[0] + ".npy") for i in range(len(file_name))]
    output_path = [os.path.join(os.path.dirname(wavs_path[i]), "predicted_mels", file_name[i].split(".")[0] + ".npy") for i in range(len(file_name))]

    texts = [torch.IntTensor(_symbols_to_sequence(x[2])) for x in inputs]
    mels = [torch.from_numpy(np.load(x)) for x in mels_path]
    mel2phes = [torch.from_numpy(np.load(x)) for x in mel2ph_path]
    pitches = [(torch.from_numpy(np.load(x)) - stats["pitch"]["mean"]) / stats["pitch"]["std"] for x in pitches_path]
    pitches = [pitches[i][:mels[i].shape[1]] for i in range(len(pitches))]
    f0s = [(np.load(x) - stats["f0"]["mean"]) / stats["f0"]["std"] for x in f0s_path]
    uvs = [torch.from_numpy(x == 0).float() for x in f0s]
    f0s = [torch.from_numpy(x) for x in f0s]
    energies = [(torch.from_numpy(np.load(x)) - stats["energy"]["mean"]) / stats["energy"]["std"] for x in energies_path]

    # padding data
    input_lengths, ids_sorted_decreasing = \
        torch.sort(torch.LongTensor([len(x) for x in texts]), dim=0, descending=True)
    max_input_len = input_lengths[0]
    num_mels = mels[0].size(0)
    max_output_len = max([x.size(1) for x in mels])

    text_padded = torch.zeros(len(inputs), max_input_len).long()
    mel2ph_padded = torch.zeros(len(inputs), max_output_len, max_input_len)
    mel_padded = torch.zeros(len(inputs), num_mels, max_output_len)
    pitch_padded = torch.zeros(len(inputs), max_output_len)
    f0_padded = torch.zeros(len(inputs), max_output_len)
    uv_padded = torch.zeros(len(inputs), max_output_len)
    energy_padded = torch.zeros(len(inputs), max_output_len)
    output_lengths = torch.LongTensor(len(inputs))

    for i in range(len(ids_sorted_decreasing)):
        text_padded[i, :texts[ids_sorted_decreasing[i]].size(0)] = texts[ids_sorted_decreasing[i]]
        mel2ph_padded[i, :mel2phes[ids_sorted_decreasing[i]].size(0), :mel2phes[ids_sorted_decreasing[i]].size(1)] = \
            mel2phes[ids_sorted_decreasing[i]]
        mel_padded[i, :, :mels[ids_sorted_decreasing[i]].size(1)] = mels[ids_sorted_decreasing[i]]
        pitch_padded[i, :pitches[ids_sorted_decreasing[i]].size(0)] = pitches[ids_sorted_decreasing[i]]
        f0_padded[i, :f0s[ids_sorted_decreasing[i]].size(0)] = f0s[ids_sorted_decreasing[i]]
        uv_padded[i, :energies[ids_sorted_decreasing[i]].size(0)] = uvs[ids_sorted_decreasing[i]]
        energy_padded[i, :energies[ids_sorted_decreasing[i]].size(0)] = energies[ids_sorted_decreasing[i]]
        output_lengths[i] = mels[ids_sorted_decreasing[i]].size(1)

    sorted_speakers = torch.LongTensor([speakers[inputs[ids_sorted_decreasing[i]][1]] for i in range(len(ids_sorted_decreasing))])
    sorted_file = [output_path[ids_sorted_decreasing[i]] for i in range(len(ids_sorted_decreasing))]

    if use_uv is True:
        pitch_padded = {
            "f0": f0_padded.to(device),
            "uv": uv_padded.to(device)
        }
    else:
        pitch_padded = pitch_padded.to(device)

    return sorted_file, \
        output_lengths, \
        (
            sorted_speakers.to(device),
            text_padded.to(device),
            mel_padded.transpose(1, 2).to(device),
            mel2ph_padded.to(device),
            pitch_padded,
            energy_padded.to(device),
            input_lengths.to(device),
            torch.max(input_lengths.data).item(),
            output_lengths.to(device),
            torch.max(output_lengths.data).item()
        )
