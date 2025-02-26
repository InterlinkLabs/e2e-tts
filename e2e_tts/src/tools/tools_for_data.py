import os
import sys

sys.path.append(".")
import tqdm
import json
import numpy as np

import torch
import torch.nn as nn

from models.g2p.g2p import normalize_phonemes as graph2phone
from .utils import *


def create_supervised_filelist(path: str, speakers: list) -> None:
    file_list = []
    print(f"Preprocessing & filter dataset: {json.dumps(speakers, ensure_ascii=False, indent=4)}")
    for spk in speakers:
        with open(os.path.join(path, spk, "metadata.lab"), "r", encoding="utf8") as f:
            metadata = [line.split("|") for line in f.read().split("\n") if line]

        for line in tqdm.tqdm(metadata, desc=spk):
            audio, text, phonemes = line
            audio = os.path.join(path, spk, "wavs", line[0])
            
            with open(os.path.join(path, spk, "durations", line[0].split(".")[0] + ".txt"), "r", encoding="utf8") as f:
                durations = f.read().strip()

            if len(phonemes.split()) != len(durations.split(", ")):
                print(text)
                print(phonemes)
                print(f"Missmatch from {spk} {os.path.basename(line[0])}: {len(phonemes.split())} - {len(durations.split(', '))}")
                exit()

            file_list.append("|".join([
                audio, 
                spk, 
                phonemes, 
                durations
            ]))

    print(f"Total dataset {len(file_list)} samples!!!")
    f = open(os.path.join(path, "file_list.txt"), "w", encoding="utf8")
    f.write("\n".join(file_list))


def create_unsupervised_filelist(path: str, speakers: list) -> None:
    file_list = []
    with open("models/g2p/dict/fix_wordss.txt", "r", encoding="utf8") as f:
        vn_words = [x.strip() for x in f.read().split("\n") if x] + [",", "."]
    print(f"Preprocessing & filter dataset: {json.dumps(speakers, ensure_ascii=False, indent=4)}")
    for spk in speakers:
        with open(os.path.join(path, spk, "metadata.csv"), "r", encoding="utf8") as f:
            metadata = [line.split("|") for line in f.read().split("\n") if line]
        with open(os.path.join(path, spk, "foreign_words.json"), "r", encoding="utf8") as f:
            foreign_dict = json.load(f)

        print(f"[==] speaker no.{speakers[spk]}: {spk}")
        for line in tqdm.tqdm(metadata, position=0, leave=False):
            file_name, text = line
            file_name = os.path.join(path, spk, "wavs", file_name)
            if not os.path.exists(file_name) or any(x not in vn_words for x in text.split()) > 0:
                continue

            # initilize phonemes
            phonemes, boundaries = graph2phone(text, foreign_dict)
            file_list.append("|".join([
                file_name, 
                spk,
                " ".join(phonemes), 
                ", ".join([str(b) for b in boundaries]), 
            ]))

    print(f"Total dataset {len(file_list)} samples!!!")
    f = open(os.path.join(path, "file_list.txt"), "w", encoding="utf8")
    f.write("\n".join(file_list))


def create_supervised_input(list_segments: list, stft: nn.Module, max_wav_value: int) -> dict:
    prosody_dict = {f"{segment[1]}_{os.path.basename(segment[0])}": {} for segment in list_segments}
    max_seq_len = 1000
    for line in tqdm.tqdm(list_segments, desc="Creating prosody"):
        file_name, speaker, _, _ = line

        src_path = os.path.dirname(os.path.dirname(file_name))
        os.makedirs(os.path.join(src_path, "mels"), exist_ok=True)
        mel_location = os.path.join(src_path, "mels", f"{os.path.basename(file_name).split('.')[0]}.npy")
        prosody_dict[f"{speaker}_{os.path.basename(file_name)}"]["mel"] = mel_location

        # create prosody path
        os.makedirs(os.path.join(src_path, "f0"), exist_ok=True)
        f0_location = os.path.join(src_path, "f0", f"{os.path.basename(file_name).split('.')[0]}.npy")
        prosody_dict[f"{speaker}_{os.path.basename(file_name)}"]["f0"] = f0_location

        os.makedirs(os.path.join(src_path, "pitch"), exist_ok=True)
        pitch_location = os.path.join(src_path, "pitch", f"{os.path.basename(file_name).split('.')[0]}.npy")
        prosody_dict[f"{speaker}_{os.path.basename(file_name)}"]["pitch"] = pitch_location

        os.makedirs(os.path.join(src_path, "energy"), exist_ok=True)
        energy_location = os.path.join(src_path, "energy", f"{os.path.basename(file_name).split('.')[0]}.npy")
        prosody_dict[f"{speaker}_{os.path.basename(file_name)}"]["energy"] = energy_location

        if all(os.path.exists(_) for _ in prosody_dict[f"{speaker}_{os.path.basename(file_name)}"].values()):
            melspec = np.load(mel_location)        
        else:
            audio, sampling_rate = load_wav_to_torch(file_name)
            assert sampling_rate == stft.sampling_rate, (
                f"Audio sample rate missmatch: given {sampling_rate} Hz, expected {stft.sampling_rate} SR"
            )
            audio_norm = audio / max_wav_value
            audio_norm = audio_norm.unsqueeze(0)
            audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
            melspec, energy = stft.mel_spectrogram(audio_norm, return_energy=True)
            melspec = melspec.squeeze(0).numpy()

            # save mel-spectrogram in np.array
            if not os.path.exists(mel_location):
                np.save(mel_location, melspec)

            # save f0s, pitches & energies
            audio_norm = audio_norm.squeeze().numpy()
            if not os.path.exists(f0_location):
                f0 = extract_f0(audio_norm, melspec.shape[1], stft.sampling_rate, stft.hop_length)
                np.save(f0_location, f0)

            if not os.path.exists(pitch_location):
                pitch = extract_pitch(audio_norm, stft.sampling_rate, stft.hop_length)
                np.save(pitch_location, pitch)

            if not os.path.exists(energy_location):
                energy = extract_energy(energy)
                np.save(energy_location, energy)

        max_seq_len = max_seq_len if max_seq_len > melspec.shape[1] else melspec.shape[1]

    return prosody_dict, max_seq_len


def create_unsupervised_input(list_segments: list, stft: nn.Module, max_wav_value: int) -> dict:
    prosody_dict = {f"{segment[1]}_{os.path.basename(segment[0])}": {} for segment in list_segments}
    max_seq_len, statistic_speakers = 1000, {}
    for line in tqdm.tqdm(list_segments, desc="Creating prosody"):
        file_name, speaker, phonemes, _ = line

        src_path = os.path.dirname(os.path.dirname(file_name))
        os.makedirs(os.path.join(src_path, "mels"), exist_ok=True)
        mel_location = os.path.join(src_path, "mels", f"{os.path.basename(file_name).split('.')[0]}.npy")
        prosody_dict[f"{speaker}_{os.path.basename(file_name)}"]["mel"] = mel_location

        # create prosody path
        os.makedirs(os.path.join(src_path, "f0"), exist_ok=True)
        f0_location = os.path.join(src_path, "f0", f"{os.path.basename(file_name).split('.')[0]}.npy")
        prosody_dict[f"{speaker}_{os.path.basename(file_name)}"]["f0"] = f0_location

        os.makedirs(os.path.join(src_path, "pitch"), exist_ok=True)
        pitch_location = os.path.join(src_path, "pitch", f"{os.path.basename(file_name).split('.')[0]}.npy")
        prosody_dict[f"{speaker}_{os.path.basename(file_name)}"]["pitch"] = pitch_location

        os.makedirs(os.path.join(src_path, "energy"), exist_ok=True)
        energy_location = os.path.join(src_path, "energy", f"{os.path.basename(file_name).split('.')[0]}.npy")
        prosody_dict[f"{speaker}_{os.path.basename(file_name)}"]["energy"] = energy_location

        os.makedirs(os.path.join(src_path, "mel2ph"), exist_ok=True)
        mel2ph_location = os.path.join(src_path, "mel2ph", f"{os.path.basename(file_name).split('.')[0]}.npy")
        prosody_dict[f"{speaker}_{os.path.basename(file_name)}"]["mel2ph"] = mel2ph_location

        if all(os.path.exists(_) for _ in prosody_dict[f"{speaker}_{os.path.basename(file_name)}"].values()):
            melspec = np.load(mel_location)
        else:
            audio, sampling_rate = load_wav_to_torch(file_name)
            assert sampling_rate == stft.sampling_rate, (
                f"Audio sample rate missmatch: given {sampling_rate} Hz, expected {stft.sampling_rate} SR"
            )
            audio_norm = audio / max_wav_value
            audio_norm = audio_norm.unsqueeze(0)
            audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
            melspec, energy = stft.mel_spectrogram(audio_norm, return_energy=True)
            melspec = melspec.squeeze(0).numpy()

            # save mel-spectrogram in np.array
            if not os.path.exists(mel_location):
                np.save(mel_location, melspec)

            # save f0s, pitches & energies
            audio_norm = audio_norm.squeeze().numpy()
            if not os.path.exists(f0_location):
                f0 = extract_f0(audio_norm, melspec.shape[1], stft.sampling_rate, stft.hop_length)
                np.save(f0_location, f0)

            if not os.path.exists(pitch_location):
                pitch = extract_pitch(audio_norm, stft.sampling_rate, stft.hop_length)
                np.save(pitch_location, pitch)

            if not os.path.exists(energy_location):
                energy = extract_energy(energy)
                np.save(energy_location, energy)

            # save attention priors
            if not os.path.exists(mel2ph_location):
                mel2ph = beta_binomial_prior_distribution(len(phonemes.split()), melspec.shape[1])
                np.save(mel2ph_location, mel2ph)

        max_seq_len = max_seq_len if max_seq_len > melspec.shape[1] else melspec.shape[1]
        if speaker not in statistic_speakers:
            statistic_speakers[speaker] = 0
        statistic_speakers[speaker] += melspec.shape[1]

    print("[*] Statistic duration per speaker:")
    for spk, tot_fr in statistic_speakers.items():
        print(f"- {spk}: {round((tot_fr * stft.hop_length / stft.sampling_rate) / 3600, 2)} hours")

    return prosody_dict, max_seq_len


def generate_mel(file_path: str, model: nn.Module, speakers: dict, stats: dict, device: torch.device=None, batch_size: int = 8) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device
    with open(file_path, "r", encoding="utf8") as f:
        file_list = [x.split("|") for x in f.read().split("\n") if x]

    # filter variable samples
    output, list_segments = [], []
    for i in tqdm.tqdm(range(len(file_list)), desc="Filter dataset"):
        file_name = os.path.basename(file_list[i][0])
        wavs_path = os.path.dirname(file_list[i][0])
        groundtruth_mels_path = os.path.join(os.path.dirname(wavs_path), "mels")
        predicted_mels_path = os.path.join(os.path.dirname(wavs_path), "predicted_mels")
        os.makedirs(predicted_mels_path, exist_ok=True)
        output.append([os.path.join(wavs_path, file_name), 
                       os.path.join(predicted_mels_path, file_name.split(".")[0] + ".npy"),
                       os.path.join(groundtruth_mels_path, file_name.split(".")[0] + ".npy"),
                       ])
        if not os.path.exists(os.path.join(predicted_mels_path, file_name.split(".")[0] + ".npy")):
            list_segments.append(file_list[i])

    # predicted-mels
    if model is not None and len(list_segments) > 0:
        model.eval()
        if len(file_list) > 0:
            print(f"Generated {len(file_list)}/{len(output)} miss mel-spectrograms!!!")
            batches = [file_list[i: i + batch_size] for i in range(len(file_list)) if i % batch_size == 0]
            for batch in tqdm.tqdm(batches, desc="Generated mel-spectrogram"):
                sorted_file, mel_lens, x = parse_input(
                    batch, 
                    speakers, 
                    stats,
                    model.config["variance"]["variance_embedding"]["use_uv"],
                    device
                )
                with torch.no_grad():
                    y_pred = model(x)[0]
                mel_pred = y_pred[1].transpose(1, 2).detach().cpu().numpy()
                for i in range(len(mel_pred)):
                    np.save(sorted_file[i], mel_pred[i][:, :mel_lens[i]])

    return output
