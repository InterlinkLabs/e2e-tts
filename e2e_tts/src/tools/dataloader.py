import os
import tqdm
import json
import random
import numpy as np
from librosa.util import normalize
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from .stft import TorchSTFT
from .tools_for_data import load_wav_to_torch
from .utils import remove_outlier
from models.g2p import _symbols_to_sequence


def prepare_dataloaders(dataset_path: str, config: dict, speakers: dict) -> DataLoader:
    with open(os.path.join(dataset_path), "r", encoding="utf8") as f:
        dataset = [_.split("|") for _ in f.read().split("\n") if _]
    random.shuffle(dataset)
    # split train/test dataset
    train_set = dataset[:-50]
    learn_alignment = config["models"]["fastspeech2"]["variance"]["duration_modelling"]["learn_alignment"]
    default_stats = {
        "f0": {
            "mean": 191.4634071641572,
            "std": 67.69532232524328
        },
        "pitch": {
            "max": 10.331658770062944,
            "min": -2.046769430004249,
            "mean": 185.01941262431194,
            "std": 62.520829576129046
        },
        "energy": {
            "max": 7.350775241851807,
            "min": -1.2576030492782593,
            "mean": 35.940244380504524,
            "std": 28.567237649702538
        }
    }


    train_set = TextMelLoader(
        train_set, config["audio"], config["models"]["fastspeech2"]["variance"]["variance_embedding"], learn_alignment, speakers, default_stats
    )
    print(json.dumps(train_set.stats, ensure_ascii=False, indent=4))
    valid_set = dataset[-50:]
    valid_set = TextMelLoader(
        valid_set, config["audio"], config["models"]["fastspeech2"]["variance"]["variance_embedding"], learn_alignment, speakers, train_set.stats
    )
    collate_fn = TextMelCollate(learn_alignment)
    max_seq_len = train_set.max_seq_len if train_set.max_seq_len >= valid_set.max_seq_len else valid_set.max_seq_len

    return train_set.stats, max_seq_len, \
        DataLoader(train_set, num_workers=1, shuffle=True,
                   sampler=None, batch_size=config["train"]["batch_size"],
                   drop_last=True, collate_fn=collate_fn), \
        DataLoader(valid_set, num_workers=1, shuffle=False,
                   sampler=None, batch_size=1,
                   pin_memory=False, drop_last=False, collate_fn=collate_fn)


class TextMelLoader(Dataset):
    """ Dataloader for training acoustic model"""
    def __init__(self,
                 dataset_list: list,
                 stft_config: dict,
                 variance_config: dict,
                 learn_alignment: bool,
                 speakers: dict,
                 stats: dict = None,
                 ):
        self.inputs = dataset_list
        random.shuffle(self.inputs)

        self.speakers = speakers
        self.max_wav_value = stft_config["signal"]["max_wav_value"]
        self.sampling_rate = stft_config["signal"]["sampling_rate"]
        self.stft = TorchSTFT(
            stft_config["stft"]["filter_length"], stft_config["stft"]["hop_length"], stft_config["stft"]["win_length"],
            stft_config["mel"]["channels"], stft_config["signal"]["sampling_rate"], 
            stft_config["mel"]["mel_fmin"], stft_config["mel"]["mel_fmax"]
        )
        self.variance_config = variance_config
        self.use_uv = variance_config["use_uv"]
        self.pitch_eps = 0.000000001 # specific hard-code (change follow config)
        self.learn_alignment = learn_alignment
        if self.learn_alignment is True:
            from .tools_for_data import create_unsupervised_input as create_input
        else:
            from .tools_for_data import create_supervised_input as create_input
        self.prosody_path, self.max_seq_len = create_input(
            list_segments=self.inputs, 
            stft=self.stft, 
            max_wav_value=self.max_wav_value
        )

        if stats is None:
            self.stats = self.build_stats()
        else:
            self.stats = stats
        
    def build_stats(self):
        f0_scaler = []
        pitch_scaler = StandardScaler()
        energy_scaler = StandardScaler()

        list_segments = ["_".join([_[1], os.path.basename(_[0])]) for _ in self.inputs]
        for file_name in tqdm.tqdm(list_segments, desc="Computing statistic quantities"):
            # build f0 stats
            values = np.load(self.prosody_path[file_name]["f0"])
            f0_scaler.append(values)
            # build pitch stats
            values = remove_outlier(self.prosody_path[file_name]["pitch"])
            if len(values) > 0:
                pitch_scaler.partial_fit(values.reshape((-1, 1)))
            # build energy stats
            values = remove_outlier(self.prosody_path[file_name]["energy"])
            if len(values) > 0:
                energy_scaler.partial_fit(values.reshape((-1, 1)))

        f0_scaler = np.concatenate(f0_scaler, 0)
        f0_scaler = f0_scaler[f0_scaler != 0]
        f0_values = {"mean": np.mean(f0_scaler).item(), "std": np.std(f0_scaler).item()}
        pitch_values = {"max": np.finfo(np.float64).min, "min": np.finfo(np.float64).max,
                        "mean": pitch_scaler.mean_[0], "std": pitch_scaler.scale_[0]}
        energy_values = {"max": np.finfo(np.float64).min, "min": np.finfo(np.float64).max,
                         "mean": energy_scaler.mean_[0], "std": energy_scaler.scale_[0]}
        
        for file_name in tqdm.tqdm(list_segments, desc="Build model stats"):
            # build pitch stats
            values = (np.load(
                self.prosody_path[file_name]["pitch"]) - pitch_values["mean"]) / pitch_values["std"]
            pitch_values["max"] = max(pitch_values["max"], max(values))
            pitch_values["min"] = min(pitch_values["min"], min(values))
            # build energy stats
            values = (np.load(
                self.prosody_path[file_name]["energy"]) - energy_values["mean"]) / energy_values["std"]
            energy_values["max"] = max(energy_values["max"], max(values))
            energy_values["min"] = min(energy_values["min"], min(values))
            # build mel-spectrogram stats
            values = np.load(self.prosody_path[file_name]["mel"])

        return {
            "f0": {k: float(v) for k, v in f0_values.items()},
            "pitch": {k: float(v) for k, v in pitch_values.items()},
            "energy": {k: float(v) for k, v in energy_values.items()},
        }

    def get_mel(self, file_name):
        melspec = torch.from_numpy(np.load(self.prosody_path[file_name]["mel"]))
        assert melspec.size(0) == self.stft.n_mel_channels, (
            f"Mel dimension mismatch: given {melspec.size(0)}, expected {self.stft.n_mel_channels}")

        return melspec

    def get_text(self, text):
        text_norm = torch.IntTensor(_symbols_to_sequence(text))

        return text_norm

    def get_boundary(self, bound):

        return [int(_) for _ in bound.split(", ")]
    
    def get_mel2ph(self, file_name):
        mel2ph = np.load(self.prosody_path[file_name]["mel2ph"])

        return torch.from_numpy(mel2ph)

    def get_dur(self, dur):
        dur = torch.IntTensor([int(_) for _ in dur.split(", ")])

        return dur

    def get_prosody(self, file_name, prosody_type):
        pros = np.load(self.prosody_path[file_name][prosody_type])
        pros = (pros - self.stats[prosody_type]["mean"]) / self.stats[prosody_type]["std"]            

        return torch.from_numpy(pros)

    def get_f0(self, file_name):
        f0 = np.load(self.prosody_path[file_name]["f0"])
        uv = f0 == 0
        
        f0 = np.log2(f0 + self.pitch_eps) if self.variance_config["pitch_quantization"] == "log" else \
            (f0 - self.stats["f0"]["mean"]) / self.stats["f0"]["std"]
        if sum(uv) == len(f0):
            f0[uv] = 0
        elif sum(uv) > 0:
            f0[uv] = np.interp(np.where(uv)[0], np.where(~uv)[0], f0[~uv])

        return torch.from_numpy(f0), torch.from_numpy(uv).float()

    def speaker2id(self, spk):

        return torch.IntTensor([self.speakers[spk]])

    def parse_input(self, data):
        file_name = f"{data[1]}_{os.path.basename(data[0])}"

        text = self.get_text(data[2])
        mel = self.get_mel(file_name)

        spk = self.speaker2id(data[1])
        if self.learn_alignment is True:
            bound = self.get_boundary(data[3])
            align = self.get_mel2ph(file_name) 
        else:
            bound = None
            align = self.get_dur(data[3])

        f0, uv = self.get_f0(file_name)
        pit = self.get_prosody(file_name, "pitch")
        ener = self.get_prosody(file_name, "energy")

        return {
            "text": text,
            "boundaries": bound, 
            "mel_spectrogram": mel, 
            "speaker": spk, 
            "alignment": align, 
            "f0": f0, 
            "uv": uv,
            "pitch": pit[:mel.shape[1]], 
            "energy": ener
        }

    def __getitem__(self, index):
        return self.parse_input(self.inputs[index])

    def __len__(self):
        return len(self.inputs)


class TextMelCollate():
    """ Zero-pads model inputs and targets based on number of frames per step"""

    def __init__(self, learn_alignment: bool=True):

        self.learn_alignment = learn_alignment

    def pad_text(self, text):
        # Right zero-pad all one-hot text sequences to max input length
        input_lengths, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([len(x) for x in text]),
            dim=0, descending=True)
        max_input_len = input_lengths[0]

        text_padded = torch.zeros(len(text), max_input_len).long()

        for i in range(len(ids_sorted_decreasing)):
            text_padded[i, :text[ids_sorted_decreasing[i]].size(0)] = text[ids_sorted_decreasing[i]]

        return text_padded, input_lengths, ids_sorted_decreasing, max_input_len

    def pad_mel(self, mel, ids_sorted_decreasing):
        # Right zero-pad mel-spec
        num_mels = mel[0].size(0)
        max_target_len = max([x.size(1) for x in mel])
        # mel padded
        mel_padded = torch.zeros(len(mel), num_mels, max_target_len)
        output_lengths = torch.LongTensor(len(mel))

        for i in range(len(ids_sorted_decreasing)):
            mel_padded[i, :, :mel[ids_sorted_decreasing[i]].size(1)] = mel[ids_sorted_decreasing[i]]
            output_lengths[i] = mel[ids_sorted_decreasing[i]].size(1)

        return mel_padded, output_lengths, max(output_lengths)

    def pad_attn_prior(self, mel2ph, max_mel_len, max_ph_len, ids_sorted_decreasing):
        # Right zero-pad mel-spec
        mel2ph_padded = torch.zeros(len(mel2ph), max_mel_len, max_ph_len)
        for i in range(len(ids_sorted_decreasing)):
            mel2ph_padded[i, :mel2ph[ids_sorted_decreasing[i]].size(0), :mel2ph[ids_sorted_decreasing[i]].size(1)] = \
                mel2ph[ids_sorted_decreasing[i]]
        
        return mel2ph_padded

    def pad_dur(self, durations, max_len, ids_sorted_decreasing):
        duration_padded = torch.zeros(len(durations), max_len)
        for i in range(len(ids_sorted_decreasing)):
            duration_padded[i, :durations[ids_sorted_decreasing[i]].size(0)] = durations[ids_sorted_decreasing[i]]

        return duration_padded

    def pad_prosody(self, f0s, uvs, pitches, energies, max_len, ids_sorted_decreasing):
        f0_padded = torch.zeros(len(f0s), max_len)
        uv_padded = torch.zeros(len(uvs), max_len)
        pitch_padded = torch.zeros(len(pitches), max_len)
        energy_padded = torch.zeros(len(energies), max_len)

        for i in range(len(ids_sorted_decreasing)):
            f0_padded[i, :f0s[ids_sorted_decreasing[i]].size(0)] = f0s[ids_sorted_decreasing[i]]
            pitch_padded[i, :pitches[ids_sorted_decreasing[i]].size(0)] = pitches[ids_sorted_decreasing[i]]
            energy_padded[i, :energies[ids_sorted_decreasing[i]].size(0)] = energies[ids_sorted_decreasing[i]]
            uv_padded[i, :uvs[ids_sorted_decreasing[i]].size(0)] = uvs[ids_sorted_decreasing[i]]

        return f0_padded, uv_padded, pitch_padded, energy_padded

    def __call__(self, batch):
        """ Collate"s training batch from normalized text and mel-spectrograms """
        texts = [x["text"] for x in batch]
        text_padded, input_lengths, ids_sorted_decreasing, max_input_lengths = self.pad_text(texts)
        speakers = torch.LongTensor([batch[idx]["speaker"] for idx in ids_sorted_decreasing])
        boundaries = [batch[idx]["boundaries"] for idx in ids_sorted_decreasing]
        mels = [x["mel_spectrogram"] for x in batch]
        mel_padded, output_lengths, max_output_lengths = self.pad_mel(mels, ids_sorted_decreasing)
            
        align = [x["alignment"] for x in batch]
        if self.learn_alignment is True:
            align_padded = self.pad_attn_prior(align, max_output_lengths, max_input_lengths, ids_sorted_decreasing)
        else:
            align_padded = self.pad_dur(align, max_input_lengths, ids_sorted_decreasing)

        f0s = [x["f0"] for x in batch]
        uvs = [x["uv"] for x in batch]
        pitches = [x["pitch"] for x in batch]
        energies = [x["energy"] for x in batch]
        f0_padded, uv_padded, pitch_padded, energy_padded = \
            self.pad_prosody(f0s, uvs, pitches, energies, max_output_lengths, ids_sorted_decreasing)

        return text_padded, input_lengths, boundaries, mel_padded, output_lengths, speakers, \
            align_padded, f0_padded, uv_padded, pitch_padded, energy_padded


class MelAudioLoader(Dataset):
    """ Dataloader for training vocodear model"""
    def __init__(
        self,
        data_files: list,
        config: dict,
        segment_size: int = 8192,
        load_mel_from_disk: bool = True,
    ) -> None:
        self.inputs = data_files
        random.shuffle(self.inputs)

        self.segment_size = segment_size
        self.config = config
        self.stft = TorchSTFT(config["stft"]["filter_length"], config["stft"]["hop_length"], config["stft"]["win_length"],
                              config["mel"]["channels"], config["signal"]["sampling_rate"], config["mel"]["mel_fmin"],
                              config["mel"]["mel_fmax"])
        self.max_wav_value = self.config["signal"]["max_wav_value"]
        self.hop_size = self.config["stft"]["hop_length"]
        self.load_mel_from_disk = load_mel_from_disk

    def __getitem__(self, index):
        file_name = self.inputs[index]
        audio, sampling_rate = load_wav_to_torch(file_name[0])
        audio = audio / self.max_wav_value
        if sampling_rate != self.config["signal"]["sampling_rate"]:
            raise ValueError("{} SR doesn't match target {} SR".format(
                sampling_rate, self.config["signal"]["sampling_rate"]))

        # degrade audio if doesn't use mel from acoustic models
        if not self.load_mel_from_disk:
            audio = torch.FloatTensor(normalize(audio.numpy() * 0.95))
        audio = audio.unsqueeze(0)

        if not self.load_mel_from_disk:
            if audio.size(1) >= self.segment_size:
                max_audio_start = audio.size(1) - self.segment_size
                audio_start = random.randint(0, max_audio_start)
                audio = audio[:, audio_start:audio_start + self.segment_size]
            else:
                audio = F.pad(audio, (0, self.segment_size - audio.size(1)), "constant")

            mel = self.stft.mel_spectrogram(audio)
            mel_loss = self.stft.mel_spectrogram(audio)

        else:
            mel_loss = torch.from_numpy(np.load(file_name[2])).unsqueeze(0)
            mel = torch.from_numpy(np.load(file_name[1])).unsqueeze(0)

            frames_per_seg = self.segment_size // self.hop_size
            if audio.size(1) >= self.segment_size:
                mel_start = random.randint(0, mel.size(2) - frames_per_seg - 1)
                mel = mel[:, :, mel_start:mel_start + frames_per_seg]
                mel_loss = mel_loss[:, :, mel_start:mel_start + frames_per_seg]

                audio = audio[:, mel_start * self.hop_size:(mel_start + frames_per_seg) * self.hop_size]
            else:
                mel = F.pad(mel, (0, frames_per_seg - mel.size(2)), "constant")
                mel_loss = F.pad(mel_loss, (0, frames_per_seg - mel.size(2)), "constant")

                audio = F.pad(audio, (0, self.segment_size - audio.size(1)), "constant")

        # print(f"{os.path.basename(file_name[0])}: {mel.shape} - {mel_loss.shape} - {audio.shape}")
        return (mel.squeeze(0), audio.squeeze(0), mel_loss.squeeze(0))

    def __len__(self):
        return len(self.inputs)
