import os
import sys

sys.path.append(".")
import tgt
import yaml
import json
import tqdm
import numpy as np

import torch

from tools.stft import TorchSTFT, generate_melspecs
from tools.tools_for_data import load_wav_to_torch
from models.g2p.g2p import normalize_phonemes


def merge_idx(values, max_idx):
    new_values = []
    for i in range(len(values[:-1])):
        txt = values[i][1]
        s = values[i][0] if i != 0 else 0
        e = values[i + 1][0]
        
        new_values.append([list(range(s, e)), txt])

    new_values.append([list(range(values[-1][0], max_idx)), values[-1][1]])
    # display target
    # print(f"max index: {max_idx}")
    # for i in range(len(values)):
    #     print(f"{values[i]} -> {new_values[i]}")

    return new_values


def extract_durations(list_grid, list_phonemes, mel_len):
    global config
    tgt_idx = [[i, list_grid[i].text] for i in range(len(list_grid)) if list_grid[i].text != ""]
    tgt_idx = merge_idx(tgt_idx, len(list_grid))
    ph_idx = [[i, ph] for i, ph in enumerate(list_phonemes) if ph not in ["<SILENT>", "</S>"]]
    ph_idx = merge_idx(ph_idx, len(list_phonemes))
    if len(tgt_idx) != len(ph_idx):
        print(f"{len(tgt_idx)} vs {len(ph_idx)}")
        for i in range(len(tgt_idx)):
            print(f"{tgt_idx[i]} - {ph_idx[i]}")
        exit()
    else:
        phonemes = []
        durations, left_overs = [], 0.0
        for i, line in enumerate(zip(tgt_idx, ph_idx)):
            [src_idx, src_ph], [dst_idx, dst_ph] = line
            if src_ph != dst_ph:
                print(f"merge {src_ph} to {dst_ph}")

            if len(src_idx) == 1 or len(dst_idx) == 1:
                phs = [dst_ph]
                start_values = [float(list_grid[src_idx[0]].start_time)]
                end_values = [float(list_grid[src_idx[-1]].end_time)]
            else:
                phs = [dst_ph, "<SILENT>" if i != len(tgt_idx) - 1 else "</S>"]
                start_values = [float(list_grid[src_idx[0]].start_time), float(list_grid[src_idx[-1]].start_time)]
                end_values = [float(list_grid[src_idx[-2]].end_time), float(list_grid[src_idx[-1]].end_time)]
            
            # print(f"{start_values[0]} - {end_values[-1]}")
            phonemes.extend(phs)
            for values in zip(start_values, end_values):
                s = values[0] * config["audio"]["signal"]["sampling_rate"] / config["audio"]["stft"]["hop_length"]
                e = values[1] * config["audio"]["signal"]["sampling_rate"] / config["audio"]["stft"]["hop_length"]

                float_value = e - s
                int_value = round(e - s)
                
                durations.append(int_value)
                left_overs += float_value - int_value
                if left_overs > 1:
                    durations[-1] += 1
                    left_overs -= 1
                elif left_overs < -1:
                    durations[-1] -= 1
                    left_overs += 1

    durations[-1] += round(left_overs) 
    total_frames = sum(durations)

    missing_frame = total_frames - mel_len

    if missing_frame > 0:
        max_value_idx = durations.index(max(durations))
        durations[max_value_idx] -= abs(missing_frame)
    elif missing_frame < 0:
        durations[-1] += abs(missing_frame)

    return phonemes, durations


if __name__ == "__main__":
    input_folder = sys.argv[1]
    output_folder = sys.argv[2]

    config = yaml.load(open("config/config.yaml", "r"), Loader=yaml.FullLoader)
    stft = TorchSTFT(config["audio"]["stft"]["filter_length"], config["audio"]["stft"]["hop_length"], config["audio"]["stft"]["win_length"],
                     config["audio"]["mel"]["channels"], config["audio"]["signal"]["sampling_rate"], config["audio"]["mel"]["mel_fmin"],
                     config["audio"]["mel"]["mel_fmax"])
    list_speakers = [_ for _ in os.listdir(input_folder) if os.path.isdir(os.path.join(input_folder, _))]

    print(list_speakers)
    for spk in list_speakers:
        os.makedirs(os.path.join(output_folder, spk, "mels"), exist_ok=True)
        with open(os.path.join(output_folder, spk, "metadata.csv"), "r", encoding="utf8") as f:
            metadata = [_.split("|") for _ in f.read().split("\n") if _]

        with open(os.path.join(output_folder, spk, "foreign_words.json"), "r", encoding="utf8") as f:
            foreign_words = json.load(f)
            if len(foreign_words) == 0:
                foreign_words = None

        mfa_metadata = []
        os.makedirs(os.path.join(output_folder, spk, "durations"), exist_ok=True)
        for line in tqdm.tqdm(metadata, desc=f"{spk}"):
            [file_name, text] = line
            tgt_file = os.path.join(input_folder, spk, f"{file_name.split('.')[0]}.TextGrid")
            if not os.path.exists(tgt_file):
                continue
            textgrid = tgt.read_textgrid(os.path.join(input_folder, spk, f"{file_name.split('.')[0]}.TextGrid"), include_empty_intervals=True)

            phonemes = [_ for _ in normalize_phonemes(line[1].replace("-", " "), foreign_words)[0]]
            # if os.path.exists(os.path.join(output_folder, spk, "mels", f"{file_name.split('.')[0]}.npy")):
            #     mel = np.load(os.path.join(output_folder, spk, "mels", f"{file_name.split('.')[0]}.npy"))
            # else:
            audio, sampling_rate = load_wav_to_torch(os.path.join(output_folder, spk, "wavs", file_name))
            assert sampling_rate == stft.sampling_rate, (
                f"Audio sample rate missmatch: given {sampling_rate} Hz, expected {stft.sampling_rate} SR")
            audio = audio / config["audio"]["signal"]["max_wav_value"]
            audio = audio.unsqueeze(0)
            audio = torch.autograd.Variable(audio, requires_grad=False)

            mel = stft.mel_spectrogram(audio).squeeze(0).numpy()
            np.save(os.path.join(output_folder, spk, "mels", f"{file_name.split('.')[0]}.npy"), mel)

            phonemes, durations = extract_durations(textgrid.get_tier_by_name("phones")._get_intervals(), phonemes, mel.shape[1])
            with open(os.path.join(output_folder, spk, "durations", f"{file_name.split('.')[0]}.txt"), "w") as f:
                f.write(", ".join([str(_) for _ in durations]))

            mfa_metadata.append("|".join([file_name, " ".join(text), " ".join(phonemes)]))

        print(f"from {len(metadata)} to {len(mfa_metadata)}")
        print(os.path.join(output_folder, spk, "metadata.lab"))
        with open(os.path.join(output_folder, spk, "metadata.lab"), "w", encoding="utf8") as f:
            f.write("\n".join(mfa_metadata))
