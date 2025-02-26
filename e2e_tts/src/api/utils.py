import os
import yaml
import json
import time
import ffmpy
import numpy as np
from datetime import datetime
from pydub import AudioSegment

import torch
import torch.nn as nn

from models import HifiGan
from modules.upload.mps_storage import DefaultMPS, ServiceMPS
from g2p import symbols, text_to_sequence


default_storage = DefaultMPS()
service_storage = ServiceMPS()


class TTS(nn.Module):
    def __init__(self, 
                 acoustic_path: str, 
                 vocoder_path: str, 
                 max_len: int = 300, 
                 device: torch.device = None
                 ):
        super(TTS, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device

        # load acoustic
        self.config = yaml.load(open(os.path.join(os.path.dirname(acoustic_path), "config.yaml"), "r"), Loader=yaml.FullLoader)
        self.speakers = json.load(open(os.path.join(os.path.dirname(acoustic_path), "speakers.json"), "r"))
        self.stats = json.load(open(os.path.join(os.path.dirname(acoustic_path), "stats.json"), "r"))
        if self.config["models"]["fastspeech2"]["variance"]["duration_modelling"]["learn_alignment"] is True:
            from models import UnsupervisedFastSpeech2 as FastSpeech2
        else:
            from models import SupervisedFastSpeech2 as FastSpeech2
        self.acoustic = FastSpeech2(
            n_symbols=len(symbols),
            n_speakers=len(self.speakers),
            n_channels=self.config["audio"]["mel"]["channels"],
            config=self.config["models"]["fastspeech2"],
            stats=self.stats
        )
        checkpoint_dict = torch.load(acoustic_path, map_location="cpu")
        self.acoustic.load_state_dict(checkpoint_dict["state_dict"])
        self.acoustic.eval().to(self.device)

        # load vocoder
        self.vocoder = HifiGan(config=self.config["models"]["hifigan"])
        checkpoint_dict = torch.load(vocoder_path, map_location="cpu")
        self.vocoder.load_state_dict(checkpoint_dict["state_dict"])
        self.vocoder.eval().to(self.device)
        
        self.hop_length = self.config["audio"]["stft"]["hop_length"]
        self.sample_rate = self.config["audio"]["signal"]["sampling_rate"]
        self.max_wav_value = 32768.0
        
        self.max_len = max_len

    def arrange_text(self, text):
        arranged_text = []
        for line in text:
            if round(len(line) / self.max_len) != 1:
                line = line.split(" , ")
                arranged_text.append(line[0])
                line.pop(0)
                while len(line) > 0:
                    if len(arranged_text[-1]) >= self.max_len:
                        arranged_text.append(line[0])
                    else:
                        arranged_text[-1] = " , ".join([arranged_text[-1], line[0]])
                    line.pop(0)
            else:
                arranged_text.append(line)

        return arranged_text

    def input_parse(self, input_texts):
        input_sequences = [torch.LongTensor(text_to_sequence(txt)) for txt in self.arrange_text(input_texts)]
        input_lens, indices = torch.sort(torch.LongTensor([len(x) for x in input_sequences]), dim=0, descending=True)
        _, revert_indices = torch.sort(indices)
        
        sorted_sequences = [input_sequences[i] for i in indices.tolist()]
        sorted_inputs = []
        s, e, total_lens = 0, 0, 0
        for i, seq_len in enumerate(input_lens):
            if s == e or total_lens + seq_len <= self.max_len:
                e = i + 1
                total_lens += seq_len
            else:
                sorted_inputs.append([s, e])
                s, total_lens = e, 0

        if sorted_inputs == [] or sorted_inputs[-1][-1] != len(input_sequences):
            sorted_inputs.append([s, len(input_sequences)])
        
        sorted_inputs = [[
            nn.utils.rnn.pad_sequence(sorted_sequences[s: e], batch_first=True), 
            input_lens[s: e],
        ] for s, e in sorted_inputs]

        return sorted_inputs, revert_indices

    def combine_audio(self, audios, lengths, distance):
        output_audio = []
        for i, audio in enumerate(audios):
            audio = audio[: lengths[i] * self.hop_length]
            audio = (audio * self.max_wav_value)
            disOfsil = np.zeros(distance)

            output_audio.extend([audio, disOfsil])

        return np.concatenate(output_audio).astype("int16")

    def inference(
        self, 
        texts: list, 
        speaker_id: str, 
        pitch_control: float=1.0, 
        energy_control: float=1.0, 
        duration_control: float=1.0,
        silence_distance: float=0.5
    ):
        inputs, revert_indices = self.input_parse(texts)
        output_audios, output_lengths = [], []
        with torch.no_grad():
            for batch in inputs:
                input_sequences, input_lengths = batch
                spk_id = torch.tensor([self.speakers[speaker_id]]).to(self.device)
                (_, mel_predicted, _), mel_lengths = \
                    self.acoustic.inference(
                        speaker=spk_id, 
                        texts=input_sequences.to(self.device), 
                        txt_lens=input_lengths.to(self.device), 
                        max_txt_len=max(input_lengths),
                        p_control=pitch_control, 
                        e_control=energy_control, 
                        d_control=duration_control
                    )
                audio_predicted = self.vocoder(mel_predicted.transpose(1, 2)).squeeze(1)
                audio_predicted = audio_predicted.detach().cpu().numpy()

                output_audios.extend(audio_predicted)
                output_lengths.extend(mel_lengths)

        output_audios = [output_audios[i] for i in revert_indices.tolist()]
        output_lengths = [output_lengths[i] for i in revert_indices.tolist()]

        generated_audio = self.combine_audio(
            audios=output_audios,
            lengths=output_lengths,
            distance=int(silence_distance * self.sample_rate)
        )
        print(f"Audio Saved: {time.strftime('%H:%M:%S', time.gmtime(generated_audio.size / 22050))}")

        return generated_audio


def audio_speed_change(input_path: str, output_path: str = None, speed_rate: float=1.0):
    if output_path is None:
        file_type = input_path.split('.')[-1]
        output_path = f"{input_path[:-len(file_type) - 1]}_{round(speed_rate, 2)}.{file_type}"

    ff = ffmpy.FFmpeg(inputs={input_path: None},
                      outputs={output_path: ["-filter:a", "atempo={}".format(speed_rate), "-y"]})
    ff.run()

    return output_path


def save_wav(datas, rate=22050, speed=1.0, audio_format="wav", path_audio=None, return_binary=0):
    if path_audio is None:
        path_audio = os.path.join("audio_generated", "audio_{}_{}.{}"\
            .format(datetime.today().strftime("%Y_%m_%d_%H_%M_%S"), time.time(), audio_format))

    os.makedirs(os.path.dirname(path_audio), exist_ok=True)
    audio = AudioSegment(data=datas.tobytes(),
                         frame_rate=22050,
                         sample_width=datas.dtype.itemsize, 
                         channels=1).set_frame_rate(rate)
    audio.export(path_audio, format="ipod" if audio_format == "m4a" else audio_format)
    final_path = audio_speed_change(input_path=path_audio, speed_rate=speed) if speed != 1.0 else path_audio

    return default_storage.upload(final_path) if return_binary == 0 else final_path


def upload_service(datas, upload_path, audio_format, speed_rate=1.0):
    save_path = os.path.join("audio/cnnd_audio/", os.path.dirname(upload_path))
    os.makedirs(save_path, exist_ok=True)

    upload_file = os.path.join(save_path, os.path.basename(upload_path))
    save_file = os.path.join(save_path, f"{os.path.basename(upload_path)[: - len(audio_format) - 1]}.raw.{audio_format}") \
        if speed_rate != 1.0 else upload_file
    
    audio = AudioSegment(datas.tobytes(),
                         frame_rate=22050,
                         sample_width=datas.dtype.itemsize, 
                         channels=1)
    audio.export(save_file, format="ipod" if audio_format == "m4a" else audio_format, parameters=["-strict", "-2"])

    if speed_rate != 1.0:
        audio_speed_change(input_path=save_file, output_path=upload_file, speed_rate=speed_rate)

    return service_storage.upload(upload_file)
