import torch
import torch.nn as nn
import torch.nn.functional as F

import os
# import psutil
# process = psutil.Process(os.getpid())

from .layers import VarianceAdaptor, Postnet
from .function import get_mask_from_lengths


class SupervisedFastSpeech2(nn.Module):
    """ FastSpeech2 """

    def __init__(self,
                 n_symbols: int,
                 n_speakers: int,
                 n_channels: int,
                 config: dict,
                 stats: dict,
                 device: torch.device = None,
                 ) -> None:
        super(SupervisedFastSpeech2, self).__init__()

        self.config = config
        self.building_block = self.config["building_block"]["block_type"]
        print(f"[*]Model use building block: {self.building_block}")
        
        if self.building_block == "transformer":
            from .blocks.transformer import Encoder, Decoder
        elif self.building_block == "conformer":
            from .blocks.conformer import Encoder, Decoder
        elif self.building_block == "fastformer":
            from .blocks.fastformer import Encoder, Decoder
        elif self.building_block == "lstransformer":
            from .blocks.lstransformer import Encoder, Decoder
        elif self.building_block == "reformer":
            from .blocks.reformer import Encoder, Decoder

        self.encoder = Encoder(
            layers=self.config["encoder_layers"],
            hidden_dim=self.config["encoder_hidden"],
            max_seq_len=self.config["max_seq_len"],
            n_symbols=n_symbols,
            config=self.config["building_block"][self.building_block]
        )
        self.decoder = Decoder(
            layers=self.config["decoder_layers"],
            hidden_dim=self.config["decoder_hidden"],
            max_seq_len=self.config["max_seq_len"],
            config=self.config["building_block"][self.building_block]
        )
        self.use_uv = self.config["variance"]["variance_embedding"]["use_uv"]
        self.variance_adaptor = VarianceAdaptor(
            hidden_dim=self.config["encoder_hidden"],
            config=self.config["variance"],
            stats=stats
        )
        self.mel_linear = nn.Linear(
            in_features=self.config["decoder_hidden"],
            out_features=n_channels
        )
        self.postnet = Postnet(
            n_channels=n_channels,
            config=self.config["postnet"]
        )
        self.speaker_emb = nn.Embedding(
            num_embeddings=n_speakers,
            embedding_dim=self.config["encoder_hidden"]
        )
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def parse_batch(self, batch):
        text_padded, input_lengths, _, mel_padded, output_lengths, speakers, \
            dur_padded, f0_padded, uv_padded, pit_padded, ener_padded = batch

        # base inputs
        spk = speakers.to(self.device)
        text_padded = text_padded.to(self.device)
        input_lengths = input_lengths.to(self.device)
        max_input_lens = torch.max(input_lengths.data).item()

        mel_padded = mel_padded.transpose(1, 2).to(self.device)
        output_lengths = output_lengths.to(self.device)
        max_output_lens = torch.max(output_lengths.data).item()

        # variance prosody inputs
        dur_padded = dur_padded.to(self.device)
        if self.use_uv is True:
            pit_padded = {"f0": f0_padded.to(self.device), "uv": uv_padded.to(self.device)}
        else:
            pit_padded = pit_padded.to(self.device)
        ener_padded = ener_padded.to(self.device)

        return ((spk, text_padded, dur_padded, pit_padded, ener_padded, input_lengths, max_input_lens, output_lengths, max_output_lens),
                (mel_padded, dur_padded))

    def forward(self, inputs, step):
        speakers, texts, d_targets, p_targets, e_targets,\
            src_lens, max_src_len, mel_lens, max_mel_len = inputs  # 10 elements per inputs batch
        src_masks = get_mask_from_lengths(src_lens, max_src_len)
        mel_masks = get_mask_from_lengths(mel_lens, max_mel_len)

        texts, _  = self.encoder(texts, src_masks)
        output = texts + self.speaker_emb(speakers).unsqueeze(1).expand(-1, max_src_len, -1) # add speaker embedding
    
        (
            output, 
            log_d_predictions, 
            duration_rounded, 
            pitch_predictions, 
            energy_predictions, 
            mel_lens, 
            mel_masks
        ), (
            p_targets,
            e_targets
        ) = \
            self.variance_adaptor(
            output, 
            src_lens, 
            src_masks, 
            mel_masks, 
            max_mel_len, 
            p_targets, 
            e_targets, 
            d_targets                                            
        )

        output, mel_masks = self.decoder(output, mel_masks)
        output = self.mel_linear(output)

        postnet_output = self.postnet(output) + output

        return (
            output, 
            postnet_output, 
            log_d_predictions, 
            duration_rounded, 
            pitch_predictions,
            energy_predictions, 
            src_masks, 
            mel_masks
        ), (
            p_targets,
            e_targets
        )

    def inference(self, speaker, texts, src_lens, max_src_len, d_control=1.0, p_control=1.0, e_control=1.0):
        src_masks = get_mask_from_lengths(src_lens, max_src_len)

        output, _ = self.encoder(texts, src_masks)
        # don't use in-place operator 
        output = output + self.speaker_emb(speaker).unsqueeze(1).expand(-1, max_src_len, -1)
        
        (
            output, 
            _, 
            duration_rounded, 
            _, 
            _, 
            mel_lens, 
            mel_masks
        ), (
            _,
            _
        ) = self.variance_adaptor(
            output, 
            src_lens, 
            src_masks, 
            d_control=d_control, 
            p_control=p_control, 
            e_control=e_control
        )

        output, mel_masks = self.decoder(output, mel_masks)
        output = self.mel_linear(output)

        postnet_output = output + self.postnet(output)

        return (
            output, 
            postnet_output, 
            duration_rounded
        ), mel_lens
