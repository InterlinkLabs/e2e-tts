import torch
import torch.nn as nn

from .layers import VarianceAdaptor, Postnet
from .function import get_mask_from_lengths


class UnsupervisedFastSpeech2(nn.Module):
    """ Unsupervised durations learning FastSpeech2 """

    def __init__(self,
                 n_symbols: int,
                 n_speakers: int,
                 n_channels: int,
                 config: dict,
                 stats: dict,
                 device: torch.device = None,
                 ) -> None:
        super(UnsupervisedFastSpeech2, self).__init__()

        self.config = config
        self.building_block = self.config["building_block"]["block_type"]
        
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
            n_channels=n_channels,
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
        text_padded, input_lengths, boundaries, mel_padded, output_lengths, speakers, \
            mel2ph_padded, f0_padded, uv_padded, pit_padded, ener_padded = batch

        # base inputs
        spk = speakers.to(self.device)
        text_padded = text_padded.to(self.device)
        input_lengths = input_lengths.to(self.device)
        max_input_lens = torch.max(input_lengths.data).item()

        mel_padded = mel_padded.transpose(1, 2).to(self.device)
        output_lengths = output_lengths.to(self.device)
        max_output_lens = torch.max(output_lengths.data).item()

        # variance prosody inputs
        mel2ph_padded = mel2ph_padded.to(self.device)
        if self.use_uv is True:
            pit_padded = {"f0": f0_padded.to(self.device), "uv": uv_padded.to(self.device)}
        else:
            pit_padded = pit_padded.to(self.device)
        ener_padded = ener_padded.to(self.device)

        return ((spk, text_padded, mel_padded, mel2ph_padded, pit_padded, ener_padded, input_lengths, max_input_lens, output_lengths, max_output_lens),
                (text_padded, boundaries, mel_padded))

    def forward(self, inputs, step=0):
        speakers, texts, mels, attn_priors, p_targets, e_targets,\
            txt_lens, max_txt_len, mel_lens, max_mel_len = inputs  # 10 elements per inputs batch

        txt_masks = get_mask_from_lengths(txt_lens, max_txt_len)
        mel_masks = get_mask_from_lengths(mel_lens, max_mel_len)

        x, txt_embed = self.encoder(texts, txt_masks)
        spk_embed = self.speaker_emb(speakers)

        (
            output, 
            log_d_predictions, 
            _, 
            pitch_predictions,
            energy_predictions, 
            mel_lens,
            mel_masks,
            attn_out
        ), (
            p_targets,
            e_targets
        ) = self.variance_adaptor(
            x,
            txt_embed,
            txt_lens,
            txt_masks,
            max_txt_len,
            spk_embed,
            mels,
            mel_lens,
            mel_masks,
            max_mel_len,
            step,
            p_targets, 
            e_targets, 
            attn_priors
        )

        output, mel_masks = self.decoder(output, mel_masks)
        output = self.mel_linear(output)

        postnet_output = self.postnet(output) + output

        return (
            output,
            postnet_output,
            log_d_predictions,
            pitch_predictions,
            energy_predictions,
            txt_lens,
            txt_masks, 
            mel_lens,
            mel_masks,
            attn_out
        ), (
            p_targets,
            e_targets
        )

    def inference(self, speaker, texts, txt_lens, max_txt_len, d_control=1.0, p_control=1.0, e_control=1.0):
        txt_masks = get_mask_from_lengths(txt_lens, max_txt_len)

        x, txt_embed = self.encoder(texts, txt_masks)
        spk_embed = self.speaker_emb(speaker)

        (
            output, 
            _, 
            duration_rounded, 
            _,
            _, 
            mel_lens,
            mel_masks,
            _
        ), (
            _,
            _
        ) = self.variance_adaptor(
            x,
            txt_embed,
            txt_lens,
            txt_masks,
            max_txt_len,
            spk_embed,
            p_control=p_control, 
            e_control=e_control, 
            d_control=d_control
        )
        
        output, mel_masks = self.decoder(output, mel_masks)
        output = self.mel_linear(output)

        postnet_output = self.postnet(output) + output

        return (
            output, 
            postnet_output, 
            duration_rounded
        ), mel_lens
