import torch
import torch.nn as nn
import torch.nn.functional as F

from .sublayers import *
from .function import *
from .blocks.utils import LinearNorm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class VarianceAdaptor(nn.Module):
    """ Variance Adaptor """

    def __init__(self,
                 n_channels: int,
                 hidden_dim: int,
                 config: dict,
                 stats: dict,
                 ) -> None:
        super(VarianceAdaptor, self).__init__()
        
        self.stats = stats
        self.hidden_size = hidden_dim
        self.n_channels = n_channels
        self.predictor_config = config['variance_predictor']
        self.predictor_grad = self.predictor_config["predictor_grad"]

        # initilize unsupervised duration learning
        self.aligner = AlignmentEncoder(
                n_mel_channels=self.n_channels,
                n_att_channels=self.n_channels,
                n_text_channels=self.hidden_size,
                temperature=config["duration_modelling"]["aligner_temperature"]
            )
        self.binarization_start_steps = config["duration_modelling"]["binarization_start_steps"]
        self.duration_predictor = DurationPredictor(
            self.hidden_size,
            n_chans=self.n_channels,
            n_layers=self.predictor_config["dur_predictor_layers"],
            dropout_rate=self.predictor_config["dropout"], 
            padding=self.predictor_config["ffn_padding"],
            kernel_size=self.predictor_config["dur_predictor_kernel"],
        )
        self.length_regulator = LengthRegulator()

        # initilize pitch learning
        self.use_uv = config["variance_embedding"]["use_uv"]
        self.pitch_feature_level = config["variance_embedding"]["pitch_feature"]
        assert self.pitch_feature_level in ["frame_level", "phoneme_level"]
        self.pitch_predictor = VariancePredictor(
            idim=self.hidden_size,
            n_chans=self.predictor_config["filter_size"],
            n_layers=self.predictor_config["pit_predictor_layers"],
            dropout_rate=self.predictor_config["dropout"],
            odim=2 if self.use_uv is True else 1,
            padding=self.predictor_config["ffn_padding"], 
            kernel_size=self.predictor_config["pit_predictor_kernel"]
        )
        self.pitch_embedding = nn.Embedding(
            num_embeddings=config["variance_embedding"]["n_bins" if self.use_uv is True else "f0_bins"],
            embedding_dim=self.hidden_size
        )

        self.pitch_quantization = config["variance_embedding"]["pitch_quantization"]
        assert self.pitch_quantization in ["linear", "log"]
        if self.pitch_quantization == "log":
            self.pitch_bins = nn.Parameter(
                data=torch.exp(torch.linspace(
                    np.log(self.stats["pitch"]["min"]), 
                    np.log(self.stats["pitch"]["max"]), 
                    config["variance_embedding"]["n_bins"] - 1
                )),
                requires_grad=False
            )
        else:
            self.pitch_bins = nn.Parameter(
                data=torch.linspace(
                    self.stats["pitch"]["min"], 
                    self.stats["pitch"]["max"], 
                    config["variance_embedding"]["n_bins"] - 1
                ),
                requires_grad=False
            )

        # initilize energy learning
        self.energy_feature_level = config["variance_embedding"]["energy_feature"]
        assert self.energy_feature_level in ["frame_level", "phoneme_level"]
        self.energy_predictor = VariancePredictor(
            idim=self.hidden_size,
            n_chans=self.predictor_config["filter_size"],
            n_layers=self.predictor_config["ener_predictor_layers"],
            dropout_rate=self.predictor_config["dropout"],
            odim=1,
            padding=self.predictor_config["ffn_padding"], 
            kernel_size=self.predictor_config["ener_predictor_kernel"]
        )
        self.energy_embedding = nn.Embedding(
            num_embeddings=config["variance_embedding"]["n_bins"],
            embedding_dim=self.hidden_size
        )
        
        self.energy_quantization = config["variance_embedding"]["energy_quantization"]
        assert self.energy_quantization in ["linear", "log"]
        if self.energy_quantization == "log":
            self.energy_bins = nn.Parameter(
                data=torch.exp(torch.linspace(
                    np.log(self.stats["energy"]["min"]),
                    np.log(self.stats["energy"]["max"]), 
                    config["variance_embedding"]["n_bins"] - 1
                )),
                requires_grad=False
            )
        else:
            self.energy_bins = nn.Parameter(
                data=torch.linspace(
                    self.stats["energy"]["min"], 
                    self.stats["energy"]["max"], 
                    config["variance_embedding"]["n_bins"] - 1
                ),
                requires_grad=False
            )

    def binarize_attention_parallel(self, attn, in_lens, out_lens):
        """For training purposes only. Binarizes attention with MAS.
        These will no longer recieve a gradient.
        Args:
            attn: B x 1 x max_mel_len x max_text_len
        """
        with torch.no_grad():
            attn_cpu = attn.data.cpu().numpy()
            attn_out = b_mas(attn_cpu, in_lens.cpu().numpy(), out_lens.cpu().numpy(), width=1)

        return torch.from_numpy(attn_out).to(attn.device)

    def get_pitch_embedding(self, x, target, control):
        # use_uv=True mean use f0-spectrograms and uv marks else mean use default pitch-spectrograms
        x = x.detach() + self.predictor_grad * (x - x.detach())
        prediction = self.pitch_predictor(x, squeeze=False if self.use_uv is True else True)
        if self.use_uv:
            if target is not None:
                f0s = target["f0"]
                uvs = target["uv"]
            else:
                prediction = prediction * control
                f0s = prediction[:, :, 0]
                uvs = prediction[:, :, 1] > 0
            
            if self.pitch_quantization == "log":
                f0s_denorm = 2 ** f0s
            else:
                f0s_denorm = f0s * self.stats["f0"]["std"] + self.stats["f0"]["mean"]
            f0s_denorm[uvs > 0] = 0
            pitch = tensor_f0_to_coarse(f0s_denorm)

        else:
            pitch = target if target is not None else prediction * control
            pitch = torch.bucketize(pitch, self.pitch_bins)

        embedding = self.pitch_embedding(pitch)

        return prediction, embedding

    def get_energy_embedding(self, x, target, control):
        # energy-spectrograms
        x = x.detach() + self.predictor_grad * (x - x.detach())
        prediction = self.energy_predictor(x, squeeze=True)
        energy = target if target is not None else prediction * control
        energy = torch.bucketize(energy, self.energy_bins)

        embedding = self.energy_embedding(energy)

        return prediction, embedding
 
    def forward(
        self, 
        x,
        txt_emb,
        txt_lens, 
        txt_mask,
        max_txt_len,
        spk_emb, 
        mel=None, 
        mel_lens=None,
        mel_mask=None, 
        max_mel_len=None,
        step=None,
        pitch_target=None, 
        energy_target=None, 
        attn_prior=None, # duration_targets=None,
        p_control=1.0, 
        e_control=1.0, 
        d_control=1.0    
    ):
        x = x + spk_emb.unsqueeze(1).expand(-1, x.shape[1], -1) 
        
        # learning unsupervised durations
        log_duration_prediction = self.duration_predictor(
            x.detach() + self.predictor_grad * (x - x.detach()), txt_mask
        )

        if attn_prior is not None:
            attn_soft, attn_logprob = self.aligner(
                mel.transpose(1, 2),
                txt_emb.transpose(1, 2),
                txt_mask.unsqueeze(-1),
                attn_prior,
                spk_emb,
            )
            attn_hard = self.binarize_attention_parallel(attn_soft, txt_lens, mel_lens)
            attn_hard_dur = attn_hard.sum(2)[:, 0, :]
            attn_out = (attn_soft, attn_hard, attn_hard_dur, attn_logprob)
            
            duration_rounded = attn_hard_dur
        else:
            attn_out = None

            duration_rounded = torch.clamp(
                (torch.round(torch.exp(log_duration_prediction) - 1) * d_control),
                min=0,
            )

        # phoneme level feature
        x_tmp = x.clone()
        if self.pitch_feature_level == "phoneme_level":
            if isinstance(pitch_target, dict):
                pitch_target = {k: get_phoneme_level(v, txt_lens, duration_rounded) for k, v in pitch_target.items()}

                # will optimize thi in futures
                pitch_target["uv"] = pitch_target["uv"].double()
                pitch_target["uv"] = torch.where(pitch_target["uv"] == 1.0, pitch_target["uv"], 0.0).float()
            elif pitch_target is not None:
                pitch_target = get_phoneme_level(pitch_target, txt_lens, duration_rounded)
            pitch_prediction, pitch_embedding = self.get_pitch_embedding(x, pitch_target, p_control)
            x_tmp = x_tmp + pitch_embedding
        if self.energy_feature_level == "phoneme_level":
            if energy_target is not None:
                energy_target = get_phoneme_level(energy_target, txt_lens, duration_rounded)
            energy_prediction, energy_embedding = self.get_energy_embedding(x, energy_target, e_control)
            x_tmp = x_tmp + energy_embedding
        x = x_tmp.clone()

        # expand duration (txt -> mel-specs)
        if attn_prior is not None and step < self.binarization_start_steps:
            x = torch.bmm(attn_soft.squeeze(1), x)
        else:
            x, mel_lens = self.length_regulator(x, duration_rounded, max_mel_len)
            mel_mask = get_mask_from_lengths(mel_lens)

        # frame level feature
        x_tmp = x.clone()
        if self.pitch_feature_level == "frame_level":
            pitch_prediction, pitch_embedding = self.get_pitch_embedding(x, pitch_target, p_control)
            x_tmp = x_tmp + pitch_embedding
        if self.energy_feature_level == "frame_level":
            energy_prediction, energy_embedding = self.get_energy_embedding(x, energy_target, e_control)
            x_tmp = x_tmp + energy_embedding
        x = x_tmp.clone()

        return (
            x, 
            log_duration_prediction, 
            duration_rounded, 
            pitch_prediction, 
            energy_prediction, 
            mel_lens, 
            mel_mask, 
            attn_out
        ), (
            pitch_target,
            energy_target
        )


class AlignmentEncoder(nn.Module):
    """ Alignment Encoder for Unsupervised Duration Modeling """

    def __init__(self, 
                n_mel_channels,
                n_att_channels,
                n_text_channels,
                temperature):
        super().__init__()
        self.temperature = temperature
        self.softmax = nn.Softmax(dim=3)
        self.log_softmax = nn.LogSoftmax(dim=3)

        self.key_proj = nn.Sequential(
            ConvNorm(
                n_text_channels,
                n_text_channels * 2,
                kernel_size=3,
                bias=True,
                w_init_gain='relu'
            ),
            nn.ReLU(),
            ConvNorm(
                n_text_channels * 2,
                n_att_channels,
                kernel_size=1,
                bias=True,
            ),
        )

        self.query_proj = nn.Sequential(
            ConvNorm(
                n_mel_channels,
                n_mel_channels * 2,
                kernel_size=3,
                bias=True,
                w_init_gain='relu',
            ),
            nn.ReLU(),
            ConvNorm(
                n_mel_channels * 2,
                n_mel_channels,
                kernel_size=1,
                bias=True,
            ),
            nn.ReLU(),
            ConvNorm(
                n_mel_channels,
                n_att_channels,
                kernel_size=1,
                bias=True,
            ),
        )

        self.key_spk_proj = LinearNorm(n_text_channels, n_text_channels)
        self.query_spk_proj = LinearNorm(n_text_channels, n_mel_channels)

    def forward(self, queries, keys, mask=None, attn_prior=None, speaker_embed=None):
        """Forward pass of the aligner encoder.
        Args:
            queries (torch.tensor): B x C x T1 tensor (probably going to be mel data).
            keys (torch.tensor): B x C2 x T2 tensor (text data).
            mask (torch.tensor): uint8 binary mask for variable length entries (should be in the T2 domain).
            attn_prior (torch.tensor): prior for attention matrix.
            speaker_embed (torch.tensor): B x C tnesor of speaker embedding for multi-speaker scheme.
        Output:
            attn (torch.tensor): B x 1 x T1 x T2 attention mask. Final dim T2 should sum to 1.
            attn_logprob (torch.tensor): B x 1 x T1 x T2 log-prob attention mask.
        """
        if speaker_embed is not None:
            keys = keys + self.key_spk_proj(speaker_embed.unsqueeze(1).expand(
                -1, keys.shape[-1], -1
            )).transpose(1, 2)
            queries = queries + self.query_spk_proj(speaker_embed.unsqueeze(1).expand(
                -1, queries.shape[-1], -1
            )).transpose(1, 2)
        keys_enc = self.key_proj(keys)  # B x n_attn_dims x T2
        queries_enc = self.query_proj(queries)

        # Simplistic Gaussian Isotopic Attention
        attn = (queries_enc[:, :, :, None] - keys_enc[:, :, None]) ** 2  # B x n_attn_dims x T1 x T2
        attn = -self.temperature * attn.sum(1, keepdim=True)

        if attn_prior is not None:
            # print(f"AlignmentEncoder \t| mel: {queries.shape} phone: {keys.shape} mask: {mask.shape} attn: {attn.shape} attn_prior: {attn_prior.shape}")
            attn = self.log_softmax(attn) + torch.log(attn_prior[:, None] + 1e-8)
            # print(f"AlignmentEncoder \t| After prior sum attn: {attn.shape}")

        attn_logprob = attn.clone()

        if mask is not None:
            attn.data.masked_fill_(mask.permute(0, 2, 1).unsqueeze(2), -float("inf"))

        attn = self.softmax(attn)  # softmax along T2
        return attn, attn_logprob


class DurationPredictor(nn.Module):
    """Duration predictor module.
    This is a module of duration predictor described in `FastSpeech: Fast, Robust and Controllable Text to Speech`_.
    The duration predictor predicts a duration of each frame in log domain from the hidden embeddings of encoder.
    .. _`FastSpeech: Fast, Robust and Controllable Text to Speech`:
        https://arxiv.org/pdf/1905.09263.pdf
    Note:
        The outputs are calculated in log domain.
    """

    def __init__(self, idim, n_layers=2, n_chans=384, kernel_size=3, dropout_rate=0.1, offset=1.0, padding="SAME"):
        """Initilize duration predictor module.
        Args:
            idim (int): Input dimension.
            n_layers (int, optional): Number of convolutional layers.
            n_chans (int, optional): Number of channels of convolutional layers.
            kernel_size (int, optional): Kernel size of convolutional layers.
            dropout_rate (float, optional): Dropout rate.
            offset (float, optional): Offset value to avoid nan in log domain.
        """
        super(DurationPredictor, self).__init__()
        self.offset = offset
        self.conv = nn.ModuleList()
        self.kernel_size = kernel_size
        self.padding = padding
        for idx in range(n_layers):
            in_chans = idim if idx == 0 else n_chans
            self.conv += [nn.Sequential(
                nn.ConstantPad1d(((kernel_size - 1) // 2, (kernel_size - 1) // 2)
                                       if padding == "SAME"
                                       else (kernel_size - 1, 0), 0),
                nn.Conv1d(in_chans, n_chans, kernel_size, stride=1, padding=0),
                nn.ReLU(),
                LayerNorm(n_chans, dim=1),
                nn.Dropout(dropout_rate)
            )]
        self.linear = nn.Linear(n_chans, 1)

    def forward(self, xs, x_masks=None):
        xs = xs.transpose(1, -1)  # (B, idim, Tmax)
        for f in self.conv:
            xs = f(xs)  # (B, C, Tmax)
            if x_masks is not None:
                xs = xs * (1 - x_masks.float())[:, None, :]

        xs = self.linear(xs.transpose(1, -1))  # [B, T, C]
        xs = xs * (1 - x_masks.float())[:, :, None]  # (B, T, C)

        return xs.squeeze(-1) # (B, Tmax)


class LengthRegulator(nn.Module):
    """ Length Regulator """

    def __init__(self):
        super(LengthRegulator, self).__init__()

    def LR(self, x, duration, max_len):
        output = list()
        mel_len = list()
        for batch, expand_target in zip(x, duration):
            expanded = self.expand(batch, expand_target)
            output.append(expanded)
            mel_len.append(expanded.shape[0])

        if max_len is not None:
            output = pad(output, max_len)
        else:
            output = pad(output)

        return output, torch.LongTensor(mel_len).to(device)

    def expand(self, batch, predicted):
        out = list()

        for i, vec in enumerate(batch):
            expand_size = predicted[i].item()
            out.append(vec.expand(max(int(expand_size), 0), -1))
        out = torch.cat(out, 0)

        return out

    def forward(self, x, duration, max_len):
        output, mel_len = self.LR(x, duration, max_len)
        
        return output, mel_len


class VariancePredictor(nn.Module):
    """ Pitch & Energy Predictor """

    def __init__(self, idim, n_layers=5, n_chans=384, odim=2, kernel_size=5, dropout_rate=0.1, padding="SAME"):
        """Initilize pitch predictor module.
        Args:
            idim (int): Input dimension.
            n_layers (int, optional): Number of convolutional layers.
            n_chans (int, optional): Number of channels of convolutional layers.
            kernel_size (int, optional): Kernel size of convolutional layers.
            dropout_rate (float, optional): Dropout rate.
        """
        super(VariancePredictor, self).__init__()
        self.conv = torch.nn.ModuleList()
        self.kernel_size = kernel_size
        self.padding = padding
        for idx in range(n_layers):
            in_chans = idim if idx == 0 else n_chans
            self.conv += [torch.nn.Sequential(
                nn.ConstantPad1d(((kernel_size - 1) // 2, (kernel_size - 1) // 2)
                                       if padding == "SAME"
                                       else (kernel_size - 1, 0), 0),
                nn.Conv1d(in_chans, n_chans, kernel_size, stride=1, padding=0),
                nn.ReLU(),
                LayerNorm(n_chans, dim=1),
                nn.Dropout(dropout_rate)
            )]
        self.linear = torch.nn.Linear(n_chans, odim)
        self.embed_positions = SinusoidalPositionalEmbedding(idim, 0, init_size=4096)
        self.pos_embed_alpha = nn.Parameter(torch.Tensor([1]))

    def forward(self, xs, squeeze=False):
        """

        :param xs: [B, T, H]
        :return: [B, T, H]
        """
        positions = self.pos_embed_alpha * self.embed_positions(xs[..., 0])
        xs = xs + positions
        xs = xs.transpose(1, -1)  # (B, idim, Tmax)
        for f in self.conv:
            xs = f(xs)  # (B, C, Tmax)
        # NOTE: calculate in log domain
        xs = self.linear(xs.transpose(1, -1))  # (B, Tmax, H)

        return xs.squeeze(-1) if squeeze else xs


class Postnet(nn.Module):
    """
    Post-net:
        - Five 1-d convolution with 512 channels and kernel size 5
    """

    def __init__(self,
                 n_channels: int,
                 config: dict,
                 ) -> None:
        super(Postnet, self).__init__()
        self.convolutions = nn.ModuleList()

        self.convolutions.append(
            nn.Sequential(
                ConvNorm(in_channels=n_channels,
                         out_channels=config["embedding_dim"],
                         kernel_size=config["kernel_size"],
                         stride=1,
                         padding=int((config["kernel_size"] - 1) / 2),
                         dilation=1,
                         w_init_gain='tanh'),
                nn.BatchNorm1d(num_features=config["embedding_dim"]))
        )

        for _ in range(1, config["conv_layers"] - 1):
            self.convolutions.append(
                nn.Sequential(
                    ConvNorm(in_channels=config["embedding_dim"],
                             out_channels=config["embedding_dim"],
                             kernel_size=config["kernel_size"], stride=1,
                             padding=int((config["kernel_size"] - 1) / 2),
                             dilation=1, w_init_gain='tanh'),
                    nn.BatchNorm1d(num_features=config["embedding_dim"]))
            )

        self.convolutions.append(
            nn.Sequential(
                ConvNorm(in_channels=config["embedding_dim"],
                         out_channels=n_channels,
                         kernel_size=config["kernel_size"],
                         stride=1,
                         padding=int((config["kernel_size"] - 1) / 2),
                         dilation=1,
                         w_init_gain='linear'),
                nn.BatchNorm1d(n_channels))
        )

    def forward(self, x):
        x = x.contiguous().transpose(1, 2)
        for i in range(len(self.convolutions) - 1):
            x = F.dropout(torch.tanh(self.convolutions[i](x)), 0.5, self.training)
        x = F.dropout(self.convolutions[-1](x), 0.5, self.training)
        x = x.contiguous().transpose(1, 2)

        return x
