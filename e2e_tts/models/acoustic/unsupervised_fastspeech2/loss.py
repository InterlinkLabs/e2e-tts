import torch
import torch.nn as nn
import torch.nn.functional as F

from .function import phone2words


class UnsupervisedFastSpeech2Loss(nn.Module):
    """ Unsupervised-durations learning FastSpeech2 Loss """

    def __init__(self, config, pitch_feature_level, energy_feature_level, use_uv):
        super(UnsupervisedFastSpeech2Loss, self).__init__()
        self.sum_loss = ForwardSumLoss()
        self.bin_loss = BinLoss()
        
        self.use_uv = use_uv
        self.binarization_loss_enable_steps = config["binarization_loss_enable_steps"]
        self.binarization_loss_warmup_steps = config["binarization_loss_warmup_steps"]
        self.dur_loss_lambda = config["dur_loss_lambda"]

        self.pitch_feature_level = pitch_feature_level
        self.energy_feature_level = energy_feature_level
    
    def build_duration_losses(self, src_inputs, log_duration_predictions, duration_targets, word_boundaries, masks) -> dict:
        B, T = src_inputs.shape
        nonpadding = masks.float()
        duration_targets = duration_targets.float() * nonpadding
        duration_predictions = torch.clamp(torch.exp(log_duration_predictions) - 1, min=0)

        loss = {}
        # calculate phonemes duration loss (log_d_predictions, attn_hard_dur)
        duration_targets.requires_grad = False
        log_duration_targets = torch.log(duration_targets + 1)
        pdur_loss = F.mse_loss(log_duration_predictions, log_duration_targets)
        loss["pdur"] = pdur_loss
        # calculate words duration loss
        wdur_loss = torch.zeros(1).to(log_duration_predictions.device)
        if self.dur_loss_lambda["wdur"] > 0:
            word_duration_predictions = phone2words(duration_predictions, word_boundaries)
            word_duration_targets = phone2words(duration_targets, word_boundaries)
            wdur_loss = F.mse_loss(
                torch.log(word_duration_predictions + 1), 
                torch.log(word_duration_targets + 1), 
                reduction="none"
            )
            word_nonpadding = (word_duration_predictions > 0).float()
            wdur_loss = (wdur_loss * word_nonpadding).sum() / word_nonpadding.sum()
        loss["wdur"] = wdur_loss
        # calculate sentences duration loss
        sdur_loss = torch.zeros(1).to(log_duration_predictions.device)
        if self.dur_loss_lambda["sdur"] > 0:
            sentence_duration_predictions = duration_predictions.sum(-1)
            sentence_duration_targets = duration_targets.sum(-1)
            sdur_loss = F.mse_loss(
                torch.log(sentence_duration_predictions + 1), 
                torch.log(sentence_duration_targets + 1), 
                reduction="mean"
            )
            sdur_loss = sdur_loss.mean()
        loss["sdur"] = sdur_loss

        return loss

    def build_align_losses(self, attn_matrix, input_lens, output_lens, step) -> dict:
        attn_soft, attn_hard, _, attn_logprob = attn_matrix

        loss = {}
        loss["ctc"] = self.sum_loss(attn_logprob=attn_logprob, in_lens=input_lens, out_lens=output_lens)
        if step < self.binarization_loss_enable_steps:
            bin_loss_weight = 0.
        else:
            bin_loss_weight = min((step - self.binarization_loss_enable_steps) / self.binarization_loss_warmup_steps, 1.0) * 1.0
        loss["bin"] = self.bin_loss(hard_attention=attn_hard, soft_attention=attn_soft) * bin_loss_weight

        return loss

    def build_pitch_losses(self, pitch_predictions, pitch_targets, masks) -> dict:

        loss = {}
        if self.use_uv is True:
            for _, pitch_target in pitch_targets.items():
                if pitch_target is not None:
                    pitch_target.requires_grad = False
            f0_targets = pitch_targets["f0"]
            uv_targets = pitch_targets["uv"]
            nonpadding = masks.float()
            
            uv_predictions = pitch_predictions[..., 1]
            uv_loss = F.binary_cross_entropy_with_logits(
                uv_predictions, uv_targets, reduction="none"
            )
            loss["uv"] = (uv_loss * nonpadding).sum() / nonpadding.sum()
            nonpadding = nonpadding * (uv_targets == 0).float()

            f0_predictions = pitch_predictions[:, :, 0]
            f0_loss = F.mse_loss(f0_predictions, f0_targets, reduction="none")
            loss["f0"] = (f0_loss * nonpadding).sum() / nonpadding.sum()
        else:
            pitch_targets.requires_grad = False
            if self.pitch_feature_level == "phoneme_level":
                pitch_predictions = pitch_predictions.masked_select(masks)
                pitch_targets = pitch_targets.masked_select(masks)
            if self.pitch_feature_level == "frame_level":
                pitch_predictions = pitch_predictions.masked_select(masks)
                pitch_targets = pitch_targets.masked_select(masks)
            loss["pitch"] = F.mse_loss(pitch_predictions, pitch_targets)

        return loss

    def build_energy_losses(self, energy_predictions, energy_targets, masks) -> dict:

        loss = {}
        energy_targets.requires_grad = False
        energy_predictions = energy_predictions.masked_select(masks)
        energy_targets = energy_targets.masked_select(masks)
        loss["energy"] = F.mse_loss(energy_predictions, energy_targets)

        return loss

    def build_melspecs_losses(self, mel_predictions, mel_targets, masks, postnet_mel_predictions=None) -> dict:
        
        loss = {}
        mel_targets.requires_grad = False
        mel_predictions = mel_predictions.masked_select(masks.unsqueeze(-1))
        mel_targets = mel_targets.masked_select(masks.unsqueeze(-1))
        loss["mel"] = F.l1_loss(mel_predictions, mel_targets)
        # calculate mel-specs after postnet if exist
        if postnet_mel_predictions is not None:
            postnet_mel_predictions = postnet_mel_predictions.masked_select(masks.unsqueeze(-1))
        loss["postnet"] = F.l1_loss(postnet_mel_predictions, mel_targets)

        return loss

    def forward(self, predictions, targets, step=None):
        mel_predictions, postnet_mel_predictions, \
            log_duration_predictions, pitch_predictions, energy_predictions, \
            src_lens, src_masks, mel_lens, mel_masks, attn_outs = predictions
        txt_inputs, word_boundaries, mel_targets, pitch_targets, energy_targets= targets

        src_masks = ~src_masks
        mel_masks = ~mel_masks

        mel_targets = mel_targets[:, : mel_masks.shape[1], :]
        mel_masks = mel_masks[:, :mel_masks.shape[1]]

        duration_loss, align_loss, pitch_loss, energy_loss = None, None, None, None
        if step is not None:
            # initilize duration loss (with 3 target: phonemes, words and sentences)
            duration_rounded = attn_outs[2]
            duration_loss = self.build_duration_losses(
                src_inputs=txt_inputs,
                log_duration_predictions=log_duration_predictions,
                duration_targets=duration_rounded,
                word_boundaries=word_boundaries,
                masks=src_masks
            )
            # initilize self-align loss
            align_loss = self.build_align_losses(
                attn_matrix=attn_outs,
                input_lens=src_lens,
                output_lens=mel_lens,
                step=step
            )
            # initilize pitch loss
            pitch_loss = self.build_pitch_losses(
                pitch_predictions=pitch_predictions,
                pitch_targets=pitch_targets,
                masks=src_masks if self.pitch_feature_level == "phoneme_level" else mel_masks
            )
            # initilize energy loss
            energy_loss = self.build_energy_losses(
                energy_predictions=energy_predictions,
                energy_targets=energy_targets,
                masks=src_masks if self.pitch_feature_level == "phoneme_level" else mel_masks
            )
        # initilize mel-spectrograms loss
        mel_loss = self.build_melspecs_losses(
            mel_predictions=mel_predictions,
            mel_targets=mel_targets,
            masks=mel_masks,
            postnet_mel_predictions=postnet_mel_predictions
        )

        return (mel_loss, duration_loss, pitch_loss, energy_loss, align_loss)


class ForwardSumLoss(nn.Module):
    def __init__(self, blank_logprob=-1):
        super().__init__()
        self.log_softmax = nn.LogSoftmax(dim=3)
        self.ctc_loss = nn.CTCLoss(zero_infinity=True)
        self.blank_logprob = blank_logprob

    def forward(self, attn_logprob, in_lens, out_lens):
        key_lens = in_lens
        query_lens = out_lens
        attn_logprob_padded = F.pad(input=attn_logprob, pad=(1, 0), value=self.blank_logprob)

        total_loss = 0.0
        for bid in range(attn_logprob.shape[0]):
            target_seq = torch.arange(1, key_lens[bid] + 1).unsqueeze(0)
            curr_logprob = attn_logprob_padded[bid].permute(1, 0, 2)[: query_lens[bid], :, : key_lens[bid] + 1]

            curr_logprob = self.log_softmax(curr_logprob[None])[0]
            loss = self.ctc_loss(
                curr_logprob,
                target_seq,
                input_lengths=query_lens[bid : bid + 1],
                target_lengths=key_lens[bid : bid + 1],
            )
            total_loss += loss

        total_loss /= attn_logprob.shape[0]

        return total_loss


class BinLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, hard_attention, soft_attention):
        log_sum = torch.log(torch.clamp(soft_attention[hard_attention == 1], min=1e-12)).sum()

        return -log_sum / hard_attention.sum()
