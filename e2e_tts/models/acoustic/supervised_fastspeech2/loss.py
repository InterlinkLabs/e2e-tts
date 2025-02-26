import torch
import torch.nn as nn
import torch.nn.functional as F


class SupervisedFastSpeech2Loss(nn.Module):
    """ FastSpeech2 Loss """

    def __init__(self, pitch_feature_level, energy_feature_level, use_uv):
        super(SupervisedFastSpeech2Loss, self).__init__()
        self.use_uv = use_uv

        self.pitch_feature_level = pitch_feature_level
        self.energy_feature_level = energy_feature_level


    def build_duration_losses(self, log_duration_predictions, duration_targets, masks) -> dict:
        loss = {}
        # calculate duration loss (log_d_predictions, duration_targets)
        log_duration_targets = torch.log(duration_targets.float() + 1)
        log_duration_targets.requires_grad = False
        log_duration_predictions = log_duration_predictions.masked_select(masks)
        log_duration_targets = log_duration_targets.masked_select(masks)

        loss["dur"] = F.mse_loss(log_duration_predictions, log_duration_targets)
        return loss

    def build_energy_losses(self, energy_predictions, energy_targets, masks) -> dict:
        loss = {}
        energy_targets.requires_grad = False
        energy_predictions = energy_predictions.masked_select(masks)
        energy_targets = energy_targets.masked_select(masks)
        loss["energy"] = F.mse_loss(energy_predictions, energy_targets)

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

    def forward(self, model_output, targets, step):
        mel_predictions, postnet_mel_predictions, \
            log_duration_predictions, _, pitch_predictions, energy_predictions,\
            src_masks, mel_masks = model_output
        mel_targets, duration_targets, pitch_targets, energy_targets = targets
        src_masks = ~src_masks
        mel_masks = ~mel_masks

        mel_targets = mel_targets[:, : mel_masks.shape[1], :]
        mel_masks = mel_masks[:, :mel_masks.shape[1]]

        # initilize duration loss
        duration_loss = self.build_duration_losses(
            log_duration_predictions=log_duration_predictions,
            duration_targets=duration_targets,
            masks=src_masks
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

        return (mel_loss, duration_loss, pitch_loss, energy_loss)
