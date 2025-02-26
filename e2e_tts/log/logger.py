import numpy as np
from torch.utils.tensorboard import SummaryWriter
from .utils import *


class text_colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class e2e_logger(SummaryWriter):
    def __init__(self, logdir, tag, sampling_rate=22050, max_wav_value=32768.0):
        super(e2e_logger, self).__init__(logdir)

        self.sampling_rate = sampling_rate
        self.max_wav_value = max_wav_value
        self.tag = tag

    def log(self, step, losses=None, lr=None, audio: list=None):
        if losses is not None:
            self.add_scalar(f"{self.tag}/total_loss", losses[0], step)

            self.add_scalar(f"{self.tag}/loss_gen_all", losses[1], step)
            self.add_scalar(f"{self.tag}/loss_dis_all", losses[2], step)
            self.add_scalar(f"{self.tag}/loss_var_all", losses[3], step)

            self.add_scalar(f"{self.tag}/loss_disc_s", losses[4], step)
            self.add_scalar(f"{self.tag}/loss_disc_f", losses[5], step)
            self.add_scalar(f"{self.tag}/loss_gen_s", losses[6], step)
            self.add_scalar(f"{self.tag}/loss_gen_f", losses[7], step)

            self.add_scalar(f"{self.tag}/loss_fm_s", losses[8], step)
            self.add_scalar(f"{self.tag}/loss_fm_f", losses[9], step)
            self.add_scalar(f"{self.tag}/loss_mel", losses[10], step)

            self.add_scalar(f"{self.tag}/loss_duration", losses[11], step)
            self.add_scalar(f"{self.tag}/loss_pitch", losses[12], step)
            self.add_scalar(f"{self.tag}/loss_energy", losses[13], step)
            
        if lr is not None:
            self.add_scalar(f"{self.tag}/learning_rate", lr, step)

        if audio is not None:
            # mel = self.stft(audio).squeeze(0)
            gt_audio, gen_audio = audio
            gt_audio = (gt_audio * self.max_wav_value).squeeze(0).detach().cpu().numpy().astype('int16')
            self.add_audio(f"{self.tag}/ground_truth_wav", gt_audio / max(abs(gt_audio)), step, sample_rate=self.sampling_rate)
            gen_audio = (gen_audio * self.max_wav_value).squeeze(0).detach().cpu().numpy().astype('int16')
            self.add_audio(f"{self.tag}/generated_wav", gen_audio / max(abs(gen_audio)), step, sample_rate=self.sampling_rate)


class acoustic_logger(SummaryWriter):
    def __init__(self, logdir):
        super(acoustic_logger, self).__init__(logdir)

    def log(self, losses, step, state_dict=None, lr=None):
        if isinstance(losses, tuple):
            for _loss in losses:
                for k, v in _loss.items():
                    self.add_scalar(f"train/{k}_loss", v, step)
        else:
            for k, v in losses.items():
                self.add_scalar(f"valid/{k}_loss", v, step)

        if lr is not None:
            self.add_scalar(f"train/learning_rate", lr, step)

        if state_dict is not None:
            # plot distribution of parameters
            for tag, value in state_dict.named_parameters():
                tag = tag.replace('.', '/')
                self.add_histogram(tag, value.data.cpu().numpy(), step)
