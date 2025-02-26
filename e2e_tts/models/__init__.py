import torch
import numpy as np

from .acoustic.supervised_fastspeech2 import SupervisedFastSpeech2, SupervisedFastSpeech2Loss
from .acoustic.unsupervised_fastspeech2 import UnsupervisedFastSpeech2, UnsupervisedFastSpeech2Loss
from .vocoder import HifiGan, iSTFT, MultiScaleDiscriminator, MultiPeriodDiscriminator, generator_loss, discriminator_loss, feature_loss


def show_params(nnet):
    print("=" * 40, "Model Parameters", "=" * 40)
    num_params = 0
    for module_name, m in nnet.named_modules():
        if module_name == '':
            for name, params in m.named_parameters():
                print(name, params.size())
                i = 1
                for j in params.size():
                    i = i * j
                num_params += i

    print('[*] Parameter Size: {}'.format(num_params))
    print("=" * 100)


def show_model(nnet):
    print("=" * 40, "Model Structures", "=" * 40)
    for module_name, m in nnet.named_modules():
        if module_name == '':
            print(m)

    print("=" * 100)


def get_param_num(model):
    num_param = sum(param.numel() for param in model.parameters())

    return num_param


class ScheduledOptim:
    """ A simple wrapper class for learning rate scheduling """

    def __init__(self,
                 model: torch.nn.Module,
                 optimize_config: dict,
                 encoder_hidden: int,
                 current_step: int
                 ) -> None:
        self._optimizer = torch.optim.Adam(
            model.parameters(),
            lr=optimize_config["learning_rate"],
            betas=optimize_config["betas"],
            eps=optimize_config["eps"],
            weight_decay=optimize_config["weight_decay"],
        )
        self.n_warmup_steps = optimize_config["warm_up_step"]
        self.anneal_steps = optimize_config["anneal_steps"]
        self.anneal_rate = optimize_config["anneal_rate"]
        self.current_step = current_step
        self.init_lr = np.power(encoder_hidden, -0.5)

    def step_and_update_lr(self):
        self._update_learning_rate()
        self._optimizer.step()

    def zero_grad(self):
        self._optimizer.zero_grad()

    def load_state_dict(self, path):
        self._optimizer.load_state_dict(path)

    def _get_lr_scale(self):
        lr = np.min(
            [
                np.power(self.current_step, -0.5),
                np.power(self.n_warmup_steps, -1.5) * self.current_step,
            ]
        )
        for s in self.anneal_steps:
            if self.current_step > s:
                lr = lr * self.anneal_rate
        return lr

    def _update_learning_rate(self):
        """ Learning rate scheduling per step """
        self.current_step += 1
        lr = self.init_lr * self._get_lr_scale()

        for param_group in self._optimizer.param_groups:
            param_group["lr"] = lr
