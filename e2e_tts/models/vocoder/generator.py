import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import Conv1d, ConvTranspose1d
from torch.nn.utils import weight_norm, remove_weight_norm

from .layers import *
from .function import init_weights
LRELU_SLOPE = 0.1


class HifiGan(nn.Module):
    def __init__(self, config: dict) -> None:
        super(HifiGan, self).__init__()
        self.num_kernels = len(config['resblock_kernel_sizes'])
        self.num_upsamples = len(config['upsample_rates'])
        self.conv_pre = weight_norm(Conv1d(80, config['upsample_initial_channel'], 7, 1, padding=3))
        resblock = ResBlock1 if config['resblock'] == 1 else ResBlock2

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(config['upsample_rates'], config['upsample_kernel_sizes'])):
            self.ups.append(weight_norm(ConvTranspose1d(config['upsample_initial_channel'] // (2 ** i),
                                                        config['upsample_initial_channel'] // (2 ** (i + 1)), 
                                                        k, u, padding=(k - u) // 2)))

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = config['upsample_initial_channel'] // (2 ** (i + 1))
            for j, (k, d) in enumerate(zip(config['resblock_kernel_sizes'], config['resblock_dilation_sizes'])):
                self.resblocks.append(resblock(ch, k, d))

        self.conv_post = weight_norm(Conv1d(ch, 1, 7, 1, padding=3))
        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)

    def forward(self, x):
        x = self.conv_pre(x)
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)

        return x

    def remove_weight_norm(self):
        print('Removing weight norm...')
        for _layer in self.ups:
            remove_weight_norm(_layer)
        for _layer in self.resblocks:
            _layer.remove_weight_norm()
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)


class iSTFT(nn.Module):
    def __init__(self, config: dict) -> None:
        super(iSTFT, self).__init__()
        self.num_kernels = len(config['resblock_kernel_sizes'])
        self.num_upsamples = len(config['upsample_rates'])
        self.conv_pre = weight_norm(Conv1d(80, config['upsample_initial_channel'], 7, 1, padding=3))
        resblock = ResBlock1 if config['resblock'] == '1' else ResBlock2

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(config['upsample_rates'], config['upsample_kernel_sizes'])):
            self.ups.append(weight_norm(ConvTranspose1d(config['upsample_initial_channel'] // (2 ** i),
                                                        config['upsample_initial_channel'] // (2 ** (i + 1)), 
                                                        k, u, padding=(k - u) // 2)))

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = config['upsample_initial_channel'] // (2 ** (i + 1))
            for j, (k, d) in enumerate(zip(config['resblock_kernel_sizes'], config['resblock_dilation_sizes'])):
                self.resblocks.append(resblock(ch, k, d))

        self.post_n_fft = config["gen_istft_n_fft"]
        self.conv_post = weight_norm(Conv1d(ch, self.post_n_fft + 2, 7, 1, padding=3))
        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)
        self.reflection_pad = nn.ReflectionPad1d((1, 0))

    def forward(self, x):
        x = self.conv_pre(x)
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.reflection_pad(x)
        x = self.conv_post(x)
        spec = torch.exp(x[:, :self.post_n_fft // 2 + 1, :])
        phase = torch.sin(x[:, self.post_n_fft // 2 + 1:, :])

        return spec, phase

    def remove_weight_norm(self):
        print('Removing weight norm...')
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)
