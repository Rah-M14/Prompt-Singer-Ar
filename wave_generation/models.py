# Copyright (c) 2022 NVIDIA CORPORATION. 
#   Licensed under the MIT license.

# Adapted from https://github.com/jik876/hifi-gan under the MIT license.
#   LICENSE is in incl_licenses directory.
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Conv1d, ConvTranspose1d, Conv2d
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm

import activations
from utils import init_weights, get_padding
from alias_free_torch import *

LRELU_SLOPE = 0.1


class AMPBlock1(torch.nn.Module):
    def __init__(self, h, channels, kernel_size=3, dilation=(1, 3, 5), activation=None):
        super(AMPBlock1, self).__init__()
        self.h = h

        self.convs1 = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0],
                               padding=get_padding(kernel_size, dilation[0]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1],
                               padding=get_padding(kernel_size, dilation[1]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[2],
                               padding=get_padding(kernel_size, dilation[2])))
        ])
        self.convs1.apply(init_weights)

        self.convs2 = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1)))
        ])
        self.convs2.apply(init_weights)

        self.num_layers = len(self.convs1) + len(self.convs2) # total number of conv layers

        if activation == 'snake': # periodic nonlinearity with snake function and anti-aliasing
            self.activations = nn.ModuleList([
                Activation1d(
                    activation=activations.Snake(channels, alpha_logscale=h.snake_logscale))
                for _ in range(self.num_layers)
            ])
        elif activation == 'snakebeta': # periodic nonlinearity with snakebeta function and anti-aliasing
            self.activations = nn.ModuleList([
                Activation1d(
                    activation=activations.SnakeBeta(channels, alpha_logscale=h.snake_logscale))
                 for _ in range(self.num_layers)
            ])
        else:
            raise NotImplementedError("activation incorrectly specified. check the config file and look for 'activation'.")

    def forward(self, x):
        acts1, acts2 = self.activations[::2], self.activations[1::2]
        for c1, c2, a1, a2 in zip(self.convs1, self.convs2, acts1, acts2):
            xt = a1(x)
            xt = c1(xt)
            xt = a2(xt)
            xt = c2(xt)
            x = xt + x

        return x

    def remove_weight_norm(self):
        for l in self.convs1:
            remove_weight_norm(l)
        for l in self.convs2:
            remove_weight_norm(l)


class AMPBlock2(torch.nn.Module):
    def __init__(self, h, channels, kernel_size=3, dilation=(1, 3), activation=None):
        super(AMPBlock2, self).__init__()
        self.h = h

        self.convs = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0],
                               padding=get_padding(kernel_size, dilation[0]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1],
                               padding=get_padding(kernel_size, dilation[1])))
        ])
        self.convs.apply(init_weights)

        self.num_layers = len(self.convs) # total number of conv layers

        if activation == 'snake': # periodic nonlinearity with snake function and anti-aliasing
            self.activations = nn.ModuleList([
                Activation1d(
                    activation=activations.Snake(channels, alpha_logscale=h.snake_logscale))
                for _ in range(self.num_layers)
            ])
        elif activation == 'snakebeta': # periodic nonlinearity with snakebeta function and anti-aliasing
            self.activations = nn.ModuleList([
                Activation1d(
                    activation=activations.SnakeBeta(channels, alpha_logscale=h.snake_logscale))
                 for _ in range(self.num_layers)
            ])
        else:
            raise NotImplementedError("activation incorrectly specified. check the config file and look for 'activation'.")

    def forward(self, x):
        for c, a in zip (self.convs, self.activations):
            xt = a(x)
            xt = c(xt)
            x = xt + x

        return x

    def remove_weight_norm(self):
        for l in self.convs:
            remove_weight_norm(l)


class BigVGAN(torch.nn.Module):
    # this is our main BigVGAN model. Applies anti-aliased periodic activation for resblocks.
    def __init__(self, h):
        super(BigVGAN, self).__init__()
        self.h = h

        self.num_kernels = len(h.resblock_kernel_sizes)
        self.num_upsamples = len(h.upsample_rates)

        # pre conv
        self.conv_pre = weight_norm(Conv1d(h.num_mels, h.upsample_initial_channel, 7, 1, padding=3))

        # define which AMPBlock to use. BigVGAN uses AMPBlock1 as default
        resblock = AMPBlock1 if h.resblock == '1' else AMPBlock2

        # transposed conv-based upsamplers. does not apply anti-aliasing
        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(h.upsample_rates, h.upsample_kernel_sizes)):
            self.ups.append(nn.ModuleList([
                weight_norm(ConvTranspose1d(h.upsample_initial_channel // (2 ** i),
                                            h.upsample_initial_channel // (2 ** (i + 1)),
                                            k, u, padding=(k - u) // 2))
            ]))

        # residual blocks using anti-aliased multi-periodicity composition modules (AMP)
        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = h.upsample_initial_channel // (2 ** (i + 1))
            for j, (k, d) in enumerate(zip(h.resblock_kernel_sizes, h.resblock_dilation_sizes)):
                self.resblocks.append(resblock(h, ch, k, d, activation=h.activation))

        # post conv
        if h.activation == "snake": # periodic nonlinearity with snake function and anti-aliasing
            activation_post = activations.Snake(ch, alpha_logscale=h.snake_logscale)
            self.activation_post = Activation1d(activation=activation_post)
        elif h.activation == "snakebeta": # periodic nonlinearity with snakebeta function and anti-aliasing
            activation_post = activations.SnakeBeta(ch, alpha_logscale=h.snake_logscale)
            self.activation_post = Activation1d(activation=activation_post)
        else:
            raise NotImplementedError("activation incorrectly specified. check the config file and look for 'activation'.")

        self.conv_post = weight_norm(Conv1d(ch, 1, 7, 1, padding=3))

        # weight initialization
        for i in range(len(self.ups)):
            self.ups[i].apply(init_weights)
        self.conv_post.apply(init_weights)

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        return n_params

    
    def forward(self, x):
        # pre conv
        x = self.conv_pre(x)

        for i in range(self.num_upsamples):
            # upsampling
            for i_up in range(len(self.ups[i])):
                x = self.ups[i][i_up](x)
            # AMP blocks
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels

        # post conv
        x = self.activation_post(x)
        x = self.conv_post(x)
        x = torch.tanh(x)

        return x

    def remove_weight_norm(self):
        print('Removing weight norm...')
        #self.get_num_params()
        for l in self.ups:
            for l_i in l:
                remove_weight_norm(l_i)
        for l in self.resblocks:
            l.remove_weight_norm()
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)


class CodeBigVGAN(BigVGAN):
    def __init__(self, h):
        super().__init__(h)
        
        self.code_num = h.code_num
        self.codebook_num = h.codebook_num
        
        self.code_embed = nn.Embedding(self.code_num * self.codebook_num, h.code_emb_dim)
        self.conv_pre = weight_norm(Conv1d(h.code_emb_dim * h.codebook_num, h.upsample_initial_channel, 7, 1, padding=3))
        self.get_num_params()
        

    def forward(self, x):
        assert x.shape[1] == self.codebook_num
        offsets = self.code_num * torch.arange(self.codebook_num)
        offsets = offsets.unsqueeze(0).unsqueeze(-1).to(x.device)
        x = x + offsets
        x = self.code_embed(x)
        x = x.transpose(2, 3)
        x = x.flatten(start_dim=1, end_dim=2)
        x = super().forward(x)

        return x


class DiscriminatorP(torch.nn.Module):
    def __init__(self, h, period, kernel_size=5, stride=3, use_spectral_norm=False):
        super(DiscriminatorP, self).__init__()
        self.period = period
        self.d_mult = h.discriminator_channel_mult
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = nn.ModuleList([
            norm_f(Conv2d(1, int(32*self.d_mult), (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(Conv2d(int(32*self.d_mult), int(128*self.d_mult), (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(Conv2d(int(128*self.d_mult), int(512*self.d_mult), (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(Conv2d(int(512*self.d_mult), int(1024*self.d_mult), (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(Conv2d(int(1024*self.d_mult), int(1024*self.d_mult), (kernel_size, 1), 1, padding=(2, 0))),
        ])
        self.conv_post = norm_f(Conv2d(int(1024*self.d_mult), 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x):
        fmap = []

        # 1d to 2d
        b, c, t = x.shape
        if t % self.period != 0: # pad first
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class MultiPeriodDiscriminator(torch.nn.Module):
    def __init__(self, h):
        super(MultiPeriodDiscriminator, self).__init__()
        self.mpd_reshapes = h.mpd_reshapes
        print("mpd_reshapes: {}".format(self.mpd_reshapes))
        discriminators = [DiscriminatorP(h, rs, use_spectral_norm=h.use_spectral_norm) for rs in self.mpd_reshapes]
        self.discriminators = nn.ModuleList(discriminators)

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class DiscriminatorR(nn.Module):
    def __init__(self, cfg, resolution):
        super().__init__()

        self.resolution = resolution
        assert len(self.resolution) == 3, \
            "MRD layer requires list with len=3, got {}".format(self.resolution)
        self.lrelu_slope = LRELU_SLOPE

        norm_f = weight_norm if cfg.use_spectral_norm == False else spectral_norm
        if hasattr(cfg, "mrd_use_spectral_norm"):
            print("INFO: overriding MRD use_spectral_norm as {}".format(cfg.mrd_use_spectral_norm))
            norm_f = weight_norm if cfg.mrd_use_spectral_norm == False else spectral_norm
        self.d_mult = cfg.discriminator_channel_mult
        if hasattr(cfg, "mrd_channel_mult"):
            print("INFO: overriding mrd channel multiplier as {}".format(cfg.mrd_channel_mult))
            self.d_mult = cfg.mrd_channel_mult

        self.convs = nn.ModuleList([
            norm_f(nn.Conv2d(1, int(32*self.d_mult), (3, 9), padding=(1, 4))),
            norm_f(nn.Conv2d(int(32*self.d_mult), int(32*self.d_mult), (3, 9), stride=(1, 2), padding=(1, 4))),
            norm_f(nn.Conv2d(int(32*self.d_mult), int(32*self.d_mult), (3, 9), stride=(1, 2), padding=(1, 4))),
            norm_f(nn.Conv2d(int(32*self.d_mult), int(32*self.d_mult), (3, 9), stride=(1, 2), padding=(1, 4))),
            norm_f(nn.Conv2d(int(32*self.d_mult), int(32*self.d_mult), (3, 3), padding=(1, 1))),
        ])
        self.conv_post = norm_f(nn.Conv2d(int(32 * self.d_mult), 1, (3, 3), padding=(1, 1)))

    def forward(self, x):
        fmap = []

        x = self.spectrogram(x)
        x = x.unsqueeze(1)
        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, self.lrelu_slope)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap

    def spectrogram(self, x):
        n_fft, hop_length, win_length = self.resolution
        x = F.pad(x, (int((n_fft - hop_length) / 2), int((n_fft - hop_length) / 2)), mode='reflect')
        x = x.squeeze(1)
        x = torch.stft(x, n_fft=n_fft, hop_length=hop_length, win_length=win_length, center=False, return_complex=True)
        x = torch.view_as_real(x)  # [B, F, TT, 2]
        mag = torch.norm(x, p=2, dim =-1) #[B, F, TT]

        return mag


class MultiResolutionDiscriminator(nn.Module):
    def __init__(self, cfg, debug=False):
        super().__init__()
        self.resolutions = cfg.resolutions
        assert len(self.resolutions) == 3,\
            "MRD requires list of list with len=3, each element having a list with len=3. got {}".\
                format(self.resolutions)
        self.discriminators = nn.ModuleList(
            [DiscriminatorR(cfg, resolution) for resolution in self.resolutions]
        )

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []

        for i, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(x=y)
            y_d_g, fmap_g = d(x=y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


def feature_loss(fmap_r, fmap_g):
    loss = 0
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            loss += torch.mean(torch.abs(rl - gl))

    return loss*2


def discriminator_loss(disc_real_outputs, disc_generated_outputs):
    loss = 0
    r_losses = []
    g_losses = []
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        r_loss = torch.mean((1-dr)**2)
        g_loss = torch.mean(dg**2)
        loss += (r_loss + g_loss)
        r_losses.append(r_loss.item())
        g_losses.append(g_loss.item())

    return loss, r_losses, g_losses


def generator_loss(disc_outputs):
    loss = 0
    gen_losses = []
    for dg in disc_outputs:
        l = torch.mean((1-dg)**2)
        gen_losses.append(l)
        loss += l

    return loss, gen_losses
