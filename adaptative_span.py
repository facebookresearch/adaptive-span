#!/usr/bin/env python3

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveMask(nn.Module):
    def __init__(self,
                 size,
                 ramp_size,
                 init_ratio=0,
                 shape=(1,),
                 sum_normalize=False):
        nn.Module.__init__(self)
        self.size = size
        self.ramp_size = ramp_size
        self.size_ratio = nn.Parameter(torch.zeros(*shape) + init_ratio)
        self.sum_normalize = sum_normalize
        mask_template = torch.linspace(1 - size, 0, steps=size)
        self.register_buffer('mask_template', mask_template)

    def forward(self, x, skip_len=0):
        mask = self.mask_template + self.size_ratio * self.size
        mask = mask / self.ramp_size + 1
        mask = mask.clamp(0, 1)
        if skip_len > 0:
            mask = mask[:, :, skip_len:]
        x = x * mask
        # TODO: move to AdaptiveSpan
        if self.sum_normalize:
            x = x / (x.sum(-1, keepdim=True) + 1e-8)  # normalize so sum is 1
        return x

    def get_max_size(self):
        max_size = math.ceil(
            self.ramp_size + self.size_ratio.max().item() * self.size)
        max_size = min(self.size, max_size)
        return max_size if max_size > 0 else 0

    def clamp_param(self):
        self.size_ratio.data.clamp_(0, 1)


class AdaptiveSpan(AdaptiveMask):
    """Adaptive attention span for Transformer"""
    def __init__(self,
                 attn_span_enabled,
                 attn_span_cache_enabled,
                 dropout,
                 hidden_size,
                 nb_heads,
                 attn_span_lim,
                 attn_span_loss,
                 block_size,
                 attn_span_len,
                 attn_span_init,
                 *args, **kwargs):
        AdaptiveMask.__init__(self,
                              size=attn_span_lim,
                              ramp_size=attn_span_len,
                              init_ratio=attn_span_init,
                              shape=(nb_heads, 1, 1),
                              sum_normalize=True)
        self.attn_span_enabled = attn_span_enabled
        self.attn_span_cache_enabled = attn_span_cache_enabled
        self.dropout = nn.Dropout(dropout)
        self.nb_heads = nb_heads
        self.head_dim = hidden_size // nb_heads
        self.attn_span_lim = attn_span_lim
        self.attn_span_loss = attn_span_loss
        self.block_size = block_size

    def compute_skip_len(self):
        """compute how long the attention span should be"""
        L = self.attn_span_lim
        skip_len = min(L - 1, L - self.get_max_size())
        skip_len = math.floor(skip_len / 64) * 64  # for better memory caching
        return skip_len

    def crop(self, key, value, key_pe, skip_len=None):
        """crop out unnecessary computation"""
        if skip_len is None:
            skip_len = self.compute_skip_len()
        cache_size = key.size(1) - self.block_size
        skip_len2 = skip_len - (self.attn_span_lim - cache_size)
        if skip_len2 > 0:
            key = key[:, skip_len2:, :]
            value = value[:, skip_len2:, :]
        elif skip_len2 < 0:
            # TODO: replace print by proper logger
            print('warning: cache is too short. cache_size={} skip_len={}'.
                  format(cache_size, skip_len))
            key = F.pad(key, [0, 0, -skip_len2, 0])
            value = F.pad(value, [0, 0, -skip_len2, 0])
        if skip_len > 0:
            if key_pe is not None:
                key_pe = key_pe[:, :, skip_len:]
        return key, value, key_pe

    def forward(self, attn, skip_len=None):
        """mask attention with the right span"""
        if skip_len is None:
            skip_len = self.compute_skip_len()
        B = attn.size(0)
        M = attn.size(1)
        attn = attn.reshape(B // self.nb_heads, self.nb_heads, M, -1)
        attn = AdaptiveMask.forward(self, attn, skip_len)
        attn = attn.view(B, M, -1)
        return attn

    def get_cache_size(self):
        """determine how long the cache should be"""
        if self.attn_span_enabled and self.attn_span_cache_enabled:
            skip_len = self.compute_skip_len()
            # give a buffer of 64 steps
            if skip_len > 64:
                return self.attn_span_lim - skip_len + 64
            return self.attn_span_lim
        else:
            return self.attn_span_lim

    def compute_extra_loss(self):
        return (self.attn_span_loss *
                self.attn_span_lim *
                self.size_ratio.mean())
