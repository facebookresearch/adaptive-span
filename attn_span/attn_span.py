#!/usr/bin/env python3

import math

import torch
import torch.nn.functional as F

from models.adaptive_mask import AdaptiveMask


# TODO: create mixin to be injected to SeqAttention
class AttnSpanMixin:
    """Adaptive attention span for Transformer"""
    def __init__(self,
                 attn_lim,
                 block_size,
                 attn_span_len,
                 attn_span_loss,
                 attn_span_init,
                 nb_heads):
        self.attn_lim = attn_lim
        self.attn_span_loss = attn_span_loss
        self.block_size = block_size
        self.span_mask = AdaptiveMask(size=attn_lim,
                                      ramp_size=attn_span_len,
                                      init_ratio=attn_span_init,
                                      shape=(nb_heads, 1, 1),
                                      sum_normalize=True)

    def attn_span_compute_skip_len(self):
        """compute how long the attention span should be"""
        L = self.attn_lim
        skip_len = min(L - 1, L - self.span_mask.get_max_size())
        skip_len = math.floor(skip_len / 64) * 64  # for better memory caching
        return skip_len

    def attn_span_get_cache_size(self):
        """determine how long the cache should be"""
        # TODO: what are model.args.attn_span and model.args.attn_span_cache?
        args = self.args
        if args.attn_span and args.attn_span_cache:
            skip_len = self.attn_span_compute_skip_len()
            # give a buffer of 64 steps
            if skip_len > 64:
                return self.attn_lim - skip_len + 64
            return self.attn_lim,
        else:
            return args.attn_lim

    def attn_span_crop(self, key, value, key_pe):
        """crop out unnecessary computation"""
        skip_len = self.attn_span_compute_skip_len()
        cache_size = key.size(1) - self.block_size
        skip_len2 = skip_len - (self.attn_lim - cache_size)
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

    def attn_span_mask(self, attn):
        """mask attention with the right span"""
        skip_len = self.attn_span_compute_skip_len()
        B = attn.size(0)
        M = attn.size(1)
        attn = attn.reshape(B // self.nb_heads, self.nb_heads, M, -1)
        attn = self.span_mask(attn, skip_len=skip_len)
        attn = attn.view(B, M, -1)
        return attn


# TODO: the next functions are not specific to AttnSpanMixin...

# compute the loss
def loss(model, attn_span_loss, attn_lim):
    loss_factor = attn_span_loss * attn_lim
    return loss_factor * sum(l.attn.attn.span_mask.size_ratio.mean()
                             for l in model.module.layers)


def param_clamp(model):
    for l in model.module.layers:
        l.attn.attn.span_mask.size_ratio.data.clamp_(0, 1)


# TODO: plot enabled should be in the plotter object not in plot
def plot(plot_enabled, model, plotter, stat_train):
    x = []
    for i, l in enumerate(model.module.layers):
        span = l.attn.attn.span_mask.size_ratio.view(-1)
        x.append(span)
        span = span.mean().item()
    x = torch.cat(x, dim=0)
    plotter.log('span_avg', x.mean().item())
    plotter.log('span_max', x.max().item())
    if plot_enabled:
        plotter.vis.line(x, win='span_latest', opts={'title': 'span_latest'})
