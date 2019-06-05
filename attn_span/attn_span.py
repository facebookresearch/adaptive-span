#!/usr/bin/env python3

import math

import torch
import torch.nn.functional as F

from models.adaptive_mask import AdaptiveMask

# Adaptive attention span for Transformer


def add_span_mask(attn_lim, attn_span_len, attn_span_init, nb_heads, model):
    model.span_mask = AdaptiveMask(attn_lim,
                                   attn_span_len,
                                   init_ratio=attn_span_init,
                                   shape=(nb_heads, 1, 1),
                                   sum_normalize=True)


# compute how long the attention span should be
def compute(args, model):
    L = args.attn_lim
    skip_len = min(L-1 , L - model.span_mask.get_max_size())
    skip_len = math.floor(skip_len / 64) * 64 # for better memory caching
    return skip_len


# determine how long the cache should be
def get_cache_size(model):
    args = model.args
    if args.attn_span and args.attn_span_cache:
        skip_len = compute(args, model)
        # give a buffer of 64 steps
        return min(args.attn_lim, args.attn_lim - skip_len + 64)
    else:
        return args.attn_lim


# crop out unnecessary computation
def crop(skip_len, key, value, key_pe, attn_lim, mem_size):
    # TODO: where is mem_sz defined?
    cache_size = key.size(1) - mem_size
    skip_len2 = skip_len - (attn_lim - cache_size)
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


# mask attention with the right span
def mask(args, attn, model, skip_len):
    B = attn.size(0)
    M = attn.size(1)
    attn = attn.reshape(B // args.nheads, args.nheads, M, -1)
    attn = model.span_mask(attn, skip_len=skip_len)
    attn = attn.view(B, M, -1)
    return attn


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
