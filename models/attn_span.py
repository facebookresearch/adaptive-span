from __future__ import print_function
from argparse import Namespace
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.adaptive_mask import AdaptiveMask

# Adaptive attention span for Transformer

def add_args(parser):
    parser.add_argument('--attn-span', action='store_true', default=False,
                        help='learn attention span')
    parser.add_argument('--attn-span-loss', type=float, default=0,
                        help='learn attention span')
    parser.add_argument('--attn-span-len', type=float, default=32,
                        help='learn attention span')
    parser.add_argument('--attn-span-init', type=float, default=0,
                        help='initial attention span ratio value')
    parser.add_argument('--attn-span-cache', action='store_true', default=False,
                        help='change cache size')


def init(args, model):
    model.span_mask = AdaptiveMask(args.attn_lim, args.attn_span_len, init_ratio=args.attn_span_init, shape=(args.nheads, 1, 1), sum_normalize=True)


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
def crop(args, skip_len, key, value, key_pe):
    cache_size = key.size(1) - args.block_sz
    skip_len2 = skip_len - (args.attn_lim - cache_size)
    if skip_len2 > 0:
        key = key[:, skip_len2:, :]
        value = value[:, skip_len2:, :]
    elif skip_len2 < 0:
        print('warning: cache is too short. cache_size={} skip_len={}'.format(cache_size, skip_len))
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
def loss(args, model, stat):
    loss = 0
    for l in model.module.layers:
        loss = loss + l.attn.attn.span_mask.size_ratio.mean() * args.attn_span_loss * args.attn_lim
    return loss


def param_clamp(args, model):
    for l in model.module.layers:
        l.attn.attn.span_mask.size_ratio.data.clamp_(0, 1)


def log(args, model, logger, stat_train):
    x = []
    for i, l in enumerate(model.module.layers):
        span = l.attn.attn.span_mask.size_ratio.view(-1)
        x.append(span)
        span = span.mean().item()
    x = torch.cat(x,dim=0)
    logger.log('span_avg', x.mean().item())
    logger.log('span_max', x.max().item())
    if args.plot:
        logger.vis.line(x, win='span_latest', opts={'title': 'span_latest'})
