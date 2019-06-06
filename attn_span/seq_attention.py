#!/usr/bin/env python3

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# TODO: review import statements
from attn_span.adaptive_mask import AdaptiveMask
from attn_span.utils import skew, unskew


class SeqAttention(nn.Module):
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
        nn.Module.__init__(self)
        self.attn_span_enabled = attn_span_enabled
        self.attn_span_cache_enabled = attn_span_cache_enabled
        self.dropout = nn.Dropout(dropout)
        self.nb_heads = nb_heads
        self.head_dim = hidden_size // nb_heads
        self.attn_span_lim = attn_span_lim
        self.attn_span_loss = attn_span_loss
        self.block_size = block_size
        self.span_mask = AdaptiveMask(size=attn_span_lim,
                                      ramp_size=attn_span_len,
                                      init_ratio=attn_span_init,
                                      shape=(nb_heads, 1, 1),
                                      sum_normalize=True)

    def forward(self, query, key, value, key_pe):
        # B = query.size(0)
        H = self.head_dim
        # L = self.attn_span_lim
        # M = self.block_size
        # query = B x M x H
        # key, value = B x (M+L) x H

        # compute attention span
        if self.attn_span_loss > 0:
            skip_len = self._compute_skip_len()
            key, value, key_pe = self._crop(key=key,
                                            value=value,
                                            key_pe=key_pe,
                                            skip_len=skip_len)

        # compute attention from context
        # B x M (dest) x (M+L) (src)
        attn_cont = torch.matmul(query, key.transpose(-1, -2))
        attn_cont = unskew(attn_cont)  # B x M x L

        # compute the effect of position embedding
        attn_pos = torch.matmul(query, key_pe)  # B x M x L_pos
        attn = attn_cont + attn_pos

        attn = attn / math.sqrt(H)  # B x M X L_pos
        attn = F.softmax(attn, dim=-1)

        if self.attn_span_loss > 0:
            attn = self._mask(attn=attn, skip_len=skip_len)
        attn = self.dropout(attn)  # B x M X L_pos

        attn_cont = skew(attn, 0)  # B x M X (L+M)
        out = torch.matmul(attn_cont, value)  # B x M x H

        return out

    def _compute_skip_len(self):
        """compute how long the attention span should be"""
        L = self.attn_span_lim
        skip_len = min(L - 1, L - self.span_mask.get_max_size())
        skip_len = math.floor(skip_len / 64) * 64  # for better memory caching
        return skip_len

    def _crop(self, key, value, key_pe, skip_len=None):
        """crop out unnecessary computation"""
        if skip_len is None:
            skip_len = self._compute_skip_len()
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

    def _mask(self, attn, skip_len=None):
        """mask attention with the right span"""
        if skip_len is None:
            skip_len = self.attn_span_compute_skip_len()
        B = attn.size(0)
        M = attn.size(1)
        attn = attn.reshape(B // self.nb_heads, self.nb_heads, M, -1)
        attn = self.span_mask(attn, skip_len=skip_len)
        attn = attn.view(B, M, -1)
        return attn

    def _get_cache_size(self):
        """determine how long the cache should be"""
        if self.attn_span_enabled and self.attn_span_cache_enabled:
            skip_len = self.attn_span_compute_skip_len()
            # give a buffer of 64 steps
            if skip_len > 64:
                return self.attn_span_lim - skip_len + 64
            return self.attn_span_lim,
        else:
            return self.attn_span_lim

    def compute_extra_loss(self):
        return (self.attn_span_loss *
                self.attn_span_lim *
                self.span_mask.size_ratio.mean())

    def clamp_param(self):
        self.span_mask.size_ratio.data.clamp_(0, 1)
