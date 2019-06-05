#!/usr/bin/env python3

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# TODO: review import statements
from models.utils import skew, unskew
import attn_span.attn_span as attn_span

# B =  _sz
# H = hidden_size
# M = mem_size
# L = attn_lim

# each position will only attent to its previous L positions
# (from the lower layer)
# no self-attention: L positions doesn't include the current step


# TODO: given models share commom fields, create a parent class/abs class/mixin

class SeqAttention(nn.Module):
    def __init__(self,
                 dropout,
                 hidden_size,
                 nb_heads,
                 attn_lim,
                 attn_span_len,
                 attn_span_init,
                 attn_span_loss,
                 mem_size):
        super(SeqAttention, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.head_dim = hidden_size // nb_heads
        self.attn_span_loss = attn_span_loss
        self.attn_lim = attn_lim
        self.mem_size = mem_size
        if self.attn_span_loss > 0:
            attn_span.add_span_mask(attn_lim=attn_lim,
                                    attn_span_len=attn_span_len,
                                    attn_span_init=attn_span_init,
                                    nb_heads=nb_heads,
                                    model=self)

    def forward(self, query, key, value, key_pe):
        # B = query.size(0)
        H = self.head_dim
        # L = self.attn_lim
        # M = self.mem_size
        # query = B x M x H
        # key, value = B x (M+L) x H

        # compute attention span
        if self.attn_span_loss > 0:
            skip_len = attn_span.compute(self.args, self)
            key, value, key_pe = attn_span.crop(skip_len=skip_len,
                                                key=key,
                                                value=value,
                                                key_pe=key_pe,
                                                attn_lim=self.attn_lim,
                                                mem_size=self.mem_size)

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
            attn = attn_span.mask(self.args, attn, self, skip_len)
        attn = self.dropout(attn)  # B x M X L_pos

        attn_cont = skew(attn, 0)  # B x M X (L+M)
        out = torch.matmul(attn_cont, value)  # B x M x H

        return out


class MultiHeadSeqAttention(nn.Module):
    def __init__(self,
                 hidden_size,
                 nb_heads,
                 mem_size,
                 *args, **kwargs):
        super(MultiHeadSeqAttention, self).__init__()
        assert hidden_size % nb_heads == 0
        self.nb_heads = nb_heads
        self.head_dim = hidden_size // nb_heads
        self.attn = SeqAttention(args)
        self.proj_query = nn.Linear(
            hidden_size, self.head_dim * nb_heads, bias=False)
        self.proj_out = nn.Linear(
            self.head_dim * nb_heads, hidden_size, bias=False)
        self.proj_val = nn.Linear(
            hidden_size, self.head_dim * nb_heads, bias=False)
        self.proj_key = nn.Linear(
            hidden_size, self.head_dim * nb_heads, bias=False)
        self.mem_size = mem_size

    def head_reshape(self, x):
        K = self.nb_heads
        D = self.head_dim
        sz = x.size()
        sz = sz[:-1] + (K, D)  # B x (M+L) x K x D
        x = x.view(sz)  # B x (M+L) x K x D
        x = x.transpose(1, 2).contiguous()  # B x K x (M+L) x D
        x = x.view(-1, x.size(-2), x.size(-1))  # B_K x (M+L) x D
        return x

    def forward(self, query, key, value, key_pe):
        B = query.size(0)
        K = self.nb_heads
        D = self.head_dim
        M = self.mem_size

        query = self.proj_query(query)
        query = self.head_reshape(query)
        value = self.proj_val(value)
        value = self.head_reshape(value)
        key = self.proj_key(key)
        key = self.head_reshape(key)

        out = self.attn(query, key, value, key_pe)  # B_K x M x D
        out = out.view(B, K, M, D)  # B x K x M x D
        out = out.transpose(1, 2).contiguous()  # B x M x K x D
        out = out.view(B, M, -1)  # B x M x K_D
        out = self.proj_out(out)
        return out


class FeedForwardLayer(nn.Module):
    def __init__(self,
                 hidden_size,
                 inner_hidden_size,
                 dropout,
                 *args,
                 **kwargs):
        super(FeedForwardLayer, self).__init__()
        self.fc1 = nn.Linear(hidden_size, inner_hidden_size)
        self.fc2 = nn.Linear(inner_hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h):
        h1 = self.fc1(h)
        h1 = F.relu(h1)
        h1 = self.dropout(h1)
        h2 = self.fc2(h1)
        return h2


class TransformerSeqLayer(nn.Module):
    def __init__(self,
                 hidden_size,
                 inner_hidden_size,
                 dropout,
                 *args, **kwargs):
        super(TransformerSeqLayer, self).__init__()
        self.attn = MultiHeadSeqAttention(args)
        self.ff = FeedForwardLayer(hidden_size=hidden_size,
                                   inner_hidden_size=inner_hidden_size,
                                   dropout=dropout)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)

    def forward(self, h, h_prev, key_pe):
        # h = B x M x H
        # h_prev = B x L x H
        h_memory = torch.cat([h_prev, h], dim=1)  # B x (M+L) x H
        attn_out = self.attn(h, h_memory, h_memory, key_pe)
        h = self.norm1(h + attn_out)  # B x M x H
        ff_out = self.ff(h)
        out = self.norm2(h + ff_out)  # B x M x H
        return out


class TransformerSeq(nn.Module):
    def __init__(self,
                 vocab_size,
                 hidden_size,
                 nb_heads,
                 nb_layers,
                 attn_lim,
                 mem_size,
                 *args, **kwargs):
        super(TransformerSeq, self).__init__()
        self.in_emb = nn.Embedding(vocab_size, hidden_size)
        self.key_pe = nn.Parameter(
            torch.randn(1, hidden_size // nb_heads, attn_lim))

        self.layers = nn.ModuleList().extend(
            TransformerSeqLayer(args)
            for _ in range(nb_layers))

        self.out_emb = nn.Linear(hidden_size, vocab_size)
        # TODO: where is mem_sz defined?
        self.mem_size = mem_size

    def forward(self, x, h_prev, target=None):
        # x : B x M
        h = self.in_emb(x)  # B x M x H
        h_cache = []
        for l, layer in enumerate(self.layers):
            cache_size = attn_span.get_cache_size(layer.attn.attn)
            if cache_size > self.mem_size:
                h_cache_l = torch.cat(
                    [h_prev[l][:, -cache_size + self.mem_size:, :], h],
                    dim=1).detach()
            else:
                h_cache_l = h[:, -cache_size:, :].detach()
            h_cache.append(h_cache_l)
            h = layer(h, h_prev[l], self.key_pe)  # B x M x H

        out = F.log_softmax(self.out_emb(h), dim=-1)

        return out, h_cache
