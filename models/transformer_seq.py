#!/usr/bin/env python3

from __future__ import print_function
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.utils import *
import models.attn_span as attn_span

# B = batch_sz
# H = hid_sz
# M = mem_sz
# L = attn_lim

# each position will only attent to its previous L positions
# (from the lower layer)
# no self-attention: L positions doesn't include the current step


class SeqAttention(nn.Module):
    def __init__(self, args):
        super(SeqAttention, self).__init__()
        self.args = args
        self.dropout = nn.Dropout(args.dropout)
        self.head_dim = args.hid_sz // args.nheads
        if self.args.attn_span_loss > 0:
            attn_span.init(args, self)

    def forward(self, query, key, value, key_pe):
        B = query.size(0)
        H = self.head_dim
        L = self.args.attn_lim
        M = self.args.mem_sz
        # query = B x M x H
        # key, value = B x (M+L) x H

        # compute attention span
        if self.args.attn_span_loss > 0:
            skip_len = attn_span.compute(self.args, self)
            key, value, key_pe = attn_span.crop(self.args, skip_len, key, value, key_pe)

        # compute attention from context
        attn_cont = torch.matmul(query, key.transpose(-1, -2)) # B x M (dest) x (M+L) (src)
        attn_cont = unskew(attn_cont) # B x M x L

        # compute the effect of position embedding
        attn_pos = torch.matmul(query, key_pe) # B x M x L_pos
        attn = attn_cont + attn_pos

        attn = attn / math.sqrt(H) # B x M X L_pos
        attn = F.softmax(attn, dim=-1)

        if self.args.attn_span_loss > 0:
            attn = attn_span.mask(self.args, attn, self, skip_len)
        attn = self.dropout(attn) # B x M X L_pos

        attn_cont = skew(attn, 0)  # B x M X (L+M)
        out = torch.matmul(attn_cont, value) # B x M x H

        return out


class MultiHeadSeqAttention(nn.Module):
    def __init__(self, args):
        super(MultiHeadSeqAttention, self).__init__()
        self.args = args
        assert args.hid_sz % args.nheads == 0
        self.head_dim = args.hid_sz // args.nheads
        self.attn = SeqAttention(args)
        self.proj_query = nn.Linear(args.hid_sz, self.head_dim * args.nheads, bias=False)
        self.proj_out = nn.Linear(self.head_dim * args.nheads, args.hid_sz, bias=False)
        self.proj_val = nn.Linear(args.hid_sz, self.head_dim * args.nheads, bias=False)
        self.proj_key = nn.Linear(args.hid_sz, self.head_dim * args.nheads, bias=False)

    def head_reshape(self, x):
        K = self.args.nheads
        D = self.head_dim
        sz = x.size()
        sz = sz[:-1] + (K, D) # B x (M+L) x K x D
        x = x.view(sz) # B x (M+L) x K x D
        x = x.transpose(1, 2).contiguous() # B x K x (M+L) x D
        x = x.view(-1, x.size(-2), x.size(-1)) # B_K x (M+L) x D
        return x

    def forward(self, query, key, value, key_pe):
        B = query.size(0)
        K = self.args.nheads
        D = self.head_dim
        M = self.args.mem_sz

        query = self.proj_query(query)
        query = self.head_reshape(query)
        value = self.proj_val(value)
        value = self.head_reshape(value)
        key = self.proj_key(key)
        key = self.head_reshape(key)

        out = self.attn(query, key, value, key_pe) # B_K x M x D
        out = out.view(B, K, M, D) # B x K x M x D
        out = out.transpose(1, 2).contiguous() # B x M x K x D
        out = out.view(B, M, -1) # B x M x K_D
        out = self.proj_out(out)
        return out


class FeedForwardLayer(nn.Module):
    def __init__(self, args):
        super(FeedForwardLayer, self).__init__()
        self.args = args
        self.fc1 = nn.Linear(args.hid_sz, args.inner_hid_sz)
        self.fc2 = nn.Linear(args.inner_hid_sz, args.hid_sz)
        self.dropout = nn.Dropout(args.dropout)

    def forward(self, h):
        h1 = self.fc1(h)
        h1 = F.relu(h1)
        h1 = self.dropout(h1)
        h2 = self.fc2(h1)
        return h2


class TransformerSeqLayer(nn.Module):
    def __init__(self, args):
        super(TransformerSeqLayer, self).__init__()
        self.args = args
        self.attn = MultiHeadSeqAttention(args)
        self.ff = FeedForwardLayer(args)
        self.norm1 = nn.LayerNorm(args.hid_sz)
        self.norm2 = nn.LayerNorm(args.hid_sz)

    def forward(self, h, h_prev, key_pe):
        # h = B x M x H
        # h_prev = B x L x H
        h_memory = torch.cat([h_prev, h], dim=1) # B x (M+L) x H

        attn_out = self.attn(h, h_memory, h_memory, key_pe)
        h = self.norm1(h + attn_out) # B x M x H
        ff_out = self.ff(h)
        out = self.norm2(h + ff_out) # B x M x H
        return out


class TransformerSeq(nn.Module):
    def __init__(self, args):
        super(TransformerSeq, self).__init__()
        self.args = args
        self.in_emb = nn.Embedding(args.vocab_sz, args.hid_sz)
        self.key_pe = nn.Parameter(torch.randn(1, args.hid_sz // args.nheads, args.attn_lim))

        self.layers = nn.ModuleList()
        attn_lim_max = 0
        for l in range(args.nlayers):
            self.layers.append(TransformerSeqLayer(args))

        self.out_emb = nn.Linear(args.hid_sz, args.vocab_sz)


    def forward(self, x, h_prev, target = None):
        # x : B x M
        h = self.in_emb(x) # B x M x H
        h_cache = []
        for l in range(self.args.nlayers):
            cache_size = attn_span.get_cache_size(self.layers[l].attn.attn)
            if cache_size > self.args.mem_sz:
                h_cache.append(torch.cat([h_prev[l][:, -cache_size+self.args.mem_sz:, :], h], dim=1).detach())
            else:
                h_cache.append(h[:, -cache_size:, :].detach())
            h = self.layers[l](h, h_prev[l], self.key_pe) # B x M x H

        out = F.log_softmax(self.out_emb(h), dim=-1)

        return out, h_cache
