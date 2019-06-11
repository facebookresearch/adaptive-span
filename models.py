#!/usr/bin/env python3

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from adaptive_span import AdaptiveSpan

# Size notations:
# B = batch_sizez, H = hidden_size, M = block_sizz, L = attn_span_lim

# each position will only attent to its previous L positions
# note that attention doesn't include the current step itself


def _skew(X, pad_value):
    """shift every row 1 step to right"""
    # X = B x M x L
    B, M, L = X.size()
    X = F.pad(X, (0, M + 1), value=pad_value)  # B x M x (L+M+1)
    X = X.view(B, -1)  # B x ML+MM+M
    X = X[:, :-M]  # B x ML+MM
    X = X.view(B, M, M + L)  # B x M x L+M
    return X


def _unskew(X):
    """reverse _skew operation"""
    # X = B x M x L+M
    B, M, L = X.size()
    L -= M
    X = X.view(B, -1)  # B x ML+MM
    X = F.pad(X, (0, M))  # B x ML+MM+M
    X = X.view(B, M, M + L + 1)  # B x M x L+M+1
    X = X[:, :, :L]  # B x M x L
    return X


class SeqAttention(nn.Module):
    """Adaptive attention span for Transformer"""
    def __init__(self,
                 attn_span_enabled,
                 attn_span_cache_enabled,
                 dropout,
                 hidden_size,
                 nb_heads,
                 attn_span_loss,
                 block_size,
                 model_params,
                 attn_span_params):
        nn.Module.__init__(self)
        self.attn_span_enabled = attn_span_enabled
        self.attn_span_cache_enabled = attn_span_cache_enabled
        self.dropout = nn.Dropout(dropout)
        self.nb_heads = nb_heads
        self.head_dim = hidden_size // nb_heads
        self.attn_span_loss = attn_span_loss
        self.block_size = block_size
        self.adaptive_span = AdaptiveSpan(
            attn_span_enabled=attn_span_enabled,
            attn_span_cache_enabled=attn_span_cache_enabled,
            dropout=dropout,
            nb_heads=nb_heads,
            hidden_size=hidden_size,
            attn_span_lim=attn_span_params['attn_span_lim'],
            attn_span_loss=attn_span_loss,
            block_size=block_size,
            attn_span_len=attn_span_params['attn_span_len'],
            attn_span_init=attn_span_params['attn_span_init'])

    def forward(self, query, key, value, key_pe):
        # query size = B x M x H
        # key, value sizes = B x (M+L) x H

        # compute attention span
        if self.attn_span_loss > 0:
            skip_len = self.adaptive_span.compute_skip_len()
            key, value, key_pe = self.adaptive_span.crop(key=key,
                                                         value=value,
                                                         key_pe=key_pe,
                                                         skip_len=skip_len)

        # compute attention from context
        # B x M (dest) x (M+L) (src)
        attn_cont = torch.matmul(query, key.transpose(-1, -2))
        attn_cont = _unskew(attn_cont)  # B x M x L

        # compute the effect of position embedding
        attn_pos = torch.matmul(query, key_pe)  # B x M x L_pos
        attn = attn_cont + attn_pos

        attn = attn / math.sqrt(self.head_dim)  # B x M X L_pos
        attn = F.softmax(attn, dim=-1)

        if self.attn_span_loss > 0:
            attn = self.adaptive_span(attn=attn, skip_len=skip_len)
        attn = self.dropout(attn)  # B x M X L_pos

        attn_cont = _skew(attn, 0)  # B x M X (L+M)
        out = torch.matmul(attn_cont, value)  # B x M x H

        return out


class MultiHeadSeqAttention(nn.Module):
    def __init__(self,
                 hidden_size,
                 nb_heads,
                 block_size,
                 model_params,
                 attn_span_params):
        nn.Module.__init__(self)
        assert hidden_size % nb_heads == 0
        self.nb_heads = nb_heads
        self.head_dim = hidden_size // nb_heads
        self.attn = SeqAttention(
            attn_span_enabled=attn_span_params['attn_span_enabled'],
            attn_span_cache_enabled=attn_span_params['attn_span_cache_enabled'],
            dropout=model_params['dropout'],
            hidden_size=hidden_size,
            nb_heads=nb_heads,
            attn_span_loss=attn_span_params['attn_span_loss'],
            block_size=model_params['block_size'],
            model_params=model_params,
            attn_span_params=attn_span_params)
        self.proj_query = nn.Linear(
            hidden_size, self.head_dim * nb_heads, bias=False)
        self.proj_out = nn.Linear(
            self.head_dim * nb_heads, hidden_size, bias=False)
        self.proj_val = nn.Linear(
            hidden_size, self.head_dim * nb_heads, bias=False)
        self.proj_key = nn.Linear(
            hidden_size, self.head_dim * nb_heads, bias=False)
        self.block_size = block_size

    def head_reshape(self, x):
        K = self.nb_heads
        D = self.head_dim
        x = x.view(x.size()[:-1] + (K, D))  # B x (M+L) x K x D
        x = x.transpose(1, 2).contiguous()  # B x K x (M+L) x D
        x = x.view(-1, x.size(-2), x.size(-1))  # B_K x (M+L) x D
        return x

    def forward(self, query, key, value, key_pe):
        B = query.size(0)
        K = self.nb_heads
        D = self.head_dim
        M = self.block_size

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
                 dropout):
        nn.Module.__init__(self)
        self.fc1 = nn.Linear(hidden_size, inner_hidden_size)
        self.fc2 = nn.Linear(inner_hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h):
        h1 = F.relu(self.fc1(h))
        h1 = self.dropout(h1)
        h2 = self.fc2(h1)
        return h2


class TransformerSeqLayer(nn.Module):
    def __init__(self,
                 hidden_size,
                 model_params,
                 attn_span_params):
        nn.Module.__init__(self)
        self.attn = MultiHeadSeqAttention(hidden_size=hidden_size,
                                          nb_heads=model_params['nb_heads'],
                                          block_size=model_params['block_size'],
                                          model_params=model_params,
                                          attn_span_params=attn_span_params)
        self.ff = FeedForwardLayer(
            hidden_size=model_params['hidden_size'],
            inner_hidden_size=model_params['inner_hidden_size'],
            dropout=model_params['dropout'])
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)

    def forward(self, h, h_cache, key_pe):
        # h = B x M x H
        # h_cache = B x L x H
        h_all = torch.cat([h_cache, h], dim=1)  # B x (M+L) x H
        attn_out = self.attn(h, h_all, h_all, key_pe)
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
                 attn_span_lim,
                 block_size,
                 model_params,
                 attn_span_params):
        nn.Module.__init__(self)
        # token embeddings
        self.in_emb = nn.Embedding(vocab_size, hidden_size)
        self.out_emb = nn.Linear(hidden_size, vocab_size)
        self.key_pe = nn.Parameter(
            torch.randn(1, hidden_size // nb_heads, attn_span_lim))

        self.layers = nn.ModuleList()
        self.layers.extend(
            TransformerSeqLayer(hidden_size=model_params['model_params'],
                                model_params=model_params,
                                attn_span_params=attn_span_params)
            for _ in range(nb_layers))

        self.block_size = block_size

    def forward(self, x, h_cache):
        # x size = B x M
        h = self.in_emb(x)  # B x M x H
        h_cache_next = []
        for l, layer in enumerate(self.layers):
            cache_size = layer.attn.attn.adaptive_span.get_cache_size()
            if cache_size > self.block_size:
                h_cache_next_l = torch.cat(
                    [h_cache[l][:, -cache_size + self.block_size:, :], h],
                    dim=1).detach()
            else:
                h_cache_next_l = h[:, -cache_size:, :].detach()
            h_cache_next.append(h_cache_next_l)
            h = layer(h, h_cache[l], self.key_pe)  # B x M x H

        out = F.log_softmax(self.out_emb(h), dim=-1)

        return out, h_cache_next
