# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

#!/usr/bin/env python3

from argparse import Namespace
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F


class PersistentMemory(nn.Module):
    def __init__(self, size, nb_heads, head_dim, dropout):
        super(PersistentMemory, self).__init__()
        self.size = size
        self.nb_heads = nb_heads
        self.head_dim = head_dim
        # different heads have different vectors
        self.key = nn.Parameter(torch.randn(self.nb_heads, self.head_dim, self.size) / math.sqrt(self.head_dim))
        self.val = nn.Parameter(torch.randn(self.nb_heads, self.size, self.head_dim) / math.sqrt(self.size))
        self.dropout = nn.Dropout(dropout)
        self.adaptive_span = None

    def forward(self, query, attn):
        key = self.key.unsqueeze(0)
        val = self.val.unsqueeze(0)

        query = query.view((-1, self.nb_heads) + query.size()[1:])
        attn_pers = torch.matmul(query, key * math.sqrt(self.head_dim))
        attn_pers = attn_pers.view((-1,) + attn_pers.size()[2:])

        # compute softmax jointly
        attn = torch.cat((attn, attn_pers), dim=-1)
        attn = attn / math.sqrt(self.head_dim) # B x M X L_total
        attn = F.softmax(attn, dim=-1)
        attn_pers = attn[:, :, -key.size(-1):]
        attn = attn[:, :, :-key.size(-1)]  # B x M X L

        # adapt attention span
        if self.adaptive_span is not None:
            attn = self.adaptive_span(attn, normalize=False)
            # normalize the sum jointly!
            attn = torch.cat((attn, attn_pers), dim=-1)
            attn = attn / (attn.sum(-1, keepdim=True) + 1e-8)
            attn_pers = attn[:, :, -key.size(-1):]
            attn = attn[:, :, :-key.size(-1)]  # B x M X L

        attn_pers = self.dropout(attn_pers) # B x M X L

        attn_pers = attn_pers.view((-1, self.nb_heads) + attn_pers.size()[1:])
        out = torch.matmul(attn_pers, val * math.sqrt(self.size))
        out = out.view((-1,) + out.size()[2:])
        return attn, out
