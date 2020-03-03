# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

#!/usr/bin/env python3

import torch
from torch import nn
from torch.nn import functional as F


class AdaptiveEmbedding(nn.Module):
    """ An adaptive embedding module from "Adaptive Input Representations for
    Neural Language Modeling" (https://arxiv.org/abs/1809.10853) """
    def __init__(self, n_tokens, d_embed, d_proj, cutoffs, div_val=4):
        super(AdaptiveEmbedding, self).__init__()

        self.n_tokens = n_tokens
        self.d_embed = d_embed
        self.d_proj = d_proj

        assert 0 < min(cutoffs) <= max(cutoffs) < n_tokens
        self.cutoffs = cutoffs + [n_tokens]
        self.cutoff_ends = [0] + self.cutoffs
        self.div_val = div_val
        assert self.div_val > 1
        assert len(self.cutoffs) > 1

        self.emb_scale = d_proj ** 0.5

        self.emb_layers = nn.ModuleList()
        self.emb_projs = nn.ParameterList()

        # embedding layers / projections
        for i in range(len(self.cutoffs)):
            l_idx, r_idx = self.cutoff_ends[i], self.cutoff_ends[i + 1]
            d_emb_i = d_embed // (div_val ** i)
            self.emb_layers.append(nn.Embedding(r_idx - l_idx, d_emb_i))
            self.emb_projs.append(nn.Linear(d_emb_i, d_proj).weight)

    def forward(self, indices):
        param = self.emb_layers[0].weight.data
        idx_flat = indices.contiguous().view(-1)
        emb_flat = torch.zeros([idx_flat.size(0), self.d_proj], dtype=param.dtype, device=param.device)

        # for each cluster
        for i in range(len(self.cutoffs)):
            # find elements in that cluster
            l_idx, r_idx = self.cutoff_ends[i], self.cutoff_ends[i + 1]
            mask_i = (idx_flat >= l_idx) & (idx_flat < r_idx)

            # if there are no elements, continue
            indices_i = mask_i.nonzero().squeeze()
            if indices_i.numel() == 0:
                continue

            # add embeddings from this cluster
            idx_i = idx_flat.index_select(0, indices_i) - l_idx
            emb_i = self.emb_layers[i](idx_i)
            emb_i = F.linear(emb_i, self.emb_projs[i])
            emb_flat = emb_flat.type_as(emb_i) if emb_flat.dtype != emb_i.dtype else emb_flat  # small hack for AMP-O1
            emb_flat.index_copy_(0, indices_i, emb_i)

        # reshape embeddings
        embed = emb_flat.view(*indices.size(), self.d_proj)

        # rescale embeddings
        embed.mul_(self.emb_scale)

        return embed


class ProjectedAdaptiveLogSoftmax(nn.Module):
    """ An efficient softmax implementation from "Efficient softmax
    approximation for GPUs" (http://arxiv.org/abs/1609.04309). """
    def __init__(self, n_tokens, d_embed, d_proj, cutoffs, div_val=4):
        super(ProjectedAdaptiveLogSoftmax, self).__init__()

        self.n_tokens = n_tokens
        self.d_embed = d_embed
        self.d_proj = d_proj

        assert 0 < min(cutoffs) <= max(cutoffs) < n_tokens
        self.cutoffs = cutoffs + [n_tokens]
        self.cutoff_ends = [0] + self.cutoffs
        self.div_val = div_val
        assert self.div_val > 1
        assert len(self.cutoffs) > 1

        self.shortlist_size = self.cutoffs[0]
        self.n_clusters = len(self.cutoffs) - 1
        self.head_size = self.shortlist_size + self.n_clusters

        # clusters parameters
        self.cluster_proj = nn.Linear(self.d_embed, self.n_clusters)

        self.out_layers = nn.ModuleList()
        self.out_projs = nn.ParameterList()

        # output layers / projections
        for i in range(len(self.cutoffs)):
            l_idx, r_idx = self.cutoff_ends[i], self.cutoff_ends[i + 1]
            d_emb_i = d_embed // (div_val ** i)
            self.out_projs.append(nn.Linear(d_emb_i, d_proj).weight)
            self.out_layers.append(nn.Linear(d_emb_i, r_idx - l_idx))

    def _compute_logit(self, hidden, weight, bias, proj):
        proj_hid = F.linear(hidden, proj.t().contiguous())  # TODO: .contiguous() not necessary?
        logit = F.linear(proj_hid, weight, bias=bias)
        return logit

    def forward(self, hidden, target):
        """
        Input:
            - `hidden` FloatTensor(shape + (d_proj,))
            - `target` LongTensor(shape)
        Output:
            - `nll` FloatTensor(shape)
        """
        assert hidden.shape[-1] == self.d_proj
        assert hidden.shape[:-1] == target.shape
        shape = target.shape
        hidden = hidden.view(-1, self.d_proj)
        target = target.view(-1)

        # construct weights and biases
        weights, biases = [], []
        for i in range(len(self.cutoffs)):
            weight_i = self.out_layers[i].weight
            bias_i = self.out_layers[i].bias
            if i == 0:
                weight_i = torch.cat([weight_i, self.cluster_proj.weight], dim=0)
                bias_i = torch.cat([bias_i, self.cluster_proj.bias], dim=0)
            weights.append(weight_i)
            biases.append(bias_i)

        # head / cluster assignments
        head_logit = self._compute_logit(hidden, weights[0], biases[0], self.out_projs[0])
        head_logprob = F.log_softmax(head_logit.float(), dim=1)

        # final log-probabilities
        nll = torch.zeros_like(target, dtype=torch.float32, device=hidden.device)

        offset = 0
        cutoff_values = [0] + self.cutoffs

        # for each cluster
        for i in range(len(cutoff_values) - 1):

            # select the target tokens in that cluster
            l_idx, r_idx = cutoff_values[i], cutoff_values[i + 1]
            mask_i = (target >= l_idx) & (target < r_idx)
            indices_i = mask_i.nonzero().squeeze()

            # if there are not any, there is nothing to do
            if indices_i.numel() == 0:
                continue

            # index in current cluster
            target_i = target.index_select(0, indices_i) - l_idx
            head_logprob_i = head_logprob.index_select(0, indices_i)

            if i == 0:
                # for targets in the head cluster, there is just the head score
                logprob_i = head_logprob_i.gather(1, target_i[:, None]).squeeze(1)
            else:
                # otherwise, we sum the cluster assignment (head) and target scores
                hidden_i = hidden.index_select(0, indices_i)
                tail_logit_i = self._compute_logit(hidden_i, weights[i], biases[i], self.out_projs[i])
                tail_logprob_i = F.log_softmax(tail_logit_i.float(), dim=1)
                logprob_i = head_logprob_i[:, -i] + tail_logprob_i.gather(1, target_i[:, None]).squeeze(1)

            # populate output
            nll.index_copy_(0, indices_i, -logprob_i)

            offset += logprob_i.size(0)

        return nll.view(shape)


def compute_dummy_loss(in_emb, out_emb):
    # hack to fix adaptive ou/in with distributed code
    dummy_loss =  0 * (
        sum(x.weight[0, 0] for x in in_emb.emb_layers) +
        sum(x[0, 0] for x in in_emb.emb_projs) +
        sum(x[0, 0] for x in out_emb.out_projs) +
        sum(x.weight[0, 0] for x in out_emb.out_layers) +
        sum(x.bias[0] for x in out_emb.out_layers)
    )
    return dummy_loss


def build_adaptive_io(vocab_size, hidden_size, adapt_io_cutoffs,
    adapt_io_divval, adapt_io_tied, **kargs):
    in_emb = AdaptiveEmbedding(
        vocab_size, hidden_size, hidden_size,
        cutoffs=adapt_io_cutoffs,
        div_val=adapt_io_divval)
    out_emb = ProjectedAdaptiveLogSoftmax(
        vocab_size, hidden_size, hidden_size,
        cutoffs=adapt_io_cutoffs,
        div_val=adapt_io_divval)
    if adapt_io_tied:
        for i in range(len(adapt_io_cutoffs) + 1):
            out_emb.out_layers[i].weight = in_emb.emb_layers[i].weight
            out_emb.out_projs[i] = in_emb.emb_projs[i]
    return in_emb, out_emb
