# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

#!/usr/bin/env python3

import math
import random

import torch



def _train_step(model, X, Y, h_cache, eval_only, loss_div=1):
    """Single training step."""

    out, h_cache, dummy_loss = model(X, h_cache, target=Y)
    if model.module.adapt_io:
        loss = out.mean() + dummy_loss.sum()
    else:
        out = out.view(-1, out.size(-1))
        loss = torch.nn.functional.nll_loss(out, Y.view(-1))
    loss_value = loss.item() / loss_div

    if not eval_only:
        # loss term from adaptive-span
        if model.module.layers[0].attn.attn.adapt_span_enabled:
            loss += sum(layer.attn.attn.adaptive_span.get_loss()
                        for layer in model.module.layers)

        (loss / loss_div).backward()

    return loss_value, h_cache


def _train_batch(model, optimizer, scheduler, X, Y, h_cache,
                 eval_only, batch_split):
    """Train on a batch."""

    optimizer.zero_grad()

    if batch_split == 1:
        # process a batch in a single step (default behaviour)
        loss_value, h_cache = _train_step(model, X, Y, h_cache, eval_only)
    else:
        # split a batch into multiple pieces that each can fit in memory
        assert X.size(0) % batch_split == 0
        split_size = X.size(0) // batch_split
        loss_value = 0
        h_cache_list = []
        for split_ind in range(batch_split):
            split_slice = slice(split_ind*split_size, (split_ind+1)*split_size)
            split_h_cache = [h[split_slice,:,:] for h in h_cache]
            split_loss_value, split_h_cache = _train_step(
                model, X[split_slice,:], Y[split_slice],
                split_h_cache, eval_only, batch_split)
            loss_value += split_loss_value
            h_cache_list.append(split_h_cache)
        h_cache = [
            torch.cat(
                [h_cache_list[i][l] for i in range(batch_split)]
            , dim=0) for l in range(len(h_cache))]

    if not eval_only:
        if scheduler is not None:
            scheduler.step()
        if optimizer.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), optimizer.grad_clip)
        optimizer.step()

        # make sure span parameters are in a correct range
        if model.module.layers[0].attn.attn.adapt_span_enabled:
            for layer in model.module.layers:
                layer.attn.attn.adaptive_span.clamp_param()

    return loss_value, h_cache


def train_iteration(model, optimizer, scheduler, data, nb_batches_per_iter,
                    block_size, eval_only, train_pos, h_cache, batch_split):
    """Single training iteration."""
    if eval_only:
        model.eval()
    else:
        model.train()

    nb_batches_per_iter_max = nb_batches_per_iter
    if eval_only:
        # eval on fewer batches during training for speed-up
        nb_batches_per_iter_max = max(1, nb_batches_per_iter // 10)
        nb_batches_per_iter_max = min(nb_batches_per_iter_max,
                                      math.ceil(data.size(1) / block_size))

    loss_all = 0
    actual_nb_batches_per_iter = 0
    for _ in range(nb_batches_per_iter_max):
        actual_nb_batches_per_iter += 1
        X = data[:, train_pos: train_pos + block_size].contiguous()
        Y = data[:, train_pos + 1: train_pos + block_size + 1].contiguous()

        loss, h_cache = _train_batch(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            X=X, Y=Y,
            h_cache=h_cache,
            eval_only=eval_only,
            batch_split=batch_split)
        loss_all += loss
        train_pos += block_size
        if train_pos >= data.size(1) - block_size:
            # reached the end. randomize the offset to reduce overfitting
            train_pos = random.randrange(block_size)
            # reset the cache
            for h in h_cache:
                h.fill_(0)

    loss_all = loss_all / actual_nb_batches_per_iter
    return loss_all, train_pos, h_cache


# do full evaluation
def full_eval(model, optimizer, scheduler, data, block_size, hidden_size):
    model.eval()
    train_pos = 0
    nb_batches_per_iter_max = math.ceil(data.size(1) / block_size)
    h_cache = [
        torch.zeros(
            data.size(0),
            layer.attn.attn.get_cache_size(),
            hidden_size).to(data.device)
        for layer in model.module.layers]

    loss_all = 0
    actual_nb_batches_per_iter = 0
    for _ in range(nb_batches_per_iter_max):
        actual_nb_batches_per_iter += 1
        X = data[:, train_pos: train_pos + block_size].contiguous()
        Y = data[:, train_pos + 1: train_pos + block_size + 1].contiguous()

        loss, h_cache = _train_batch(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            X=X, Y=Y,
            h_cache=h_cache,
            eval_only=True,
            batch_split=1)
        loss_all += loss
        train_pos += block_size
        if train_pos >= data.size(1) - block_size:
            # Skip the remaining tokens as it can't make a whole block.
            # An effect on performance should be negligable for a large data.
            break

    loss_all = loss_all / actual_nb_batches_per_iter
    return loss_all
