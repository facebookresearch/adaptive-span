import random
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import models.attn_span as attn_span


# separating batch training reduces memory usage (removes overlap?)
def train_batch(args, model, optimizer, scheduler, data, offset, stat, test_only=False, h_cache=None):
    X = data[:, offset:offset+args.mem_sz].contiguous()
    Y = data[:, offset+1:offset+args.mem_sz+1].contiguous()

    out, h_cache = model(X, h_cache, Y)
    out = out.view(-1, out.size(-1))
    loss = F.nll_loss(out, Y.view(-1))
    stat['loss'] = stat.get('loss', 0) + loss.item()

    if not test_only:
        if args.attn_span_loss > 0:
            loss = loss + attn_span.loss(args, model, stat)

        if scheduler is not None:
            scheduler.step()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if args.attn_span_loss > 0:
            attn_span.param_clamp(args, model)

    return h_cache

def train(args, model, optimizer, scheduler, data, test_only=False, train_pos=-1, h_cache=None):
    stat = dict()
    if test_only:
        model.eval()
    else:
        model.train()

    nbatches_max = args.nbatches
    pos_shift_len = args.mem_sz
    if test_only:
        if args.full_test:
            assert train_pos == 0
            nbatches_max = data.size(1)
            for h in h_cache:
                h.fill_(0)
        else:
            # reduce test batches for speed-up
            nbatches_max = max(1, args.nbatches // 10)
            nbatches_max = min(nbatches_max, math.ceil(data.size(1) / pos_shift_len))

    if args.test_mode:
        from tqdm import tqdm
        pbar = tqdm(total=data.size(1))

    pos_max = data.size(1) -  args.mem_sz
    nbatches = 0
    for batch_ind in range(nbatches_max):
        if train_pos >= 0:
            offset = train_pos
        else:
            offset = random.randrange(pos_max)
        if args.test_mode:
            pbar.update(pos_shift_len)

        nbatches += 1
        h_cache = train_batch(args, model, optimizer, scheduler, data, offset, stat, test_only, h_cache)

        if train_pos >= 0:
            train_pos += pos_shift_len
            if train_pos >= pos_max:
                if args.full_test:
                    train_pos = 0
                    # only test once
                    break
                # randomize offset to reduce overfitting
                train_pos = random.randrange(args.mem_sz)
                for h in h_cache:
                    h.fill_(0)

    if args.test_mode:
        pbar.close()
    for k, v in stat.items():
        stat[k] = v / nbatches
    return stat, train_pos, h_cache
