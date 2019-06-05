#!/usr/bin/env python3

import math
import time

import torch

# TODO: review import statements
from utils.checkpoint import load_checkpoint, save_checkpoint
from models import attn_span


# separating batch training reduces memory usage (removes overlap?)
def _train_batch(model,
                 optimizer,
                 scheduler,
                 data,
                 offset,
                 stat,
                 attn_lim,
                 attn_span_loss,
                 test_only=False,
                 h_cache=None):
    # TODO: where is mem_sz defined?
    X = data[:, offset: offset + args.mem_sz].contiguous()
    Y = data[:, offset + 1: offset + args.mem_sz + 1].contiguous()

    out, h_cache = model(X, h_cache, Y)
    out = out.view(-1, out.size(-1))
    loss = F.nll_loss(out, Y.view(-1))
    stat['loss'] = stat.get('loss', 0) + loss.item()

    if not test_only:
        if attn_span_loss > 0:
            loss = loss + attn_span.loss(model, attn_span_loss, attn_lim)

        if scheduler is not None:
            scheduler.step()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if attn_span_loss > 0:
            attn_span.param_clamp(model)

    return h_cache


def _train_single_epoch(model,
                        optimizer,
                        scheduler,
                        data,
                        nb_batches,
                        full_test,
                        attn_lim,
                        attn_span_loss,
                        test_only=False,
                        train_pos=-1,
                        h_cache=None):
    stat = dict()
    # TODO: clean-up full_test, test_only, test_mode, attn_span_loss > 0, etc
    if test_only:
        model.eval()
    else:
        model.train()

    nb_batches_max = nb_batches
    # TODO: where is mem_sz defined?
    pos_shift_len = args.mem_sz
    if test_only:
        if full_test:
            assert train_pos == 0
            nb_batches_max = data.size(1)
            for h in h_cache:
                h.fill_(0)
        else:
            # reduce test batches for speed-up
            nb_batches_max = max(1, nb_batches // 10)
            nb_batches_max = min(nb_batches_max,
                                 math.ceil(data.size(1) / pos_shift_len))

    # TODO: where is test_mode defined?
    if args.test_mode:
        from tqdm import tqdm
        pbar = tqdm(total=data.size(1))

    # TODO: where is mem_sz defined?
    pos_max = data.size(1) - args.mem_sz
    nbatches = 0
    for batch_ind in range(nb_batches_max):
        if train_pos >= 0:
            offset = train_pos
        else:
            offset = random.randrange(pos_max)
        # TODO: where is test_mode defined?
        if args.test_mode:
            pbar.update(pos_shift_len)

        nbatches += 1
        h_cache = _train_batch(
            model=model, optimizer=optimizer,
            scheduler=scheduler, data=data, offset=offset,
            stat=stat, attn_lim=attn_lim, attn_span_loss=attn_span_loss,
            test_only=test_only, h_cache=h_cache)

        if train_pos >= 0:
            train_pos += pos_shift_len
            if train_pos >= pos_max:
                if full_test:
                    train_pos = 0
                    # only test once
                    break
                # randomize offset to reduce overfitting
                # TODO: where is mem_sz defined?
                train_pos = random.randrange(args.mem_sz)
                for h in h_cache:
                    h.fill_(0)

    # TODO: where is test_mode defined?
    if args.test_mode:
        pbar.close()
    for k, v in stat.items():
        stat[k] = v / nbatches
    return stat, train_pos, h_cache


def _train(device,
           model,
           optimizer,
           scheduler,
           plotter,
           train_data,
           val_data,
           test_data,
           checkpoint_path,
           checkpoint_freq,
           load_only,
           batch_size,
           hidden_size,
           full_test,
           nb_epochs,
           distributed,
           world_size,
           nb_batches,
           attn_lim,
           attn_span_loss,
           plot_enabled,
           *args, **kwargs):
    # resume training from last checkpoint
    ep_init = load_checkpoint(
        checkpoint_path=checkpoint_path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        plotter=plotter,
        distributed=compute_params['distributed'])

    # hid == cache init
    # pos: 0 --> sequential /  -1 --> random
    pos = [0] * 3
    hid = [
        [
            torch.zeros(
                batch_size,
                attn_span.get_cache_size(l.attn.attn),
                hidden_size).to(device, dtype=torch.float32)
            for l in model.module.layers
        ]
        for _ in range(3)
    ]

    if full_test:
        with torch.no_grad():
            stat_val, pos[1], hid[1] = _train_single_epoch(
                args, model=model,
                optimizer=optimizer, scheduler=scheduler, data=val_data,
                attn_lim=attn_lim, attn_span_loss=attn_span_loss,
                test_only=True, train_pos=pos[1], h_cache=hid[1])
            # TODO: replace print by logger
            print('val: {:.3f}bpc'.format(stat_val['loss'] / math.log(2)))

            stat_test, pos[2], hid[2] = _train_single_epoch(
                args, model=model,
                optimizer=optimizer, scheduler=scheduler, data=test_data,
                test_only=True, train_pos=pos[2], h_cache=hid[2])
            # TODO: replace print by logger
            print('test: {:.3f}bpc'.format(stat_test['loss'] / math.log(2)))
        return

    for ep in range(ep_init, nb_epochs):
        t_sta = time.time()
        # here the loss includes auxilary losses such as multi-position
        # training
        stat_train, pos[0], hid[0] = _train_single_epoch(
            args, model=model,
            optimizer=optimizer, scheduler=scheduler, data=train_data,
            attn_lim=attn_lim, attn_span_loss=attn_span_loss,
            train_pos=pos[0], h_cache=hid[0])
        elapsed = 1000 * (time.time() - t_sta) / nbatches
        with torch.no_grad():
            stat_val, pos[1], hid[1] = _train_single_epoch(
                args, model=model,
                optimizer=optimizer, scheduler=scheduler, data=val_data,
                attn_lim=attn_lim, attn_span_loss=attn_span_loss,
                test_only=True, train_pos=pos[1], h_cache=hid[1])

        if distributed:
            X = torch.zeros(2).to(device)
            X[0] = stat_train['loss']
            X[1] = stat_val['loss']
            torch.distributed.reduce(X, 0)
            if rank == 0:
                stat_train['loss'] = X[0] / world_size
                stat_val['loss'] = X[1] / world_size
            # why is there a continue? no plot? no checkpoint?
            else:
                continue

        if attn_span_loss > 0:
            attn_span.plot(
                plot_enabled=plot_enabled, model=model,
                plotter=plotter, stat_train=stat_train)

        plotter.step(ep=ep,
                     nb_batches=nb_batches,
                     stat_train=stat_train,
                     stat_val=stat_val,
                     elapsed=elapsed)

        # TODO: sqve_checkpoint should only save
        # load_only and freq should be tested in an outer condition
        save_checkpoint(
            checkpoint_path=checkpoint_path,
            checkpoint_freq=checkpoint_freq,
            ep=ep,
            model=model,
            optimizer=optimizer,
            plotter=plotter,
            scheduler=scheduler,
            load_only=load_only)


def train(trainer_params,
          compute_params,
          model_params,
          attn_span_params,
          optim_params,
          plot_params,
          device,
          model,
          optimizer,
          scheduler,
          plotter,
          train_data,
          val_data,
          test_data):
    _train(device=device,
           model=model,
           optimizer=optimizer,
           scheduler=scheduler,
           plotter=plotter,
           train_data=train_data,
           val_data=val_data,
           test_data=test_data,
           **{**compute_params,
              **model_params,
              **attn_span_params,
              **optim_params,
              **trainer_params,
              **plot_params})
