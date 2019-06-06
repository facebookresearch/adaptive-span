#!/usr/bin/env python3

import math
import time

import torch

# TODO: review import statements
from utils.logger import Logger
from utils.plotter import Plotter
from utils.checkpoint import load_checkpoint, save_checkpoint
from models import attn_span


def _log_iter(logger,
              iter_no,
              nb_batches,
              stat_train,
              stat_val,
              elapsed,
              attn_span_loss,
              model):
    X = (iter_no + 1) * nb_batches
    # TODO: why log(2)
    train_mp_bpc = stat_train['loss'] / math.log(2)
    val_bpc = stat_val['loss'] / math.log(2)
    print('{}\ttrain: {:.2f}bpc\tval: {:.2f}bpc\tms/batch: {:.1f}'.format(
        X, train_mp_bpc, val_bpc, elapsed))
    logger.log(title='X', value=X)
    logger.log(title='train_mp_bpc', value=train_mp_bpc)
    logger.log(title='val_bpc', value=val_bpc)

    span_latest = []
    if attn_span_loss > 0:
        for i, l in enumerate(model.module.layers):
            span = l.attn.attn.span_mask.size_ratio.view(-1)
            span_latest.append(span)
            # TODO: why this line?
            span = span.mean().item()
        span_latest = torch.cat(span_latest, dim=0)
        logger.log('span_avg', span_latest.mean().item())
        logger.log('span_max', span_latest.max().item())
    return span_latest


def _plot_iter(plotter, span_latest, logger):
    plotter.plot(title='train_mp_bpc',
                 X=logger.get_data('X'),
                 Y=logger.get_data('train_mp_bpc'))
    plotter.plot(title='val_bpc',
                 X=logger.get_data('X'),
                 Y=logger.get_data('val_bpc'))
    if span_latest:
        plotter.plot(title='span_latest',
                     Y=logger.get_data('span_latest'))
    plotter.save()


def _is_checkpoint(iter_no, checkpoint_freq):
    return checkpoint_freq > 0 and (iter_no + 1) % checkpoint_freq == 0


def _save_iter(load_only,
               checkpoint_freq,
               checkpoint_path,
               iter_no,
               model,
               optimizer,
               scheduler,
               logger):
    if not load_only:
        actual_checkpoint_path = checkpoint_path
        if _is_checkpoint(iter_no, checkpoint_freq):
            actual_checkpoint_path += f".{iter_no+1}"
        save_checkpoint(
            checkpoint_path=actual_checkpoint_path,
            iter_no=iter_no,
            model=model,
            optimizer=optimizer,
            logger=logger,
            scheduler=scheduler)


# separating batch training reduces memory usage (removes overlap?)
def _train_batch(model,
                 optimizer,
                 scheduler,
                 data,
                 offset,
                 stat,
                 attn_span_lim,
                 attn_span_loss,
                 block_size,
                 test_only=False,
                 h_cache=None):
    X = data[:, offset: offset + block_size].contiguous()
    Y = data[:, offset + 1: offset + block_size + 1].contiguous()

    out, h_cache = model(X, h_cache, Y)
    out = out.view(-1, out.size(-1))
    loss = F.nll_loss(out, Y.view(-1))
    stat['loss'] = stat.get('loss', 0) + loss.item()

    if not test_only:
        if attn_span_loss > 0:
            loss += sum(l.attn.attn.compute_extra_loss()
                        for l in model.module.layers)

        if scheduler is not None:
            scheduler.step()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if attn_span_loss > 0:
            for l in model.module.layers:
                l.attn.attn.clamp_param()

    return h_cache


def _train_single_iteration(model,
                            optimizer,
                            scheduler,
                            data,
                            nb_batches,
                            full_test,
                            attn_span_lim,
                            attn_span_loss,
                            block_size,
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
    pos_shift_len = block_size
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

    pos_max = data.size(1) - block_size
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
            stat=stat, attn_span_lim=attn_span_lim, attn_span_loss=attn_span_loss,
            test_only=test_only, h_cache=h_cache)

        if train_pos >= 0:
            train_pos += pos_shift_len
            if train_pos >= pos_max:
                if full_test:
                    train_pos = 0
                    # only test once
                    break
                # randomize offset to reduce overfitting
                train_pos = random.randrange(block_size)
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
           train_data,
           val_data,
           test_data,
           checkpoint_path,
           checkpoint_freq,
           load_only,
           batch_size,
           hidden_size,
           full_test,
           nb_iter,
           distributed,
           world_size,
           nb_batches,
           attn_span_lim,
           attn_span_loss,
           plot_enabled,
           plot_env,
           plot_host,
           *args, **kwargs):
    # create logger and plotter
    logger = Logger()
    plotter = Plotter(
        plot_enabled=plot_enabled, plot_env=plot_env, plot_host=plot_host)

    # resume training from last checkpoint
    iter_init = load_checkpoint(
        checkpoint_path=checkpoint_path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        plotter=plotter,
        distributed=env_params['distributed'])

    # hid == cache init
    # pos: 0 --> sequential /  -1 --> random
    pos = [0] * 3
    hid = [
        [
            torch.zeros(
                batch_size,
                l.attn.attn.get_cache_size(),
                hidden_size).to(device, dtype=torch.float32)
            for l in model.module.layers
        ]
        for _ in range(3)
    ]

    if full_test:
        with torch.no_grad():
            stat_val, pos[1], hid[1] = _train_single_iteration(
                model=model,
                optimizer=optimizer, scheduler=scheduler, data=val_data,
                attn_span_lim=attn_span_lim, attn_span_loss=attn_span_loss,
                test_only=True, train_pos=pos[1], h_cache=hid[1])
            # TODO: replace print by logger
            print('val: {:.3f}bpc'.format(stat_val['loss'] / math.log(2)))

            stat_test, pos[2], hid[2] = _train_single_iteration(
                model=model,
                optimizer=optimizer, scheduler=scheduler, data=test_data,
                test_only=True, train_pos=pos[2], h_cache=hid[2])
            # TODO: replace print by logger
            print('test: {:.3f}bpc'.format(stat_test['loss'] / math.log(2)))
        return

    for iter_no in range(iter_init, nb_iter):
        t_sta = time.time()
        # here the loss includes auxilary losses such as multi-position
        # training
        stat_train, pos[0], hid[0] = _train_single_iteration(
            model=model,
            optimizer=optimizer, scheduler=scheduler, data=train_data,
            attn_span_lim=attn_span_lim, attn_span_loss=attn_span_loss,
            train_pos=pos[0], h_cache=hid[0])
        elapsed = 1000 * (time.time() - t_sta) / nbatches
        with torch.no_grad():
            stat_val, pos[1], hid[1] = _train_single_iteration(
                model=model,
                optimizer=optimizer, scheduler=scheduler, data=val_data,
                attn_span_lim=attn_span_lim, attn_span_loss=attn_span_loss,
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

        span_latest = _log_iter(logger=logger,
                                iter_no=iter_no,
                                nb_batches=nb_batches,
                                stat_train=stat_train,
                                stat_val=stat_val,
                                elapsed=elapsed,
                                attn_span_los=attn_span_loss,
                                model=model)
        _plot_iter(logger=logger, span_latest=span_latest, plotter=plotter)
        _save_iter(load_only=load_only,
                   checkpoint_freq=checkpoint_freq,
                   checkpoint_path=actual_checkpoint_path,
                   iter_no=iter_no,
                   model=model,
                   optimizer=optimizer,
                   scheduler=scheduler,
                   logger=logger)


def train(trainer_params,
          env_params,
          model_params,
          attn_span_params,
          optim_params,
          plotter_params,
          device,
          model,
          optimizer,
          scheduler,
          train_data,
          val_data,
          test_data):
    _train(device=device,
           model=model,
           optimizer=optimizer,
           scheduler=scheduler,
           train_data=train_data,
           val_data=val_data,
           test_data=test_data,
           **{**env_params,
              **model_params,
              **attn_span_params,
              **optim_params,
              **trainer_params,
              **plotter_params})
