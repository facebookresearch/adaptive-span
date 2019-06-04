#!/usr/bin/env python3

from __future__ import print_function
import time
import math
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from data import get_data
from models.transformer_seq import TransformerSeq
import models.attn_span as attn_span
from trainer import train

from utils.plotter import Plotter
from utils.checkpoint import load_checkpoint, save_checkpoint
from utils.adagrad import Adagrad

# TODO: remove
import submitit



def launch(model_params, optim_params, data_params, plot_params, compute_params, checkpoint_params, test_params, *, **other_params):




def launch(hid_sz: int,
           inner_hid_sz: int,
           nlayers: int,
           attn_lim: int,
           block_sz: int,
           nheads: int,
           dropout: float,
           lr: float,
           momentum: float,
           batch_sz: int,
           nbatches: int,
           nepochs: int,
           optim: str,
           lr_warmup: int,
           grad_clip: float,
           wdecay: float,
           data: str,
           plot: bool,
           plot_env: str,
           plot_host: str,
           no_cuda: bool,
           checkpoint: str,
           checkpoint_freq: int,
           load_only: bool,
           test_mode: bool,
           full_test: bool,
           distributed: bool,
           local_rank: int,
           submitit: bool,
           dist_init: str):
    use_cuda = not no_cuda and torch.cuda.is_available()

    # TODO: SHOULD BE IN submit_fair !!!
    if distributed:
        if submitit:
            job_env = submitit.JobEnvironment()
            local_rank = job_env.local_rank
            rank = job_env.global_rank
            world_size = job_env.num_tasks
            torch.distributed.init_process_group(
                backend='nccl',
                init_method=dist_init,
                rank=job_env.global_rank,
                world_size=job_env.num_tasks
            )
        else:
            torch.distributed.init_process_group(
                backend='nccl',
                init_method='env://'
            )
            rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()
        torch.cuda.set_device(local_rank)

    device = torch.device("cuda" if use_cuda else "cpu")
    dtype = torch.float32
    neg_inf = float(np.finfo(np.float32).min)

    train_data, val_data, test_data = get_data(
        data_path=data,
        batch_size=batch_sz,
        device=device)

    if distributed:
        batch_sz = batch_sz // world_size
        slice_data = slice(
            batch_sz * rank,
            batch_sz * (rank + 1))
        train_data = train_data[slice_data]
        val_data = val_data[slice_data]
        test_data = test_data[slice_data]

    plotter = Plotter(plot='plot', plot_env='plot_env', plot_host='plot_host')

    # MODEL
    model = TransformerSeq(args)

    if distributed:
        model = model.to(device, dtype=dtype)
        model = nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank)
    else:
        model = nn.DataParallel(model)
        model = model.to(device, dtype=dtype)

    nparameters = 0
    params = []
    for param in model.parameters():
        if param.requires_grad:
            nparameters += param.numel()
            params.append(param)
    print('nparameters={:.2f}M'.format(nparameters / 1e6))

    # OPTIM param
    if optim == 'sgd':
        optimizer = optim.SGD(
            params, lr=lr,
            weight_decay=wdecay,
            momentum=momentum)
    elif optim == 'adagrad':
        optimizer = Adagrad(
            params,
            lr=lr,
            weight_decay=wdecay,
            grad_clip=grad_clip)
    else:
        raise RuntimeError('wrong arg')
    if lr_warmup > 0:
        scheduler = optim.lr_scheduler.LambdaLR(
            optimizer, lambda ep: min(1, ep / lr_warmup))
    else:
        scheduler = None

    ep_init = load_checkpoint(
        checkpoint, model, optimizer, plotter, scheduler)

    # hid == cache init
    # pos: 0 --> sequential /  -1 --> random
    pos = [0 for _ in range(3)]
    hid = [
        [
            torch.zeros(
                batch_sz,
                attn_span.get_cache_size(l.attn.attn),
                hid_sz).to(device, dtype=dtype)
            for l in model.module.layers
        ]
        for _ in range(3)
    ]

    if test_mode:
        with torch.no_grad():
            stat_val, pos[1], hid[1] = train(
                args, model, optimizer, scheduler, val_data,
                test_only=True, train_pos=pos[1], h_cache=hid[1])
            print('val: {:.3f}bpc'.format(stat_val['loss'] / math.log(2)))

            stat_test, pos[2], hid[2] = train(
                args, model, optimizer, scheduler, test_data,
                test_only=True, train_pos=pos[2], h_cache=hid[2])
            print('test: {:.3f}bpc'.format(stat_test['loss'] / math.log(2)))
        return

    for ep in range(ep_init, nepochs):
        t_sta = time.time()
        ep = ep
        # here the loss includes auxilary losses such as multi-position
        # training
        stat_train, pos[0], hid[0] = train(
            args, model, optimizer, scheduler, train_data,
            train_pos=pos[0], h_cache=hid[0])
        elapsed = 1000 * (time.time() - t_sta) / nbatches
        with torch.no_grad():
            stat_val, pos[1], hid[1] = train(
                args, model, optimizer, scheduler, val_data,
                test_only=True, train_pos=pos[1], h_cache=hid[1])

        if distributed:
            X = torch.zeros(2).to(device)
            X[0] = stat_train['loss']
            X[1] = stat_val['loss']
            torch.distributed.reduce(X, 0)
            if rank == 0:
                stat_train['loss'] = X[0] / world_size
                stat_val['loss'] = X[1] / world_size
            else:
                continue

        if attn_span_loss > 0:
            attn_span.log(args, model, plotter, stat_train)

        plotter.step(ep, nbatches, stat_train, stat_val, elapsed)
        save_checkpoint(
            checkpoint, checkpoint_freq,
            ep, model, optimizer, plotter, scheduler)


if __name__ == '__main__':
    launch(**get_params())
