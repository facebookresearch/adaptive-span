<<<<<<< HEAD
#!/usr/bin/env python3

import argparse
=======
from __future__ import print_function
import time
import copy
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
from utils.logger import Logger
import utils.checkpoint as checkpoint
from utils.adagrad import Adagrad

# TODO: remove
import submitit

def get_parser():
    parser = argparse.ArgumentParser()
    # model related
    parser.add_argument('--hid-sz', type=int, default=256,
                        help='hidden size (i.e. model size)')
    parser.add_argument('--inner-hid-sz', type=int, default=1024,
                        help='inner hidden size of FF layer')
    parser.add_argument('--nlayers', type=int, default=8,
                        help='number of layers')
    parser.add_argument('--attn-lim', type=int, default=32,
                        help='length of the attention span')
    parser.add_argument('--block-sz', type=int, default=64,
                        help='block size (the length of sequence to process in parallel)')
    parser.add_argument('--nheads', type=int, default=2,
                        help='the number of attention heads')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='dropout rate of ReLU and attention')
    # optimization related
    parser.add_argument('--lr', type=float, default=0.03,
                        help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum')
    parser.add_argument('--batch-sz', type=int, default=64,
                        help='batch size')
    parser.add_argument('--nbatches', type=int, default=1000,
                        help='the number of batches in each epoch')
    parser.add_argument('--nepochs', type=int, default=1000,
                        help='the number of epochs to train')
    parser.add_argument('--optim', type=str, default='sgd',
                        help='optimization method: sgd | adagrad')
    parser.add_argument('--lr-warmup', type=int, default=0,
                        help='linearly increase LR from 0 during first X updates')
    parser.add_argument('--grad-clip', type=float, default=0,
                        help='[Adagrad only] clip gradient of each module by a given value')
    # data related
    parser.add_argument('--data', type=str, default='',
                        help='data location (must contain train.txt, valid.txt and test.txt)')
    # plotting
    parser.add_argument('--plot', action='store_true', default=False,
                        help='plot in visdom')
    parser.add_argument('--plot-env', type=str, default='main',
                        help='plot env name')
    parser.add_argument('--plot-host', type=str, default='http://localhost',
                        help='visdom host name')
    # misc
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--checkpoint', type=str, default='',
                        help='path to save/load model')
    parser.add_argument('--checkpoint-freq', type=int, default=0,
                        help='keep a copy of model every K epochs (0 means keep only the last)')
    parser.add_argument('--full-test', action='store_true', default=False,
                        help='do testing on whole validation and test data')
    parser.add_argument('--distributed', action='store_true', default=False,
                        help='distributed training')
    parser.add_argument('--local_rank', type=int, default=0,
                        help='used in distributed training')
    parser.add_argument('--submitit', action='store_true', default=False,
                        help='using submitit')
    parser.add_argument('--dist-init', type=str, default='',
                        help='distributed training')
    attn_span.add_args(parser)
    return parser


# TODO: remove
class SubmititMain:
    def __call__(self, args):
        main(args)

    def checkpoint(self, args):
        return submitit.helpers.DelayedSubmission(self, args)


def main(args):
    print(args)

    # TODO: remove
    args = copy.deepcopy(args) # important for requeue!!! don't change original args

    use_cuda = not args.no_cuda and torch.cuda.is_available()

    if args.distributed:
        if args.submitit:
            # TODO: remove
            job_env = submitit.JobEnvironment()
            args.local_rank = job_env.local_rank
            args.rank = job_env.global_rank
            args.world_size = job_env.num_tasks
            torch.distributed.init_process_group(backend='nccl', init_method=args.dist_init, rank=job_env.global_rank, world_size=job_env.num_tasks)
        else:
            torch.distributed.init_process_group(backend='nccl', init_method='env://')
            args.rank = torch.distributed.get_rank()
            args.world_size = torch.distributed.get_world_size()
        torch.cuda.set_device(args.local_rank)

    device = torch.device("cuda" if use_cuda else "cpu")
    dtype = torch.float32
>>>>>>> master

from config import PARAMS_CONFIG
from workflow import (
    set_up_env,
    get_device,
    get_train_val_test_data,
    get_vocab_size,
    get_model,
    update_optim_params,
    get_optimizer_and_scheduler,
    train)


def _parse_args(params_config, args):
    parser = argparse.ArgumentParser()
    for params_category in params_config:  # e.g., 'model_params'
        for param_flag, param_config in params_config[params_category].items():
            # e.g., param_flag = '--block-sz'
            parser.add_argument(param_flag, **param_config)
    return parser.parse_args(args)


def get_params(params_config, args=None):
    namespace = _parse_args(params_config, args)
    return {
        params_category: {
            param_config['dest']:
                namespace.__getattribute__(param_config['dest'])
            for param_config in params_config[params_category].values()
        }
        for params_category in params_config
    }


def launch(env_params,
           model_params,
           attn_span_params,
           optim_params,
           data_params,
           plotter_params,
           trainer_params,
           *args,
           **kwargs):
    # ENVIRONMENT
    set_up_env(env_params)
    device = get_device(env_params)

    # DATA
    train_data, val_data, test_data = get_train_val_test_data(
        data_params=data_params,
        env_params=env_params,
        optim_params=optim_params,
        device=device)
    vocab_size = get_vocab_size(data_params)

    # MODEL
<<<<<<< HEAD
    model = get_model(
        model_params=model_params,
        attn_span_params=attn_span_params,
        env_params=env_params,
        device=device,
        vocab_size=vocab_size)

    # OPTIMIZER AND SCHEDULER
    update_optim_params(
        optim_params=optim_params, env_params=env_params)
    optimizer, scheduler = get_optimizer_and_scheduler(
        model=model, optim_params=optim_params)

    # train
    train(trainer_params=trainer_params,
          env_params=env_params,
          model_params=model_params,
          attn_span_params=attn_span_params,
          optim_params=optim_params,
          plotter_params=plotter_params,
          device=device,
          model=model,
          optimizer=optimizer,
          scheduler=scheduler,
          train_data=train_data,
          val_data=val_data,
          test_data=test_data)
=======
    model = TransformerSeq(args)

    if args.distributed:
        model = model.to(device, dtype=dtype)
        model = nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank)
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
    if args.optim == 'sgd':
        optimizer = optim.SGD(params, lr=args.lr, momentum=args.momentum)
    elif args.optim == 'adagrad':
        optimizer = Adagrad(params, lr=args.lr, grad_clip=args.grad_clip)
    else:
        raise RuntimeError('wrong optim mode!')
    if args.lr_warmup > 0:
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda ep: min(1, ep/args.lr_warmup))
    else:
        scheduler = None

    ep_init = checkpoint.load(args, model, optimizer, logger, scheduler)

    # current position in the data. 0=train 1=valid 2=test
    data_pos = [0 for _ in range(3)]
    # initialize the cache of hidden states with zeros
    hid_cache = [[torch.zeros(args.batch_sz, attn_span.get_cache_size(l.attn.attn), args.hid_sz).to(device, dtype=dtype) for l in model.module.layers] for _ in range(3)]

    if args.full_test:
        with torch.no_grad():
            stat_val, data_pos[1], hid_cache[1] = train(args, model, optimizer, scheduler, val_data,
                test_only=True, train_pos=data_pos[1], h_cache=hid_cache[1])
            print('val: {:.3f}bpc'.format(stat_val['loss']/math.log(2)))

            stat_test, data_pos[2], hid_cache[2] = train(args, model, optimizer, scheduler, test_data,
                test_only=True, train_pos=data_pos[2], h_cache=hid_cache[2])
            print('test: {:.3f}bpc'.format(stat_test['loss']/math.log(2)))
        return

    for ep in range(ep_init, args.nepochs):
        t_sta = time.time()
        args.ep = ep
        stat_train, data_pos[0], hid_cache[0] = train(args, model, optimizer, scheduler, train_data,
            train_pos=data_pos[0], h_cache=hid_cache[0])
        elapsed = 1000 * (time.time() - t_sta) / args.nbatches
        with torch.no_grad():
            stat_val, data_pos[1], hid_cache[1] = train(args, model, optimizer, scheduler, val_data,
                test_only=True, train_pos=data_pos[1], h_cache=hid_cache[1])

        if args.distributed:
            # collect results from all the workers
            X = torch.zeros(2).to(device)
            X[0] = stat_train['loss']
            X[1] = stat_val['loss']
            torch.distributed.reduce(X, 0)
            if args.rank == 0:
                stat_train['loss'] = X[0] / args.world_size
                stat_val['loss'] = X[1] / args.world_size
            else:
                continue

        if args.attn_span_loss > 0:
            attn_span.log(args, model, logger, stat_train)

        logger.step(args, stat_train, stat_val, elapsed)
        checkpoint.save(args, model, optimizer, logger, scheduler)
>>>>>>> master


if __name__ == '__main__':
    launch(**get_params(params_config=PARAMS_CONFIG, args=None))
