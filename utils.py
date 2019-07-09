# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

#!/usr/bin/env python3

import os
import math
import argparse

import torch
from adagrad_with_grad_clip import AdagradWithGradClip


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


##############################################################################
# ENVIRONMENT
##############################################################################

def _torch_distributed_init_process_group(local_rank):
    torch.distributed.init_process_group(
        backend='nccl',
        init_method='env://'
    )
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    print('my rank={} local_rank={}'.format(rank, local_rank))
    torch.cuda.set_device(local_rank)
    return {
        'rank': rank,
        'world_size': world_size,
    }

def set_up_env(env_params):
    assert torch.cuda.is_available()
    if env_params['distributed']:
        env_params.update(
            _torch_distributed_init_process_group(
                local_rank=env_params['local_rank']))
    env_params['device'] = torch.device('cuda')


##############################################################################
# OPTIMIZER AND SCHEDULER
##############################################################################

def _get_grad_requiring_params(model):
    nb_parameters = 0
    grad_requiring_params = []
    for param in model.parameters():
        if param.requires_grad:
            nb_parameters += param.numel()
            grad_requiring_params.append(param)
    print('nb_parameters={:.2f}M'.format(nb_parameters / 1e6))
    return grad_requiring_params


def _get_optimizer(model,
                   optim,
                   lr: float,
                   momentum: float,
                   grad_clip: float):
    if optim == 'sgd':
        return torch.optim.SGD(_get_grad_requiring_params(model),
                               lr=lr,
                               momentum=momentum)
    elif optim == 'adagrad':
        return AdagradWithGradClip(_get_grad_requiring_params(model),
                                   lr=lr,
                                   grad_clip=grad_clip)
    else:
        raise RuntimeError("wrong type of optimizer "
                           "- must be 'sgd' or 'adagrad")


def _get_scheduler(optimizer, lr_warmup):
    if lr_warmup > 0:
        return torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda ep: min(1, ep / lr_warmup))
    return None


def get_optimizer_and_scheduler(model, optim_params):
    optimizer = _get_optimizer(model=model,
                               optim=optim_params['optim'],
                               lr=optim_params['lr'],
                               momentum=optim_params['momentum'],
                               grad_clip=optim_params['grad_clip'])
    scheduler = _get_scheduler(optimizer=optimizer,
                               lr_warmup=optim_params['lr_warmup'])
    return optimizer, scheduler


##############################################################################
# CHECKPOINT
##############################################################################

def _load_checkpoint(checkpoint_path, model, optimizer, scheduler, logger,
                     distributed):
    print('loading from a checkpoint at {}'.format(checkpoint_path))
    if distributed:
        # the model is saved from gpu0 so we need to map it to CPU first
        checkpoint_state = torch.load(
            checkpoint_path, map_location=lambda storage, loc: storage)
    else:
        checkpoint_state = torch.load(checkpoint_path)
    iter_init = checkpoint_state['iter_no'] + 1  # next iteration
    model.load_state_dict(checkpoint_state['model'])
    optimizer.load_state_dict(checkpoint_state['optimizer'])
    logger.load_state_dict(checkpoint_state['logger'])
    if 'scheduler_iter' in checkpoint_state:
        # we only need the step count
        scheduler.step(checkpoint_state['scheduler_iter'])
    return iter_init


def load_checkpoint(checkpoint_path, model, optimizer, scheduler, logger,
                    distributed):
    if checkpoint_path and os.path.exists(checkpoint_path):
        return _load_checkpoint(checkpoint_path=checkpoint_path,
                                model=model,
                                optimizer=optimizer,
                                scheduler=scheduler,
                                logger=logger,
                                distributed=distributed)
    return 0


def save_checkpoint(checkpoint_path, iter_no, model,
                    optimizer, scheduler, logger):
    if checkpoint_path:
        checkpoint_state = {
            'iter_no': iter_no,  # last completed iteration
            'model': model.state_dict(),
            'logger': logger.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        if scheduler is not None:
            checkpoint_state['scheduler_iter'] = scheduler.last_epoch
        torch.save(checkpoint_state, checkpoint_path)


##############################################################################
# LOGGER
##############################################################################

class Logger:
    def __init__(self):
        self._state_dict = dict()

    def load_state_dict(self, state_dict):
        self._state_dict = state_dict

    def state_dict(self):
        return self._state_dict

    def _log(self, title, value):
        if title not in self._state_dict:
            self._state_dict[title] = []
        self._state_dict[title].append(value)

    def log_iter(self, iter_no, nb_batches_per_iter, loss_train, loss_val,
                 elapsed, model):
        step = (iter_no + 1) * nb_batches_per_iter
        train_bpc = float(loss_train / math.log(2))
        val_bpc = float(loss_val / math.log(2))
        msg = 'steps: {}'.format(step)
        msg += '\ttrain: {:.3f}bpc\tval: {:.3f}bpc'.format(train_bpc, val_bpc)
        msg += '\tms/batch: {:.1f}'.format(elapsed)
        self._log(title='step', value=step)
        self._log(title='train_bpc', value=train_bpc)
        self._log(title='val_bpc', value=val_bpc)

        if model.module.layers[0].attn.attn.adapt_span_enabled:
            avg_spans = []
            max_spans = []
            for layer in model.module.layers:
                avg_spans.append(
                    layer.attn.attn.adaptive_span.get_current_avg_span())
                max_spans.append(
                    layer.attn.attn.adaptive_span.get_current_max_span())
            span_avg = float(sum(avg_spans)) / len(avg_spans)
            span_max = float(max(max_spans))
            self._log('span_avg', span_avg)
            self._log('span_max', span_max)
            msg += "\tspan_avg: {:.0f}\tspan_max: {:.0f}".format(span_avg, span_max)

        print(msg)
