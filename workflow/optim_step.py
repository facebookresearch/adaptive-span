#!/usr/bin/env python3

import torch.optim as optim

# TODO: review import statements
from utils.adagrad import Adagrad


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
                   weight_decay: float,
                   momentum: float,
                   grad_clip: float,
                   *args,
                   **kwargs):
    if optim == 'sgd':
        return optim.SGD(_get_grad_requiring_params(model),
                         lr=lr,
                         weight_decay=weight_decay,
                         momentum=momentum)
    elif optim == 'adagrad':
        return Adagrad(_get_grad_requiring_params(model),
                       lr=lr,
                       weight_decay=weight_decay,
                       grad_clip=grad_clip)
    else:
        raise RuntimeError("wrong type of optimizer "
                           "- must be 'sgd' or 'adagrad")


def _get_scheduler(optimizer, lr_warmup: int, *args, **kwargs):
    if lr_warmup > 0:
        return optim.lr_scheduler.LambdaLR(
            optimizer, lambda ep: min(1, ep / lr_warmup))
    return None


def update_optim_params(optim_params, env_params):
    optim_params['batch_size'] //= env_params['world_size']


def get_optimizer_and_scheduler(model, optim_params):
    optimizer = _get_optimizer(model=model, **optim_params)
    scheduler = _get_scheduler(optimizer=optimizer, **optim_params)
    return optimizer, scheduler
