#!/usr/bin/env python3

import os

import torch


def _load_checkpoint(checkpoint_path,
                     model,
                     optimizer,
                     scheduler,
                     logger,
                     distributed) -> int:
    print('loading from ' + checkpoint_path)
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
        scheduler.step(checkpoint_state['scheduler_iter'])
    return iter_init


def load_checkpoint(checkpoint_path,
                    model,
                    optimizer,
                    scheduler,
                    logger,
                    distributed) -> int:
    if checkpoint_path and os.path.exists(checkpoint_path):
        return _load_checkpoint(checkpoint_path=checkpoint_path,
                                model=model,
                                optimizer=optimizer,
                                scheduler=scheduler,
                                logger=logger,
                                distributed=distributed)
    return 0


def save_checkpoint(checkpoint_path,
                    iter_no,
                    model,
                    optimizer,
                    scheduler,
                    logger):
        checkpoint_state = {
            'iter_no' = iter_no,  # last completed iteration
            'model' = model.state_dict(),
            'logger' = logger.state_dict(),
            'optimizer' = optimizer.state_dict(),
        }
        if scheduler is not None:
            checkpoint_state['scheduler_iter'] = scheduler.last_epoch
        torch.save(checkpoint_state, checkpoint_path)
