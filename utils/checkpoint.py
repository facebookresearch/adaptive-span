#!/usr/bin/env python3

import os
import shutil

import torch


def _load_checkpoint(checkpoint_path,
                     model,
                     optimizer,
                     plotter,
                     scheduler,
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
    plotter.load_state_dict(checkpoint_state['plotter'])
    optimizer.load_state_dict(checkpoint_state['optimizer'])
    if 'scheduler_iter' in checkpoint_state:
        scheduler.step(checkpoint_state['scheduler_iter'])
    return iter_init


def load_checkpoint(checkpoint_path,
                    model,
                    optimizer,
                    scheduler,
                    plotter,
                    distributed) -> int:
    if checkpoint_path and os.path.exists(checkpoint_path):
        return _load_checkpoint(checkpoint_path=checkpoint_path,
                                model=model,
                                optimizer=optimizer,
                                plotter=plotter,
                                scheduler=scheduler,
                                distributed=distributed)
    return 0


def is_checkpoint(iter_no, checkpoint_freq):
    return (iter_no + 1) % checkpoint_freq == 0


def save_checkpoint(checkpoint_path,
                    iter_no,
                    model,
                    optimizer,
                    scheduler,
                    plotter):
        checkpoint_state = {
            'iter_no' = iter_no,  # last completed iteration
            'model' = model.state_dict(),
            'plotter' = plotter.state_dict(),
            'optimizer' = optimizer.state_dict(),
        }
        if scheduler is not None:
            checkpoint_state['scheduler_iter'] = scheduler.last_epoch
        torch.save(checkpoint_state, checkpoint_path)
