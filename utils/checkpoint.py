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
        f = torch.load(checkpoint_path,
                       map_location=lambda storage, loc: storage)
    else:
        f = torch.load(checkpoint_path)
    iter_init = f['iter_no']
    model.load_state_dict(f['model'])
    plotter.set_state(f['plotter'])
    optimizer.load_state_dict(f['optimizer'])
    if 'scheduler_iter' in f:
        # scheduler.load_state_dict(f['scheduler'])
        scheduler.step(f['scheduler_iter'])

    return ep_init


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


def save_checkpoint(checkpoint_path,
                    checkpoint_freq,
                    iter_no,
                    model,
                    optimizer,
                    plotter,
                    scheduler,
                    load_only):
    if checkpoint_path and not load_only:
        if os.path.exists(checkpoint_path):
            if checkpoint_freq > 0 and iter_no > 0 and iter_no % checkpoint_freq == 0:
                shutil.copyfile(
                    checkpoint_path, checkpoint_path + '.' + str(ep))
        f = dict()
        f['iter_no'] = iter_no + 1
        f['model'] = model.state_dict()
        f['plotter'] = plotter.get_state()
        f['optimizer'] = optimizer.state_dict()
        if scheduler is not None:
            f['scheduler_iter'] = scheduler.last_epoch
        torch.save(f, checkpoint_path)
