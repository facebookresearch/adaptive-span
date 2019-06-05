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
    ep_init = f['epoch']
    model.load_state_dict(f['model'])
    plotter.set_state(f['plotter'])
    optimizer.load_state_dict(f['optimizer'])
    if 'scheduler_epoch' in f:
        # scheduler.load_state_dict(f['scheduler'])
        scheduler.step(f['scheduler_epoch'])

    return ep_init


def load_checkpoint(checkpoint_path,
                    model,
                    optimizer,
                    scheduler,
                    plotter,
                    distributed) -> int:
    if checkpoint_path and os.path.exists(checkpoint_path):
        try:
            return _load_checkpoint(checkpoint_path, model, optimizer,
                                    plotter, scheduler, distributed)
        # TODO: BE MORE SPECIFIC ABOUT THE EXCEPTION !!!
        except:
            print('load failed')
            # try the backup checkpoint
            if os.path.exists(checkpoint_path + '.bak'):
                try:
                    return _load_checkpoint(checkpoint_path + '.bak', model,
                                            optimizer, plotter, scheduler,
                                            distributed)
                # TODO: BE MORE SPECIFIC ABOUT THE EXCEPTION !!!
                except:
                    print('load failed')

    return 0


def save_checkpoint(checkpoint_path,
                    checkpoint_freq,
                    ep,
                    model,
                    optimizer,
                    plotter,
                    scheduler,
                    load_only):
    if checkpoint_path and not load_only:
        if os.path.exists(checkpoint_path):
            if checkpoint_freq > 0 and ep > 0 and ep % checkpoint_freq == 0:
                try:
                    shutil.copyfile(
                        checkpoint_path, checkpoint_path + '.' + str(ep))
                # TODO: BE MORE SPECIFIC ABOUT THE EXCEPTION !!!
                except:
                    print('save copy failed')
            # make a backup in case this save fails
            os.replace(checkpoint_path, checkpoint_path + '.bak')
        f = dict()
        f['epoch'] = ep + 1
        f['model'] = model.state_dict()
        f['plotter'] = plotter.get_state()
        f['optimizer'] = optimizer.state_dict()
        if scheduler is not None:
            f['scheduler_epoch'] = scheduler.last_epoch
        torch.save(f, checkpoint_path)
