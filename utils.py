#!/usr/bin/env python3

import os

import torch


##############################################################################
# CHECKPOINT
##############################################################################

def _load_checkpoint(checkpoint_path,
                     model,
                     optimizer,
                     scheduler,
                     logger,
                     distributed):
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
        # # we only need the step count
        scheduler.step(checkpoint_state['scheduler_iter'])
    return iter_init


def load_checkpoint(checkpoint_path,
                    model,
                    optimizer,
                    scheduler,
                    logger,
                    distributed):
    if checkpoint_path and os.path.exists(checkpoint_path):
        return _load_checkpoint(checkpoint_path=checkpoint_path,
                                model=model,
                                optimizer=optimizer,
                                scheduler=scheduler,
                                logger=logger,
                                distributed=distributed)
    return 0


def is_checkpoint(iter_no, checkpoint_freq):
    return checkpoint_freq > 0 and (iter_no + 1) % checkpoint_freq == 0


def save_checkpoint(checkpoint_path,
                    iter_no,
                    model,
                    optimizer,
                    scheduler,
                    logger):
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

    def log(self, title, value):
        if title not in self._state_dict:
            self._state_dict[title] = []
        self._state_dict[title].append(value)

    def __contains__(self, title):
        return title in self._state_dict

    def __iter__(self):
        for title in self._state_dict:
            yield title

    def get_data(self, title):
        if title not in self:
            raise KeyError(title)
        return self._state_dict[title]


##############################################################################
# PLOTTER
##############################################################################

class Plotter:
    def __init__(self, plot_enabled, plot_env, plot_host, *args, **kwargs):
        self.plot_enabled = plot_enabled
        self.plot_env = plot_env
        if plot_enabled:
            import visdom
            self.vis = visdom.Visdom(
                env=plot_env, server=plot_host)

    def plot(self, title, Y, X=None):
        if self.plot_enabled:
            self.vis.line(X=X, Y=Y, win=title, opts={'title': title})

    def save(self):
        if self.plot_enabled:
            self.vis.save([self.plot_env])
