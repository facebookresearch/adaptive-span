import os
import shutil
import torch
import torch.nn as nn


def load_path(args, model, optimizer, logger, scheduler):
    path = args.checkpoint
    print('loading from ' + path)
    if args.distributed:
        # the model is saved from gpu0 so we need to map it to CPU first
        f = torch.load(path, map_location=lambda storage, loc: storage)
    else:
        f = torch.load(path)
    ep_init = f['epoch']
    model.load_state_dict(f['model'])
    logger.set_state(f['logger'])
    optimizer.load_state_dict(f['optimizer'])
    if 'scheduler_epoch' in f:
        # scheduler.load_state_dict(f['scheduler'])
        scheduler.step(f['scheduler_epoch'])

    return ep_init


def load(args, model, optimizer, logger, scheduler):
    ep_init = 0
    if args.checkpoint != '' and os.path.exists(args.checkpoint):
        try:
            ep_init = load_path(args, model, optimizer, logger, scheduler)
        except:
            print('load failed')
            # try the backup checkpoint
            if os.path.exists(args.checkpoint + '.bak'):
                try:
                    ep_init = load_path(args + '.bak', model, optimizer, logger, scheduler)
                except:
                    print('load failed')

    return ep_init


def save(args, model, optimizer, logger, scheduler):
    if args.checkpoint != '' and args.load_only == False:
        if os.path.exists(args.checkpoint):
            if args.checkpoint_freq > 0 and args.ep > 0 and args.ep % args.checkpoint_freq == 0:
                try:
                    shutil.copyfile(args.checkpoint, args.checkpoint + '.' + str(args.ep))
                except:
                    print('save copy failed')
            # make a backup in case this save fails
            os.replace(args.checkpoint, args.checkpoint + '.bak')
        f = dict()
        f['epoch'] = args.ep + 1
        f['model'] = model.state_dict()
        f['logger'] = logger.get_state()
        f['optimizer'] = optimizer.state_dict()
        if scheduler is not None:
            f['scheduler_epoch'] = scheduler.last_epoch
        torch.save(f, args.checkpoint)
