# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

#!/usr/bin/env python3

import math
import time

import torch

from config import PARAMS_CONFIG
from data import get_train_val_test_data
from models import TransformerSeq
from trainer import train_iteration, full_eval
from utils import (
    get_params,
    set_up_env,
    get_optimizer_and_scheduler,
    load_checkpoint,
    save_checkpoint,
    Logger)


def launch(env_params,
           model_params,
           adapt_io_params,
           adapt_span_params,
           pers_mem_params,
           optim_params,
           data_params,
           trainer_params):
    # ENVIRONMENT (device, distributed, etc.)
    set_up_env(env_params)
    device = env_params['device']
    distributed = env_params['distributed']

    if distributed == False or env_params['rank'] == 0:
        print('model_params:\t', model_params)
        print('optim_params:\t', optim_params)
        print('data_params:\t', data_params)
        print('trainer_params:\t', trainer_params)
        print('adapt_io_params:\t', adapt_io_params)
        print('adapt_span_params:\t', adapt_span_params)
        print('pers_mem_params:\t', pers_mem_params)

    # DATA
    train_data, val_data, test_data = get_train_val_test_data(
        data_params=data_params,
        env_params=env_params,
        batch_size=trainer_params['batch_size'],
        device=device,
        sort_dict=adapt_io_params['adapt_io_enabled'])

    # MODEL
    model = TransformerSeq(
        vocab_size=data_params['vocab_size'], **model_params,
        adapt_io_params=adapt_io_params,
        adapt_span_params=adapt_span_params,
        pers_mem_params=pers_mem_params)
    if distributed:
        local_rank = env_params['local_rank']
        model = model.to(device)
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank)
    else:
        model = torch.nn.DataParallel(model)
        model = model.to(device)

    # OPTIMIZER AND SCHEDULER
    optimizer, scheduler = get_optimizer_and_scheduler(
        model=model, optim_params=optim_params)

    # create logger
    logger = Logger(data_params['data_unit'])

    # resume training from last checkpoint if exists
    iter_init = load_checkpoint(
        trainer_params['checkpoint_path'], model, optimizer, scheduler,
        logger, distributed)

    if trainer_params['full_eval_mode']:
        # evaluate the model on test data
        with torch.no_grad():
            loss_val = full_eval(model, optimizer, scheduler, val_data,
                                 model_params['block_size'],
                                 model_params['hidden_size'])
            loss_test = full_eval(model, optimizer, scheduler, test_data,
                                  model_params['block_size'],
                                  model_params['hidden_size'])
            if distributed:
                # collect results into rank0
                stats = torch.tensor(
                    [loss_val, loss_test]).to(device)
                torch.distributed.reduce(stats, 0)
                if env_params['rank'] == 0:
                    loss_val = stats[0] / env_params['world_size']
                    loss_test = stats[1] / env_params['world_size']
                else:
                    return

            if data_params['data_unit'] == 'bpc':
                print('val: {:.3f}bpc'.format(loss_val / math.log(2)))
                print('test: {:.3f}bpc'.format(loss_test / math.log(2)))
            else:
                print('val: {:.2f}ppl'.format(math.exp(loss_val)))
                print('test: {:.2f}ppl'.format(math.exp(loss_test)))
        return

    # position of current batch
    data_pos = [0] * 2
    # initialize caches for train and valid
    hid_cache = [[
        torch.zeros(
            train_data.size(0),
            layer.attn.attn.get_cache_size(),
            model_params['hidden_size']).to(device)
        for layer in model.module.layers] for _ in range(2)]

    nb_batches_per_iter = trainer_params['nb_batches_per_iter']
    for iter_no in range(iter_init, trainer_params['nb_iter']):
        t_sta = time.time()
        loss_train, data_pos[0], hid_cache[0] = train_iteration(
            model, optimizer, scheduler, train_data, nb_batches_per_iter,
            model_params['block_size'], False, data_pos[0], hid_cache[0],
            trainer_params['batch_split'])
        elapsed = 1000 * (time.time() - t_sta) / nb_batches_per_iter
        with torch.no_grad():
            loss_val, data_pos[1], hid_cache[1] = train_iteration(
                model, optimizer, scheduler, val_data, nb_batches_per_iter,
                model_params['block_size'], True, data_pos[1], hid_cache[1],
                trainer_params['batch_split'])

        if distributed:
            # collect results into rank0
            stats = torch.tensor(
                [loss_train, loss_val]).to(device)
            torch.distributed.reduce(stats, 0)
            if env_params['rank'] == 0:
                loss_train = stats[0] / env_params['world_size']
                loss_val = stats[1] / env_params['world_size']
            else:
                continue

        logger.log_iter(iter_no, nb_batches_per_iter, loss_train,
                        loss_val, elapsed, model)
        save_checkpoint(trainer_params['checkpoint_path'],
                        iter_no, model, optimizer, scheduler, logger)


if __name__ == '__main__':
    launch(**get_params(params_config=PARAMS_CONFIG))
