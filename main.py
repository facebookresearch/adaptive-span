#!/usr/bin/env python3

import argparse

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
    update_optim_params(
        optim_params=optim_params, env_params=env_params)

    # DATA
    train_data, val_data, test_data = get_train_val_test_data(
        data_params=data_params,
        env_params=env_params,
        optim_params=optim_params,
        device=device)
    vocab_size = get_vocab_size(data_params)

    # MODEL
    model = get_model(
        model_params=model_params,
        attn_span_params=attn_span_params,
        env_params=env_params,
        device=device,
        vocab_size=vocab_size)

    # OPTIMIZER AND SCHEDULER
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


if __name__ == '__main__':
    launch(**get_params(params_config=PARAMS_CONFIG, args=None))
