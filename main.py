#!/usr/bin/env python3

from utils import params

from workflow import (
    env_step,
    model_step,
    optim_step,
    plot_step,
    data_step,
    trainer_step)

from config import PARAMS_CONFIG


def launch(env_params,
           model_params,
           attn_span_params,
           optim_params,
           data_params,
           plot_params,
           trainer_params,
           *args,
           **kwargs):
    # set-up environment
    env_step.set_up_env(env_params)
    device = env_step.get_device(env_params)

    # get data
    train_data, val_data, test_data = data_step.get_train_val_test_data(
        data_params=data_params, env_params=env_params, device=device)
    vocab_size = data_step.get_vocab_size(data_params)

    # create model
    model = model_step.get_model(
        model_params=model_params,
        attn_span_params=attn_span_params,
        env_params=env_params,
        device=device,
        vocab_size=vocab_size)

    # create optimizer and scheduler
    optim_step.update_optim_params(
        optim_params=optim_params, env_params=env_params)
    optimizer, scheduler = optim_step.get_optimizer_and_scheduler(
        model=model, optim_params=optim_params)

    # create plotter
    plotter = plot_step.get_plotter(plot_params)

    # train
    trainer_step.train(
        trainer_params=trainer_params,
        env_params=env_params,
        model_params=model_params,
        attn_span_params=attn_span_params,
        optim_params=optim_params,
        plot_params=plot_params,
        device=device,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        plotter=plotter,
        train_data=train_data,
        val_data=val_data,
        test_data=test_data)


if __name__ == '__main__':
    launch(**params.get_params(params_config=PARAMS_CONFIG, args=None))
