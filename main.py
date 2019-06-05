#!/usr/bin/env python3

from utils import params

from workflow import (
    compute_step,
    model_step,
    optim_step,
    plot_step,
    data_step,
    trainer_step)

from config import PARAMS_CONFIG


def launch(compute_params,
           model_params,
           attn_span_params,
           optim_params,
           data_params,
           plot_params,
           trainer_params,
           *args,
           **kwargs):
    # initialize torch with computation parameters
    compute_step.initialize_computation_and_update_params(compute_params)
    device = compute_step.get_device(compute_params)

    # create model
    model = model_step.get_model(
        model_params=model_params,
        attn_span_params=attn_span_params,
        compute_params=compute_params,
        device=device)

    # create optimizer and scheduler
    optim_step.update_optim_params(
        optim_params=optim_params, compute_params=compute_params)
    optimizer, scheduler = optim_step.get_optimizer_and_scheduler(
        model=model, optim_params=optim_params)

    # create plotter
    plotter = plot_step.get_plotter(plot_params)

    # get data
    train_data, val_data, test_data = data_step.get_train_val_test_data(
        data_params=data_params, compute_params=compute_params, device=device)

    # train
    trainer_step.train(
        trainer_params=trainer_params,
        compute_params=compute_params,
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
