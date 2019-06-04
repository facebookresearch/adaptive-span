#!/usr/bin/env python3

import argparse


def _parse_args(params_config, args):
    parser = argparse.ArgumentParser()
    for params_category in params_config:  # e.g., 'model_params'
        for param_flag, param_config in params_config[params_category].items():
            # e.g., param_flag = '--block-sz'
            parser.add_argument(param_flag, **param_config)
    return parser.parse_args(args)


def get_params(params_config, args=None):
    namespace = _parse_args(params_config, args)

    # TODO: should return dataclasses instead of dict
    return {
        params_category: {
            param_config['dest']:
                namespace.__getattribute__(param_config['dest'])
            for param_config in params_config[params_category].values()
        }
        for params_category in params_config
    }
