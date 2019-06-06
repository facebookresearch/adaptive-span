#!/usr/bin/env python3


# command-line arguments with their default values

PARAMS_CONFIG = {
    # env-specific
    'env_params': {
        '--no-cuda': {
            'action': 'store_true',
            'default': False,
            'help': 'disables CUDA training',
            'dest': 'no_cuda'
        },
        '--distributed': {
            'action': 'store_true',
            'default': False,
            'help': 'distributed training',
            'dest': 'distributed'
        },
        # TODO: remove submitit
        '--submitit': {
            'action': 'store_true',
            'default': False,
            'help': 'using submitit',
            'dest': 'submitit_enabled'
        },
        '--dist-init': {
            'type': str,
            'default': '',
            'help': 'distributed training',
            'dest': 'dist_init'
        },
        # TODO: what is the goal of this?
        # can we set it if not distributed? if not submitit?
        '--local-rank': {
            'type': int,
            'default': 0,
            'help': 'used in distributed training',
            'dest': 'local_rank'
        },
    },
    # data-specific
    'data_params': {
        '--data': {
            'type': str,
            'default': '',
            'help': 'data location '
                    '(must contain train.txt, valid.txt and test.txt)',
            'dest': 'data_path'
        },
    },
    # model-specific
    'model_params': {
        '--hid-sz': {
            'type': int,
            'default': 256,
            'help': 'hidden size (i.e. model size)',
            'dest': 'hidden_size'
        },
        '--inner-hid-sz': {
            'type': int,
            'default': 1024,
            'help': 'inner hidden size of FF layer',
            'dest': 'inner_hidden_size'
        },
        '--nlayers': {
            'type': int,
            'default': 8,
            'help': 'number of layers',
            'dest': 'nb_layers'
        },
        '--block-sz': {
            'type': int,
            'default': 64,
            'help': 'block size '
                    '(the length of sequence to process in parallel)',
            'dest': 'block_size'
        },
        '--nheads': {
            'type': int,
            'default': 2,
            'help': 'number of attention heads',
            'dest': 'nb_heads'
        },
        '--dropout': {
            'type': float,
            'default': 0.2,
            'help': 'dropout rate of ReLU and attention',
            'dest': 'dropout'
        },
    },
    # attention-span-specific - refinement on model-specific params
    'attn_span_params': {
        '--attn-span-lim': {
            'type': int,
            'default': 32,
            'help': 'length of the attention span',
            'dest': 'attn_span_lim'
        },
        # TODO: why condition attn_span_loss while there is attn_span_enabled?
        '--attn-span': {
            'action': 'store_true',
            'default': False,
            'help': 'learn attention span',
            'dest': 'attn_span_enabled'
        },
        '--attn-span-loss': {
            'type': float,
            'default': 0,
            'help': 'learn attention span',
            'dest': 'attn_span_loss'
        },
        '--attn-span-len': {
            'type': int,
            'default': 32,
            'help': 'learn attention span',
            'dest': 'attn_span_len'
        },
        '--attn-span-init': {
            'type': float,
            'default': 0,
            'help': 'initial attention span ratio value',
            'dest': 'attn_span_init'
        },
        '--attn-span-cache': {
            'action': 'store_true',
            'default': False,
            'help': 'change cache size',
            'dest': 'attn_span_cache_enabled'
        },
    },
    # optimization-specific
    'optim_params': {
        '--lr': {
            'type': float,
            'default': 0.03,
            'help': 'learning rate',
            'dest': 'lr'
        },
        '--momentum': {
            'type': float,
            'default': 0.9,
            'help': 'SGD momentum',
            'dest': 'momentum'
        },
        '--batch-sz': {
            'type': int,
            'default': 64,
            'help': 'batch size',
            'dest': 'batch_size'
        },
        '--nbatches': {
            'type': int,
            'default': 1000,
            'help': 'number of batches in each iteration',
            'dest': 'nb_batches'
        },
        '--niter': {
            'type': int,
            'default': 1000,
            'help': 'number of iterations to train',
            'dest': 'nb_iter'
        },
        '--optim': {
            'type': str,
            'default': 'sgd',
            'help': 'optimization method: sgd | adagrad',
            'dest': 'optim'
        },
        '--lr-warmup': {
            'type': int,
            'default': 0,
            'help': 'linearly increase LR from 0 '
                    'during first lr_warmup updates',
            'dest': 'lr_warmup'
        },
        '--grad-clip': {
            'type': float,
            'default': 0,
            'help': '[only works with adagrad!] '
                    'clip gradient of each parameter by a given '
                    'value',
            'dest': 'grad_clip'
        },
    },
    # trainer-specific
    'trainer_params': {
        '--checkpoint': {
            'type': str,
            'default': '',
            'help': 'path to save/load model',
            'dest': 'checkpoint_path'
        },
        '--checkpoint-freq': {
            'type': int,
            'default': 0,
            'help': 'keep a copy of model every checkpoint_freq iterations '
                    '(0 means keep only the last)',
            'dest': 'checkpoint_freq'
        },
        '--load-only': {
            'action': 'store_true',
            'default': False,
            'help': 'do not save to checkpoint',
            'dest': 'load_only'
        },
        '--full-test': {
            'action': 'store_true',
            'default': False,
            'help': 'do testing on whole validation and test data',
            'dest': 'full_test'
        },
    },
    # plotter-specific
    'plotter_params': {
        '--plot': {
            'action': 'store_true',
            'default': False,
            'help': 'plot in visdom',
            'dest': 'plot_enabled'
        },
        '--plot-env': {
            'type': str,
            'default': 'main',
            'help': 'plot env name',
            'dest': 'plot_env'
        },
        '--plot-host': {
            'type': str,
            'default': 'http://localhost',
            'help': 'visdom host name',
            'dest': 'plot_host'
        },
    },
}
