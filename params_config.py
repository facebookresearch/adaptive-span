#!/usr/bin/env python3


PARAMS_CONFIG = {
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
        '--attn-lim': {
            'type': int,
            'default': 32,
            'help': 'limit attention span',
            'dest': 'attn_lim'
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
            'help': 'number of batches in each epoch',
            'dest': 'nb_batches'
        },
        '--nepochs': {
            'type': int,
            'default': 1000,
            'help': 'number of epochs to train',
            'dest': 'nb_epochs'
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
            'help': 'linearly increase LR from 0 during K updates',
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
        '--wdecay': {
            'type': float,
            'default': 0,
            'help': 'weight decay',
            'dest': 'weight_decay'
        },
    },
    # data-specific
    'data_params': {
        '--data': {
            'type': str,
            'default': 0.03,
            # TODO: REMOVE DEFAULT!!!
            'help': '/private/home/sainbar/data/pennchar',
            'dest': 'data_path'
        },
    },
    # plot-specific
    'plot_params': {
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
    # computation-specific
    'compute_params': {
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
    }
}


    # computation related
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--distributed', action='store_true', default=False,
                        help='distributed training')
    parser.add_argument('--local-rank', type=int, default=0,
                        help='')
    parser.add_argument('--submitit', action='store_true', default=False,
                        help='using submitit')
    parser.add_argument('--dist-init', type=str, default='',
                        help='distributed training')
    # checkpoint related
    parser.add_argument('--checkpoint', type=str, default='',
                        help='path to save/load model')
    parser.add_argument('--checkpoint-freq', type=int, default=0,
                        help='how often to keep a copy')
    parser.add_argument('--load-only', action='store_true', default=False,
                        help='do not save to checkpoint')
    # test related
    parser.add_argument('--test-mode', action='store_true', default=False,
                        help='only do testing')
    parser.add_argument('--full-test', action='store_true', default=False,
                        help='do testing on whole data')
    # attention model related
    parser.add_argument('--attn-span', action='store_true', default=False,
                        help='learn attention span',
                        dest='attn_span')
    parser.add_argument('--attn-span-loss', type=float, default=0,
                        help='learn attention span',
                        dest='attn_span_loss')
    parser.add_argument('--attn-span-len', type=float, default=32,
                        help='learn attention span',
                        dest='attn_span_len')
    parser.add_argument('--attn-span-init', type=float, default=0,
                        help='initial attention span ratio value',
                        dest='attn_span_init')
    parser.add_argument('--attn-span-cache', action='store_true', default=False,
                        help='change cache size',
                        dest='attn_span_cache')
