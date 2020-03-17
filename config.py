# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

#!/usr/bin/env python3

# command-line arguments with their default values

PARAMS_CONFIG = {
    # env-specific
    'env_params': {
        '--distributed': {
            'action': 'store_true',
            'default': False,
            'help': 'enable distributed training.'
                    '(otherwise will use all available GPUs with dataparallel)',
            'dest': 'distributed'
        },
        '--local_rank': {
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
            'default': 'data/text8',
            'help': 'data location '
                    '(must contain train.txt, valid.txt and test.txt)',
            'dest': 'data_path'
        },
        '--data-unit': {
            'type': str,
            'default': 'bpc',
            'choices': ['bpc', 'ppl'],
            'help': 'loss unit to log',
            'dest': 'data_unit'
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
            'help': 'number of self-attention heads',
            'dest': 'nb_heads'
        },
        '--attn-span': {
            'type': int,
            'default': 32,
            'help': 'length of the attention span',
            'dest': 'attn_span'
        },
        '--dropout': {
            'type': float,
            'default': 0.2,
            'help': 'dropout rate of ReLU and attention',
            'dest': 'dropout'
        },
        '--emb-dropout': {
            'type': float,
            'default': 0.,
            'help': 'the dropout rate applied on I/O embeddings',
            'dest': 'emb_dropout'
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
                    'clip gradient of each module parameters by a given '
                    'value',
            'dest': 'grad_clip'
        },
    },
    # trainer-specific
    'trainer_params': {
        '--batch-sz': {
            'type': int,
            'default': 64,
            'help': 'batch size',
            'dest': 'batch_size'
        },
        '--batch-split': {
            'type': int,
            'default': 1,
            'help': 'split a batch into smaller parts to fit in GPU memory',
            'dest': 'batch_split'
        },
        '--nbatches': {
            'type': int,
            'default': 1000,
            'help': 'number of batches in each iteration',
            'dest': 'nb_batches_per_iter'
        },
        '--niter': {
            'type': int,
            'default': 1000,
            'help': 'number of iterations to train',
            'dest': 'nb_iter'
        },
        '--checkpoint': {
            'type': str,
            'default': '',
            'help': 'path to save/load model',
            'dest': 'checkpoint_path'
        },
        '--full-eval-mode': {
            'action': 'store_true',
            'default': False,
            'help': 'do evaluation on the whole validation and the test data',
            'dest': 'full_eval_mode'
        },
    },
    # adaptive I/O specific params
    'adapt_io_params': {
        '--adapt-io': {
            'action': 'store_true',
            'default': False,
            'help': 'enable adaptive input and output representations',
            'dest': 'adapt_io_enabled'
        },
        '--adapt-io-tied': {
            'action': 'store_true',
            'default': False,
            'help': 'tie the input parameters with the output parameters',
            'dest': 'adapt_io_tied'
        },
        '--adapt-io-divval': {
            'type': int,
            'default': 4,
            'help': 'dimension division value',
            'dest': 'adapt_io_divval'
        },
        '--adapt-io-cutoffs': {
            'type': int,
            'default': [20000, 40000, 200000],
            'help': 'cutoffs values',
            'dest': 'adapt_io_cutoffs'
        },
    },
    # adaptive attention span specific params
    'adapt_span_params': {
        '--adapt-span': {
            'action': 'store_true',
            'default': False,
            'help': 'enable adaptive attention span',
            'dest': 'adapt_span_enabled'
        },
        '--adapt-span-loss': {
            'type': float,
            'default': 0,
            'help': 'the loss coefficient for span lengths',
            'dest': 'adapt_span_loss'
        },
        '--adapt-span-ramp': {
            'type': int,
            'default': 32,
            'help': 'ramp length of the soft masking function',
            'dest': 'adapt_span_ramp'
        },
        '--adapt-span-init': {
            'type': float,
            'default': 0,
            'help': 'initial attention span ratio',
            'dest': 'adapt_span_init'
        },
        '--adapt-span-cache': {
            'action': 'store_true',
            'default': False,
            'help': 'adapt cache size as well to reduce memory usage',
            'dest': 'adapt_span_cache'
        },
    },
    # persistent memory specific params
    'pers_mem_params': {
        '--pers-mem-size': {
            'type': int,
            'default': 0,
            'help': 'the number of persistent memory vectors',
            'dest': 'pers_mem_size'
        },
    },
}
