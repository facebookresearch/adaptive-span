#!/usr/bin/env python3

SUBMIT_PARAMS_CONFIG = {
    # FAIR-specific params
    'submit_params': {
        '--name': {
            'type': str,
            'default': '',
            'help': '',
            'dest': 'name'
        },
        '--folder': {
            'type': str,
            'default': '',
            'help': '',
            'dest': 'folder'
        },
        '--partition': {
            'type': str,
            'default': 'learnfair',
            'help': '',
            'dest': 'partition'
        },
        '--ngpus': {
            'type': int,
            'default': 8,
            'help': '',
            'dest': 'nb_gpus'
        },
        '--nodes': {
            'type': int,
            'default': 1,
            'help': '',
            'dest': 'nodes'
        },
        '--constraint': {
            'type': str,
            'default': 'volta32gb',
            'help': '',
            'dest': 'constraint'
        },
        '--args': {
            'type': str,
            'default': '',
            'help': 'args for the actual function',
            'dest': 'args'
        },
    }
}

EXECUTOR_MEM_GB = 128
EXECUTOR_CPUS_PER_TASK = 2
EXECUTOR_TIMEOUT_MIN = 4320
EXECUTOR_SIGNAL_DELAY_S = 120
