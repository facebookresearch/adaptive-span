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
            'dest': 'ngpus'
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
            'help': '',
            'dest': 'args'
        },
    }
}
