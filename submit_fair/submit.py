#!/usr/bin/env python3

import os
import argparse
import uuid
from pathlib import Path

import submitit

# TODO: review import statements
from submit_params_config import *
from utils.params import get_params
from main import launch_from_cli
from params_config import PARAMS_CONFIG


class SubmitMyFunc:
    def __init__(self, func):
        self.func = func

    def __call__(self, args):
        self.call(args)

    def checkpoint(self, args):
        return submitit.helpers.DelayedSubmission(self, args)


submit_params = get_params(params_config=SUBMIT_PARAMS_CONFIG, args=None)
folder = Path(submit_params['folder'])
os.makedirs(str(folder), exist_ok=True)
init_file = folder / f"{uuid.uuid4().hex}_init"  # not used when nodes=1


job_args = submit_params['args'].split() + [
    '--submitit', '--dist-init', init_file.as_uri()]
job_params = get_params(params_config=PARAMS_CONFIG, args=job_args)


executor = submitit.AutoExecutor(folder=folder / "%j", max_num_timeout=10)
executor.update_parameters(
    mem_gb=EXECUTOR_MEM_GB,
    gpus_per_node=submit_params['nb_gpus'],
    tasks_per_node=submit_params['nb_gpus'],  # one task per GPU
    cpus_per_task=EXECUTOR_CPUS_PER_TASK,
    nodes=submit_params['nodes'],
    timeout_min=EXECUTOR_TIMEOUT_MIN,
    # Below are cluster dependent parameters
    partition=submit_params['partition'],
    signal_delay_s=EXECUTOR_SIGNAL_DELAY_S,
    constraint=submit_params['constraint'],
)
if submit_params['partition'] == 'priority':
    executor.update_parameters(
        comment='NIPS deadline May 23rd')

executor.update_parameters(name=submit_params['name'])
submitted_func = SubmitMyFunc(launch)
job = executor.submit(submitted_func, job_params)
print('submited {} {}'.format(job.job_id, submit_params['name']))
