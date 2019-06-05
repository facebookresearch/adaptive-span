#!/usr/bin/env python3

import torch

# TODO: remove (with decorator?)
import submitit


# TODO: env set up, change name

def _torch_distributed_init_process_group(distributed: bool,
                                          submitit_enabled: bool,
                                          dist_init: str,
                                          local_rank: int,
                                          *args,
                                          **kwargs):
    rank, world_size = 0, 1
    if distributed:
        # TODO: SHOULD BE IN submit_fair !!!
        if submitit_enabled:
            job_env = submitit.JobEnvironment()
            rank = job_env.global_rank
            world_size = job_env.num_tasks
            local_rank = job_env.local_rank
            torch.distributed.init_process_group(
                backend='nccl',
                init_method=dist_init,
                rank=rank,
                world_size=world_size
            )
        else:
            torch.distributed.init_process_group(
                backend='nccl',
                init_method='env://'
            )
            rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()
        torch.cuda.set_device(local_rank)
    return {
        'rank': rank,
        'local_rank': local_rank,
        'world_size': world_size,
    }


def _get_device(cuda_enabled: bool):
    if cuda_enabled and torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def initialize_computation_and_update_params(compute_params):
    compute_params.update(
        _torch_distributed_init_process_group(**compute_params))
    compute_params['device'] = _get_device(
        cuda_enabled=not compute_params['no_cuda'])


def get_device(compute_params):
    return compute_params['device']
