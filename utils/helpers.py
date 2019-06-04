#!/usr/bin/env python3

# TODO: the name "helpers" is not appropriate


import torch

# TODO: remove (with decorator?)
import submitit


def torch_distributed_init_process_group(distributed: bool,
                                         submitit_enabled: bool,
                                         dist_init: str,
                                         local_rank: int,
                                         *args,
                                         **kwargs):
    if distributed:
        # TODO: SHOULD BE IN submit_fair !!!
        if submitit_enabled:
            job_env = submitit.JobEnvironment()
            local_rank = job_env.local_rank
            torch.distributed.init_process_group(
                backend='nccl',
                init_method=dist_init,
                rank=job_env.global_rank,
                world_size=job_env.num_tasks
            )
        else:
            torch.distributed.init_process_group(
                backend='nccl',
                init_method='env://'
            )
            rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()
        torch.cuda.set_device(local_rank)


def get_device(cuda_enabled: bool):
    if cuda_enabled and torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')
