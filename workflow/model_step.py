#!/usr/bin/env python3

import torch
import torch.nn as nn

from models import TransformerSeq


def _get_model(device,
               model_params,
               attn_params,
               local_rank,
               distributed,
               *args,
               **kwargs):
    model = TransformerSeq(
        **{**model_params, **attn_params})
    if distributed:
        model = model.to(device, dtype=torch.float32)
        model = nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank)
    else:
        model = nn.DataParallel(model)
        model = model.to(device, dtype=torch.float32)
    return model


def get_model(model_params, attn_params, compute_params, device):
    return _get_model(device=device,
                      model_params=model_params,
                      attn_params=attn_params,
                      **compute_params)
