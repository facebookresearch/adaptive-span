import random
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# This class implements an adaptive masking function

class AdaptiveMask(nn.Module):
    def __init__(self, size, ramp_size, init_ratio=0, shape=(1,), sum_normalize=False):
        super(AdaptiveMask, self).__init__()
        self.size = size
        self.ramp_size = ramp_size
        self.size_ratio = nn.Parameter(torch.zeros(*shape) + init_ratio)
        self.sum_normalize = sum_normalize
        mask_template = torch.linspace(1-size, 0, steps=size)
        self.register_buffer('mask_template', mask_template)

    def forward(self, x, skip_len=0):
        mask = self.mask_template + self.size_ratio * self.size
        mask = mask / self.ramp_size + 1
        mask = mask.clamp(0, 1)
        if skip_len > 0:
            mask = mask[:, :, skip_len:]
        x = x * mask
        if self.sum_normalize:
            x = x / (x.sum(-1, keepdim=True) + 1e-8) # normalize so sum is 1
        return x

    def get_max_size(self):
        max_size = self.size_ratio.max().item()
        max_size = self.ramp_size + max_size * self.size
        max_size = max(0, min(self.size, math.ceil(max_size)))
        return max_size
