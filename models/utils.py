from __future__ import print_function
from argparse import Namespace
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F


# shift every row 1 step to right
def skew(X, pad_value):
    # X = B x M x L
    B, M, L = X.size()
    X = F.pad(X, (0,M+1), value=pad_value) # B x M x (L+M+1)
    X = X.view(B, -1) # B x ML+MM+M
    X = X[:, :-M] # B x ML+MM
    X = X.view(B, M, M + L) # B x M x L+M
    return X


def unskew(X):
    # X = B x M x L+M
    B, M, L = X.size()
    L -= M
    X = X.view(B, -1) # B x ML+MM
    X = F.pad(X, (0,M))  # B x ML+MM+M
    X = X.view(B, M, M+L+1) # B x M x L+M+1
    X = X[:, :, :L] # B x M x L
    return X


def parse_layerwise(args):
    args_copy = Namespace(**vars(args))
    use_copy = False
    if args.attn_lim_layerwise != '':
        lims = args.attn_lim_layerwise.split(':')
        args_copy.attn_lim = int(lims[l])
        use_copy = True
    if args.inner_hid_sz_layerwise != '':
        sizes = args.inner_hid_sz_layerwise.split(':')
        args_copy.inner_hid_sz = int(sizes[l])
        use_copy = True
    if args.attn_key_mode_layerwise != '':
        mode = args.attn_key_mode_layerwise.split(':')
        args_copy.attn_key_mode = mode[l]
        use_copy = True
    if args.nheads_layerwise != '':
        nheads = args.nheads_layerwise.split(':')
        args_copy.nheads = int(nheads[l])
        use_copy = True
    if args.attn_type_layerwise != '':
        x = args.attn_type_layerwise.split(':')
        args_copy.attn_type = x[l]
        use_copy = True
    if args.head_dim_layerwise != '':
        x = args.head_dim_layerwise.split(':')
        args_copy.head_dim = int(x[l])
        use_copy = True
    if not use_copy:
        args_copy = args
    return args_copy
