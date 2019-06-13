#!/usr/bin/env bash

set -e
#
# pkill visdom
# visdom &

ENTRYPOINT_PATH='/private/home/sainbar/adaptive-span/main.py'
DATA_PATH='/private/home/sainbar/data/enwik8_copy'
python3 $ENTRYPOINT_PATH --data $DATA_PATH \
--hid-sz 256 --inner-hid-sz 1024 --block-sz 256 --batch-sz 64 --nlayers 8 \
--lr 0.07 --momentum 0 --dropout 0 --optim adagrad --lr-warmup 8000 \
--attn-span-lim 512 --nheads 4 --grad-clip 0.03 \
--attn-span-loss 0.000002 \
--plot --plot-env main --nbatches 100
