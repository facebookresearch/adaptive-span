#!/usr/bin/env bash

# get the data
bash get_data.sh
mkdir -p checkpoints

args="
--data data/enwik8 \
--nlayers 8 \
--hid-sz 256 \
--inner-hid-sz 1024 \
--nheads 4 \
--attn-span 1024 \
--block-sz 256 \
--batch-sz 64 \
--lr 0.07 \
--momentum 0 \
--dropout 0 \
--optim adagrad \
--lr-warmup 8000 \
--grad-clip 0.03 \
--niter 150 \
--nbatches 1000 \
--adapt-span \
--adapt-span-loss 0.000002 \
--adapt-span-cache
--checkpoint checkpoints/enwik8_small.pt
"


echo "Training ..."
# using the pytorch distributed launching
python3 main.py $args


echo "Evaluation ..."
# use a smaller batch size to reduce tokens without context and omitted tokens.
python3 main.py $args --full-eval-mode --batch-sz 8
