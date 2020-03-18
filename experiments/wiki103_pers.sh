#!/usr/bin/env bash

# Change ngpus to match the number of GPUs available.
# If run out of GPU memory, increase "--batch-split" argument.

# get the data
bash get_data.sh
mkdir -p checkpoints

ngpus=8
args="
--data data/wikitext-103 \
--nlayers 36 \
--hid-sz 512 \
--inner-hid-sz 1 \
--nheads 8 \
--attn-span 2048 \
--block-sz 256 \
--batch-sz 64 \
--lr 0.00025 \
--momentum 0 \
--dropout 0.3 \
--emb-dropout 0.1 \
--optim adam \
--lr-warmup 8000 \
--grad-clip 1 \
--niter 800 \
--nbatches 1000 \
--adapt-span \
--adapt-span-loss 0.0000005 \
--adapt-span-cache \
--pers-mem-size 2048 \
--adapt-io \
--adapt-io-tied \
--data-unit ppl \
--batch-split 2 \
--distributed \
--checkpoint checkpoints/wiki103_pers.pt
"

echo "Training ..."
# using the pytorch distributed launching
python3 -m torch.distributed.launch --nproc_per_node=$ngpus main.py $args


echo "Fine-tuning ..."
# train another 50k steps with a 10x smaller learning rate
python3 -m torch.distributed.launch --nproc_per_node=$ngpus main.py $args \
  --lr 0.000025 --niter 850


echo "Evaluation ..."
# use a smaller batch size to reduce tokens without context and omitted tokens.
python3 -m torch.distributed.launch --nproc_per_node=$ngpus main.py $args \
  --full-eval-mode --batch-sz 8
