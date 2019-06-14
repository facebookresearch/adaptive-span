#!/usr/bin/env bash

lr=0.07
mom=0
wu=32000
clip=0.03
head=8
lim=4096
do=0.3
bsz=64

time_stamp=$(date +%Y%m%d_%H%M)
name="as_text8_${time_stamp}"
echo running $name
./submit_multi_nodes.sh $name learnfair 8 "volta32gb&bldg2" 1 \
--niter 120 --nbatches 10000 \
--data /private/home/sainbar/data/text8_copy \
--hid-sz 512 --inner-hid-sz 2048 --block-sz 512 --batch-sz $bsz --nlayers 12 \
--lr $lr --momentum $mom --dropout $do --optim adagrad --lr-warmup $wu \
--attn-span-lim $lim --nheads $head --grad-clip $clip \
--attn-span-loss 0.0000005 --attn-span-cache
