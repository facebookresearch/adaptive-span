#!/usr/bin/env bash

lr=0.07
mom=0
wu=32000
clip=0.03
head=8
lim=8192
do=0.4
bsz=64

time_stamp=$(date +%Y%m%d_%H%M)
name="as_enwiki_large_${time_stamp}"
echo running $name
./submit_multi_nodes.sh $name learnfair 8 "volta&bldg1" 4 \
--niter 150 --nbatches 1000 \
--data /private/home/sainbar/data/enwik8_copy \
--hid-sz 768 --inner-hid-sz 4096 --block-sz 512 --batch-sz $bsz --nlayers 24 \
--lr $lr --momentum $mom --dropout $do --optim adagrad --lr-warmup $wu \
--attn-span-lim $lim --nheads $head --grad-clip $clip \
--attn-span-loss 0.0000005 --attn-span-cache
