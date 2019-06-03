#!/bin/bash

cd /private/home/sainbar/projects/char
lr=0.07
mom=0
wu=32000
clip=0.03
head=8
lim=512
do=0.3
bsz=64

for l in 0.0000005; do
  for lim in 4096; do
    name="13span_mem${lim}_loss${l}_cache_more"
    echo running $name
    ./submit_dist.sh $name priority "gpu:volta:8 -C volta32gb" \
    --nepochs 120 --nbatches 10000 \
    --data /private/home/sainbar/data/text8 \
    --hid-sz 512 --inner-hid-sz 2048 --mem-sz 512 --batch-sz $bsz --nlayers 12 \
    --lr $lr --momentum $mom --dropout $do --optim adagrad --lr-warmup $wu \
    --attn-lim $lim --seq --nheads $head --grad-clip $clip \
    --attn-val-mode context --attn-span-loss $l \
    --share-pos-emb --attn-span-head --attn-span-init 0 \
    --attn-span-cache --plot-mem --checkpoint-freq 10
  done
done
