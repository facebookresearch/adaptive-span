# python main.py --data /private/home/sainbar/data/text8 \
# --hid-sz 256 --inner-hid-sz 1024 --mem-sz 256 --batch-sz 64 --nlayers 8 \
# --lr 0.07 --momentum 0 --dropout 0 --optim adagrad --lr-warmup 8000 \
# --attn-lim 512 --seq --nheads 4 --grad-clip 0.03 --attn-val-mode context \
# --attn-span-loss 0.000002 --share-pos-emb --attn-span-head --attn-span-init 0 \
# --plot --plot-env as_test_small --nbatches 100
#
# python main.py --data /private/home/sainbar/data/text8 \
# --hid-sz 256 --inner-hid-sz 1024 --mem-sz 256 --batch-sz 64 --nlayers 8 \
# --lr 0.07 --momentum 0 --dropout 0 --optim adagrad --lr-warmup 8000 \
# --attn-lim 512 --nheads 4 --grad-clip 0.03 \
# --attn-span-loss 0.000002 --attn-span-head --attn-span-init 0 \
# --plot --plot-env as_test_small2 --nbatches 100


python main.py --data /private/home/sainbar/data/text8 \
--hid-sz 256 --inner-hid-sz 1024 --mem-sz 256 --batch-sz 64 --nlayers 8 \
--lr 0.07 --momentum 0 --dropout 0 --optim adagrad --lr-warmup 8000 \
--attn-lim 512 --nheads 4 --grad-clip 0.03 \
--attn-span-loss 0.000002 \
--plot --plot-env as_test_small3 --nbatches 100
