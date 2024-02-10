#!/bin/bash
python main.py \
        --devices 1 \
        --epochs 300 \
        --lr 0.0005 \
        --method vit \
        --k 10 \
        --data cifar100-lt \
        --words /mnt/data/ztl/mycode/data/cifar-100-python/words.txt \
        --config ./src/vpt/configs/files/simple/cifar100-lt.yaml \
        --train \
        --naive \
        NUM_GPUS 1
