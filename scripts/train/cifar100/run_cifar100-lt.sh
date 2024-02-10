#!/bin/bash
python main.py \
        --epochs 200 \
        --lr 0.0005 \
        --method tot \
        --k 10 \
        --data cifar100-lt \
        --words /mnt/data/ztl/mycode/data/cifar-100-python/words.txt \
        --config ./src/vpt/configs/files/simple/cifar100-lt.yaml \
        --tree ./tots/cifar100-10.json \
        --train \
        --naive \
        NUM_GPUS 1
