#!/bin/bash
python main.py \
        --epochs 50 \
        --lr 0.0005 \
        --method tot \
        --k 2 \
        --data cifar10-lt \
        --words /root/autodl-tmp/data/cifar-10-batches-py/words.txt \
        --config ./src/vpt/configs/files/simple/cifar10-lt.yaml \
        --tree ./tots/no_other/cifar10-2.json \
        --train \
        --naive \
        NUM_GPUS 1