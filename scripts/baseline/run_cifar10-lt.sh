#!/bin/bash
python main.py \
        --devices 0 \
        --epochs 300 \
        --lr 0.0005 \
        --method vit \
        --k 2 \
        --data cifar10-lt \
        --words /root/autodl-tmp/data/cifar-10-batches-py/words.txt \
        --config ./src/vpt/configs/files/simple/cifar10-lt.yaml \
        --train \
        --naive \
        NUM_GPUS 1
