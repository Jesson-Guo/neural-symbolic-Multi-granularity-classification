#!/bin/bash
python main.py \
        --epochs 30 \
        --lr 0.0005 \
        --method tot \
        --k 2 \
        --data cifar10-lt \
        --words /root/autodl-tmp/data/cifar-10-batches-py/words.txt \
        --train \
        --naive \
        NUM_GPUS 1
