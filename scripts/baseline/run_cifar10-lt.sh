#!/bin/bash
python main.py \
        --epochs 20 \
        --lr 0.0005 \
        --method vit \
        --k 2 \
        --data cifar10 \
        --words /root/autodl-tmp/data/cifar-10-batches-py/words.txt \
        --train \
        --naive
