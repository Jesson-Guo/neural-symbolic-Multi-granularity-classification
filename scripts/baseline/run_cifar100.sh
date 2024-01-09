#!/bin/bash
python main.py \
        --epochs 50 \
        --lr 0.5 \
        --root /root/autodl-tmp/data \
        --method tot \
        --k 10 \
        --data cifar100 \
        --words /root/autodl-tmp/data/cifar-100-python/words.txt \
        NUM_GPUS 1
