#!/bin/bash
python main.py \
        --epochs 10 \
        --lr 0.05 \
        --root /root/autodl-tmp/data \
        --load /root/mycode/neural-symbolic-Multi-granularity-classification/cifar10.json \
        --data cifar10 \
        --words /root/autodl-tmp/data/cifar-10-batches-py/words.txt
