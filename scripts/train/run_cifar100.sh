#!/bin/bash
python main.py \
        --epochs 30 \
        --lr 0.5 \
        --root /root/autodl-tmp/data \
        --method tot \
        --load /root/mycode/neural-symbolic-Multi-granularity-classification/cifar100.json \
        --data cifar100 \
        --words /root/autodl-tmp/data/cifar-100-python/words.txt

python main.py \
        --epochs 30 \
        --lr 0.05 \
        --root /root/autodl-tmp/data \
        --method vpt \
        --load /root/mycode/neural-symbolic-Multi-granularity-classification/cifar100.json \
        --data cifar100 \
        --words /root/autodl-tmp/data/cifar-100-python/words.txt
