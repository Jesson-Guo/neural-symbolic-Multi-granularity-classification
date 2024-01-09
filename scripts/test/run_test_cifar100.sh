#!/bin/bash
python main.py \
        --method vit \
        --k 10 \
        --data cifar100 \
        --words /root/autodl-tmp/data/cifar-100-python/words.txt \
        --test \
        --naive \
        NUM_GPUS 1
