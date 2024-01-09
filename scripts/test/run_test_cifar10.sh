#!/bin/bash
python main.py \
        --method tot \
        --k 2 \
        --data cifar10 \
        --words /root/autodl-tmp/data/cifar-10-batches-py/words.txt \
        --test \
        --naive \
        NUM_GPUS 1
