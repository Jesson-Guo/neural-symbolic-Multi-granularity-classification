#!/bin/bash
python main.py \
        --method tot \
        --k 2 \
        --data cifar10 \
        --words /mnt/data/ztl/mycode/data/cifar-10-batches-py/words.txt \
        --test \
        --naive \
        NUM_GPUS 1
