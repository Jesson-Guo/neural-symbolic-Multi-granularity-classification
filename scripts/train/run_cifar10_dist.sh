#!/bin/bash
python -m torch.distributed.launch --nproc_per_node=3 --master_port 29501 \
        main.py \
        --epochs 50 \
        --lr 0.0005 \
        --method tot \
        --k 2 \
        --data cifar10 \
        --words /root/autodl-tmp/data/cifar-10-batches-py/words.txt \
        --train \
        --naive
