#!/bin/bash
python -m torch.distributed.launch --nproc_per_node=4 --master_port 29501 \
        main.py \
        --epochs 30 \
        --lr 0.0005 \
        --method vit \
        --k 2 \
        --data cifar10 \
        --words /root/autodl-tmp/data/cifar-10-batches-py/words.txt \
        --train \
        --naive
