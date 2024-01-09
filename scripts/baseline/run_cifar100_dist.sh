#!/bin/bash
python -m torch.distributed.launch --nproc_per_node=4 --master_port 29501 \
        main.py \
        --devices 0,1,2,3 \
        --epochs 50 \
        --lr 0.0005 \
        --method vit \
        --k 10 \
        --data cifar100 \
        --words /root/autodl-tmp/data/cifar-100-python/words.txt \
        --train \
        --naive
