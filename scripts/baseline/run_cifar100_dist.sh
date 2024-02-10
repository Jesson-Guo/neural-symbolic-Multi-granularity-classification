#!/bin/bash
python -m torch.distributed.launch --nproc_per_node=2 --master_port 29501 \
        main.py \
        --devices 0,1 \
        --epochs 50 \
        --lr 0.0005 \
        --method vit \
        --k 10 \
        --data cifar100 \
        --words /mnt/data/ztl/mycode/data/cifar-100-python/words.txt \
        --config ./src/vpt/configs/files/simple/cifar100.yaml \
        --loss ldam \
        --train \
        --pretrained \
