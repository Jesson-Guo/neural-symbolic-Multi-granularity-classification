#!/bin/bash
python -m torch.distributed.launch --nproc_per_node=4 --master_port 29502 \
        main.py \
        --epochs 30 \
        --lr 0.5 \
        --root /root/autodl-tmp/data \
        --method tot \
        --k 10 \
        --data cifar100 \
        --words /root/autodl-tmp/data/cifar-100-python/words.txt
