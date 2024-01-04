#!/bin/bash
python main.py \
        --model vit_base_patch16_224.orig_in21k_ft_in1k \
        --ckpt ./weights/vit_base_patch16_224_in21k_ft_cifar10.pth \
        --root /root/autodl-tmp/data \
        --save ./thought.json \
        --load ./cifar10.json \
        --data cifar10 \
        --classes 10 \
        --words /root/autodl-tmp/data/cifar-10-batches-py/words.txt \
        --pretrained
