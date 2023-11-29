#!/bin/bash
python -m torch.distributed.launch --nproc_per_node=4 --master_port 29504 \
        main.py \
        --arch tiny-imagenet \
        --resume 0 \
        --data_path ../data \
        --epochs 100 \
        --batch_size 64 \
        --num_classes 200 \
        --lr 0.001 \
        --lamb 0.001 \
        --use_cbam 0 \
        --ngpu 2 \
        --prefix RESNET18_TINY_IMAGENET \
        --eval 1 \
        --wnids ../data/tiny-imagenet-200/wnids.txt \
        --words ../data/tiny-imagenet-200/words.txt \
        --hier ./structure_released.xml \
        --cap 600000 \
        --agent dqn \
        --ckpt ./checkpoints/RESNET18_CIFAR100_model_best_0.pt \
        -j 8
