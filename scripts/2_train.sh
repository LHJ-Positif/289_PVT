#!/bin/bash

INFO='model_gpu'
EPOCHS=50
LR=0.00005
GPU=2

export CUDA_VISIBLE_DEVICES=$GPU

python train.py --epochs=$EPOCHS --lr=$LR --info=$INFO
