#!/bin/bash

INFO='model_gpu'
EPOCH=15 #use your best epoch
LR=0.00005


python forward.py --epoch=$EPOCH --lr=$LR --info=$INFO


