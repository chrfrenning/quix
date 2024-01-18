#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,2,3 
torchrun \
    --standalone \
    --nnodes 1 \
    --nproc-per-node 3 \
    torchrun_test.py \
    --model resnet18 \
    --custom-runid torchrun \
    --project testproject \
    --dataset Caltech256 \
    --num-classes 257 \
    --epochs 50 \
    --aug3 true \
    --input-ext jpg \
    --target-ext cls \
    --data-path /work2/litdata/ \
    --batch-size 512 \
    --lr-init 3e-5 \
    --zro False \
    --model-ema True \
    --stdout True
