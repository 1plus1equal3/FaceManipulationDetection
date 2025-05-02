#!/bin/bash

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda deactivate
conda activate fmd

# training on GPU 1
CUDA_VISIBLE_DEVICES=1 torchrun --nproc_per_node=1 src/train.py