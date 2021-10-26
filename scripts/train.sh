#!/bin/bash


DIFFUSION_STEPS="${1:-500}"
NUM_THREADS="${2:-1}"
MODEL_FLAGS="--image_size 64 --num_channels 128 --num_res_blocks 3"
DIFFUSION_FLAGS="--diffusion_steps ${DIFFUSION_STEPS} --noise_schedule linear"
TRAIN_FLAGS="--lr 1e-4 --batch_size 4 --lr_anneal_steps 20"

if [$NUM_THREADS -eq 1]; then
	python image_train.py --data_dir ../datasets $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS
else
	mpiexec -n &NUM_THREADS python image_train.py --data_dir ../datasets $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS