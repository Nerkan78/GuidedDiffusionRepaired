#!/bin/bash

FROM="${1:64}"
TO="${1:256}"

if ["$FROM" == "64"] && ["$TO" == "256"]; then
	MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond True --diffusion_steps 1000 --large_size 256  --small_size 64 --learn_sigma True --noise_schedule linear --num_channels 192 --num_heads 4 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True"
	python super_res_sample.py $MODEL_FLAGS --model_path models/64_256_upsampler.pt --base_samples 64_samples.npz $SAMPLE_FLAGS
elif ["$FROM" == "128"] && ["$TO" == "512"]; then
	MODEL_FLAGS="--attention_resolutions 32,16 --class_cond True --diffusion_steps 1000 --large_size 512 --small_size 128 --learn_sigma True --noise_schedule linear --num_channels 192 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True"
	python super_res_sample.py $MODEL_FLAGS --model_path models/128_512_upsampler.pt $SAMPLE_FLAGS --base_samples 128_samples.npz
else
	echo "only 64 -> 256 and 128 -> 512 are available"
fi