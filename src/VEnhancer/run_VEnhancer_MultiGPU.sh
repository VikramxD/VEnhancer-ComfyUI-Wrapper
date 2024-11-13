N_PARALLEL_GPUS=4

torchrun --nproc_per_node=${N_PARALLEL_GPUS} enhance_a_video_MultiGPU.py \
--version v1 \
--up_scale 4 --target_fps 24 --noise_aug 250 \
--solver_mode 'fast' --steps 15 \
--input_path prompts \
--prompt_path prompts/text_prompts.txt \
--save_dir "results_with_${N_PARALLEL_GPUS}gpu/"
