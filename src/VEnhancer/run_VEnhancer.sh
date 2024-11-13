python enhance_a_video.py \
--version v1 \
--up_scale 4 --target_fps 24 --noise_aug 250 \
--solver_mode 'fast' --steps 15 \
--input_path prompts \
--prompt_path prompts/text_prompts.txt \
--save_dir 'results/'
