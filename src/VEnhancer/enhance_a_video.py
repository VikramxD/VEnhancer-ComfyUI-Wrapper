"""
VEnhancer: A text-guided video enhancement model that can upscale resolution,
adjust frame rates, and enhance video quality based on text prompts.
"""

import os
import glob
from typing import List, Optional

import torch
from easydict import EasyDict
from huggingface_hub import hf_hub_download

from VEnhancer.inference_utils import (
    get_logger,
    load_video,
    preprocess,
    adjust_resolution,
    make_mask_cond,
    collate_fn,
    tensor2vid,
    save_video,
    load_prompt_list,
)
from VEnhancer.video_to_video.utils.seed import setup_seed
from VEnhancer.video_to_video.video_to_video_model import VideoToVideo
from VEnhancer.configs.venhnacer_config import VEnhancerConfig

logger = get_logger()


class VEnhancer:
    """Video enhancement model with text guidance.

    This class implements a video enhancement model that can upscale resolution,
    adjust frame rates, and enhance video quality based on text prompts.
    """

    def __init__(self, config: Optional[VEnhancerConfig] = None):
        """Initialize VEnhancer model.

        Args:
            config: VEnhancerConfig object containing model settings
        """
        self.config = config or VEnhancerConfig()
        
        if not self.config.model_path:
            self.download_model()
        else:
            self.model_path = self.config.model_path
            
        assert os.path.exists(self.model_path), "Error: checkpoint Not Found!"
        logger.info(f"checkpoint_path: {self.model_path}")

        os.makedirs(self.config.result_dir, exist_ok=True)

        model_cfg = EasyDict(__name__="model_cfg")
        model_cfg.model_path = self.model_path
        self.model = VideoToVideo(model_cfg)

    def enhance_a_video(
        self, 
        video_path: str, 
        prompt: str,
        up_scale: Optional[float] = None,
        target_fps: Optional[int] = None,
        noise_aug: Optional[int] = None,
    ) -> str:
        """Enhance a video using text guidance.

        Args:
            video_path: Path to input video file
            prompt: Text prompt for enhancement guidance
            up_scale: Optional upscaling factor (overrides config)
            target_fps: Optional target FPS (overrides config)
            noise_aug: Optional noise augmentation level (overrides config)

        Returns:
            str: Path to enhanced video file
        """
        up_scale = up_scale or self.config.up_scale
        target_fps = target_fps or self.config.target_fps
        noise_aug = noise_aug or self.config.noise_aug

        save_name = os.path.splitext(os.path.basename(video_path))[0]
        caption = prompt + self.model.positive_prompt
        logger.info(f"Processing with prompt: {prompt}")

        # Load and preprocess video
        input_frames, input_fps = load_video(video_path)
        in_f_num = len(input_frames)
        logger.info(f"Input frames: {in_f_num}, FPS: {input_fps}")

        # Calculate frame interpolation
        interp_f_num = max(round(target_fps / input_fps) - 1, 0)
        interp_f_num = min(interp_f_num, 8)
        target_fps = input_fps * (interp_f_num + 1)
        logger.info(f"Target FPS: {target_fps}")

        # Process video data
        video_data = preprocess(input_frames)
        _, _, h, w = video_data.shape
        target_h, target_w = adjust_resolution(h, w, up_scale)
        logger.info(f"Resolution: {h}x{w} → {target_h}x{target_w}")

        # Prepare conditioning
        mask_cond = torch.Tensor(make_mask_cond(in_f_num, interp_f_num)).long()
        noise_aug = min(max(noise_aug, 0), 300)
        logger.info(f"Noise augmentation: {noise_aug}")

        # Prepare inference data
        pre_data = {
            "video_data": video_data,
            "y": caption,
            "mask_cond": mask_cond,
            "s_cond": self.config.s_cond,
            "interp_f_num": interp_f_num,
            "target_res": (target_h, target_w),
            "t_hint": noise_aug,
        }

        setup_seed(self.config.seed)

        # Run inference
        with torch.no_grad():
            data_tensor = collate_fn(pre_data, "cuda:0")
            output = self.model.test(
                data_tensor,
                total_noise_levels=900,
                steps=self.config.steps,
                solver_mode=self.config.solver_mode,
                guide_scale=self.config.guide_scale,
                noise_aug=noise_aug,
            )

        # Save results
        output = tensor2vid(output)
        save_video(output, self.config.result_dir, f"{save_name}.mp4", fps=target_fps)
        return os.path.join(self.config.result_dir, save_name)

    def download_model(self) -> None:
        """Download model checkpoint from Hugging Face."""
        filename = "venhancer_v2.pt" if self.config.version == "v2" else "venhancer_paper.pt"
        ckpt_dir = "./ckpts/"
        os.makedirs(ckpt_dir, exist_ok=True)
        
        local_file = os.path.join(ckpt_dir, filename)
        if not os.path.exists(local_file):
            logger.info("Downloading the VEnhancer checkpoint...")
            hf_hub_download(
                repo_id=self.config.repo_id,
                filename=filename,
                local_dir=ckpt_dir
            )
        self.model_path = local_file

    def process_batch(
        self,
        input_path: str,
        prompt: Optional[str] = None,
        prompt_path: Optional[str] = None,
        filename_as_prompt: bool = False,
    ) -> List[str]:
        """Process multiple videos in batch.

        Args:
            input_path: Path to input video or directory
            prompt: Optional text prompt for all videos
            prompt_path: Optional path to file containing prompts
            filename_as_prompt: Use filename as prompt

        Returns:
            List[str]: Paths to enhanced video files

        Raises:
            TypeError: If input_path is neither a file nor directory
        """
        # Get list of video files
        if os.path.isdir(input_path):
            file_path_list = sorted(glob.glob(os.path.join(input_path, "*.mp4")))
        elif os.path.isfile(input_path):
            file_path_list = [input_path]
        else:
            raise TypeError("input must be a directory or video file!")

        # Handle prompts
        prompt_list = None
        if os.path.isfile(prompt_path or ""):
            prompt_list = load_prompt_list(prompt_path)
            assert len(prompt_list) == len(file_path_list)

        enhanced_paths = []
        for idx, file_path in enumerate(file_path_list):
            logger.info(f"Processing video {idx + 1}/{len(file_path_list)}")
            
            # Determine prompt for current video
            current_prompt = prompt
            if filename_as_prompt:
                current_prompt = os.path.splitext(os.path.basename(file_path))[0]
            elif prompt_list is not None:
                current_prompt = prompt_list[idx]
            elif not current_prompt:
                prompt_file = os.path.splitext(file_path)[0] + ".txt"
                if os.path.isfile(prompt_file):
                    current_prompt = load_prompt_list(prompt_file)[0]
                else:
                    current_prompt = "a good video"

            # Process video
            output_path = self.enhance_a_video(file_path, current_prompt)
            enhanced_paths.append(output_path)

        return enhanced_paths
