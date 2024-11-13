"""
VEnhancerMULTIGPU: A distributed text-guided video enhancement model that can upscale resolution,
adjust frame rates, and enhance video quality based on text prompts using multiple GPUs.
"""

import os
import glob
from typing import List, Optional

import torch
import torch.distributed as dist
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
from VEnhancer.video_to_video.context_parallel import (
    get_context_parallel_rank,
    initialize_context_parallel,
)
from VEnhancer.video_to_video.utils.seed import setup_seed
from VEnhancer.video_to_video.video_to_video_model_parallel import VideoToVideoParallel
from VEnhancer.configs.distributred_venhancer_config import  DistributedConfig

logger = get_logger()


class DistributedVEnhancer:
    """Distributed video enhancement model with text guidance.

    This class implements a multi-GPU video enhancement model that can upscale resolution,
    adjust frame rates, and enhance video quality based on text prompts.
    """

    def __init__(
        self,
        dist_config: Optional[DistributedConfig] = None
    ):
        """Initialize distributed VEnhancer model.

        Args:
            model_config: VEnhancerConfig object containing model settings
            dist_config: DistributedConfig object containing distributed setup
        """
        self.dist_config = dist_config or DistributedConfig()
        
        self._setup_distributed()
        
        if not self.model_config.model_path:
            self.download_model()
        else:
            self.model_path = self.model_config.model_path
            
        assert os.path.exists(self.model_path), "Error: checkpoint Not Found!"
        logger.info(f"checkpoint_path: {self.model_path}")

        os.makedirs(self.model_config.result_dir, exist_ok=True)

        model_cfg = EasyDict(__name__="model_cfg")
        model_cfg.model_path = self.model_path
        self.model = VideoToVideoParallel(model_cfg)

    def _setup_distributed(self) -> None:
        """Initialize distributed training environment."""
        dist.init_process_group(
            backend=self.dist_config.backend,
            rank=self.dist_config.rank,
            world_size=self.dist_config.world_size,
            init_method=self.dist_config.init_method,
        )
        torch.cuda.set_device(self.dist_config.local_rank)
        initialize_context_parallel(self.dist_config.world_size)
        logger.info(f"Initialized process group: rank={self.dist_config.rank}, "
                   f"world_size={self.dist_config.world_size}")

    def enhance_a_video(
        self, 
        video_path: str, 
        prompt: str,
        up_scale: Optional[float] = None,
        target_fps: Optional[int] = None,
        noise_aug: Optional[int] = None,
    ) -> str:
        """Enhance a video using text guidance with distributed processing.

        Args:
            video_path: Path to input video file
            prompt: Text prompt for enhancement guidance
            up_scale: Optional upscaling factor (overrides config)
            target_fps: Optional target FPS (overrides config)
            noise_aug: Optional noise augmentation level (overrides config)

        Returns:
            str: Path to enhanced video file
        """
        up_scale = up_scale or self.model_config.up_scale
        target_fps = target_fps or self.model_config.target_fps
        noise_aug = noise_aug or self.model_config.noise_aug

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
            "s_cond": self.model_config.s_cond,
            "interp_f_num": interp_f_num,
            "target_res": (target_h, target_w),
            "t_hint": noise_aug,
        }

        setup_seed(self.model_config.seed)

        # Run distributed inference
        with torch.no_grad():
            data_tensor = collate_fn(pre_data, f"cuda:{self.dist_config.local_rank}")
            output = self.model.test(
                data_tensor,
                total_noise_levels=900,
                steps=self.model_config.steps,
                solver_mode=self.model_config.solver_mode,
                guide_scale=self.model_config.guide_scale,
                noise_aug=noise_aug,
            )

        output = tensor2vid(output)

        # Save results only on main process
        if get_context_parallel_rank() == 0:
            save_video(output, self.model_config.result_dir, f"{save_name}.mp4", fps=target_fps)
        dist.barrier()
        return os.path.join(self.model_config.result_dir, save_name)

    def download_model(self) -> None:
        """Download model checkpoint from Hugging Face."""
        filename = "venhancer_v2.pt" if self.model_config.version == "v2" else "venhancer_paper.pt"
        ckpt_dir = "./ckpts/"
        os.makedirs(ckpt_dir, exist_ok=True)
        
        local_file = os.path.join(ckpt_dir, filename)
        if not os.path.exists(local_file):
            if get_context_parallel_rank() == 0:
                logger.info("Downloading the VEnhancer checkpoint...")
                hf_hub_download(
                    repo_id=self.model_config.repo_id,
                    filename=filename,
                    local_dir=ckpt_dir
                )
        dist.barrier()
        self.model_path = local_file

    def process_batch(
        self,
        input_path: str,
        prompt: Optional[str] = None,
        prompt_path: Optional[str] = None,
        filename_as_prompt: bool = False,
    ) -> List[str]:
        """Process multiple videos in batch with distributed processing.

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
        if os.path.isdir(input_path):
            file_path_list = sorted(glob.glob(os.path.join(input_path, "*.mp4")))
        elif os.path.isfile(input_path):
            file_path_list = [input_path]
        else:
            raise TypeError("input must be a directory or video file!")

        prompt_list = None
        if os.path.isfile(prompt_path or ""):
            prompt_list = load_prompt_list(prompt_path)
            assert len(prompt_list) == len(file_path_list)

        enhanced_paths = []
        for idx, file_path in enumerate(file_path_list):
            logger.info(f"Processing video {idx + 1}/{len(file_path_list)}")
            
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

            output_path = self.enhance_a_video(file_path, current_prompt)
            enhanced_paths.append(output_path)

        return enhanced_paths