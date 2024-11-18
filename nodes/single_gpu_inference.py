
"""Node for running VEnhancer inference on single GPU."""

from typing import Dict, Any, Tuple
import os
import time
import torch
from loguru import logger


class SingleGPUInferenceNode:
    """ComfyUI node for running VEnhancer inference on single GPU."""

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "model": ("SINGLE_GPU_MODEL",),
                "video": ("VIDEO",),
                "prompt": ("STRING", {"default": "a good video", "multiline": True}),
                "up_scale": ("FLOAT", {"default": 4.0, "min": 1.0, "max": 8.0}),
                "target_fps": ("INT", {"default": 24, "min": 8, "max": 60}),
                "noise_aug": ("INT", {"default": 200, "min": 50, "max": 300}),
            }
        }

    RETURN_TYPES = ("SINGLE_GPU_VIDEO",)
    FUNCTION = "enhance_video"
    CATEGORY = "generators/venhancer"

    def __init__(self):
        """Initialize SingleGPUInferenceNode with logging."""
        self.logger = logger.bind(node="SingleGPUInference")
        self.output_dir = os.path.abspath(os.path.join("ComfyUI", "output"))
        os.makedirs(self.output_dir, exist_ok=True)

    def enhance_video(self, model: Any, video: str, **kwargs) -> Tuple[str]:
        """
        Enhance video using loaded VEnhancer model.

        Args:
            model: Loaded VEnhancer model instance
            video: Path to input video
            **kwargs: Enhancement parameters including prompt, up_scale, etc.

        Returns:
            Tuple[str]: Path to enhanced video file
        """
        try:
            start_time = time.time()
            init_mem = torch.cuda.memory_allocated()

            # Configure model output directory
            model.config.result_dir = self.output_dir

            # Get base filename
            video_name = os.path.splitext(os.path.basename(video))[0]
            
            self.logger.info("Starting video enhancement", extra={
                "config": {
                    "input_video": video,
                    "output_dir": self.output_dir,
                    "prompt": kwargs.get("prompt"),
                    "up_scale": kwargs.get("up_scale"),
                    "target_fps": kwargs.get("target_fps"),
                    "noise_aug": kwargs.get("noise_aug")
                }
            })

            # Run enhancement
            output_path = model.enhance_a_video(
                video_path=video,
                **kwargs
            )

            # Verify output
            if not os.path.exists(output_path):
                raise FileNotFoundError(f"Enhanced video not found at: {output_path}")

            # Log performance metrics
            enhance_time = time.time() - start_time
            peak_mem = torch.cuda.max_memory_allocated()
            current_mem = torch.cuda.memory_allocated()

            self.logger.success("Enhancement completed", extra={
                "metrics": {
                    "total_time": f"{enhance_time:.2f}s",
                    "peak_gpu_memory": f"{peak_mem/1024/1024:.1f}MB",
                    "current_gpu_memory": f"{current_mem/1024/1024:.1f}MB",
                    "output_path": output_path,
                    "output_size": f"{os.path.getsize(output_path)/1024/1024:.1f}MB"
                }
            })

            return (output_path,)

        except Exception as e:
            self.logger.exception("Enhancement failed", extra={
                "error": str(e),
                "input_video": video,
                "gpu_state": torch.cuda.memory_summary(),
                "system_info": {
                    "gpu_name": torch.cuda.get_device_name(),
                    "gpu_memory": f"{torch.cuda.get_device_properties(0).total_memory/1024/1024/1024:.1f}GB",
                    "cuda_version": torch.version.cuda
                }
            })
            raise