"""Node for running VEnhancer inference on single GPU."""

from typing import Dict, Any, Tuple
import time
import torch
from loguru import logger


class SingleGPUInferenceNode:
    """ComfyUI node for running VEnhancer inference on single GPU."""

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        """Define input parameters for video enhancement."""
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

    def enhance_video(self, model: Any, video: str, **kwargs) -> Tuple[str]:
        """
        Enhance video using loaded VEnhancer model.

        Args:
            model: Loaded VEnhancer model instance
            video: Path to input video
            **kwargs: Enhancement parameters including:
                - prompt: Text prompt for enhancement
                - up_scale: Upscaling factor
                - target_fps: Target frame rate
                - noise_aug: Noise augmentation level

        Returns:
            Tuple[str]: Path to enhanced video

        Raises:
            RuntimeError: If GPU memory is insufficient
            Exception: If enhancement fails
        """
        try:
            start_time = time.time()
            init_mem = torch.cuda.memory_allocated()

            self.logger.info("Starting video enhancement", extra={
                "config": {
                    "video": video,
                    "prompt": kwargs.get("prompt"),
                    "up_scale": kwargs.get("up_scale"),
                    "target_fps": kwargs.get("target_fps"),
                    "noise_aug": kwargs.get("noise_aug")
                }
            })

            # Run enhancement
            output_path = model.enhance_a_video(video_path=video, **kwargs)

            # Log metrics
            enhance_time = time.time() - start_time
            peak_mem = torch.cuda.max_memory_allocated()
            mem_used = peak_mem - init_mem

            self.logger.success("Enhancement completed", extra={
                "metrics": {
                    "total_time": f"{enhance_time:.2f}s",
                    "gpu_memory_used": f"{mem_used/1024/1024:.1f}MB",
                    "peak_gpu_memory": f"{peak_mem/1024/1024:.1f}MB",
                    "gpu_utilization": f"{torch.cuda.utilization()}%"
                }
            })

            return (output_path,)

        except Exception as e:
            self.logger.exception("Enhancement failed", extra={
                "error": str(e),
                "gpu_state": torch.cuda.memory_summary()
            })
            raise