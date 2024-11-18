"""Node for running VEnhancer inference in distributed multi-GPU mode."""

from typing import Dict, Any, Tuple
import time
import torch
import torch.distributed as dist
from loguru import logger


class MultiGPUInferenceNode:
    """ComfyUI node for running VEnhancer inference across multiple GPUs."""

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        """Define input parameters for distributed video enhancement."""
        return {
            "required": {
                "model": ("MULTI_GPU_MODEL",),
                "video": ("VIDEO",),
                "prompt": ("STRING", {"default": "a good video", "multiline": True}),
                "up_scale": ("FLOAT", {"default": 4.0, "min": 1.0, "max": 8.0}),
                "target_fps": ("INT", {"default": 24, "min": 8, "max": 60}),
                "noise_aug": ("INT", {"default": 200, "min": 50, "max": 300}),
                "sync_mode": (["barrier", "gather"], {"default": "barrier"}),
            }
        }

    RETURN_TYPES = ("MULTI_GPU_VIDEO",)
    FUNCTION = "enhance_video"
    CATEGORY = "generators/venhancer"

    def __init__(self):
        """Initialize MultiGPUInferenceNode with logging."""
        self.logger = logger.bind(node="MultiGPUInference")

    def enhance_video(self, model: Any, video: str, sync_mode: str, **kwargs) -> Tuple[str]:
        """
        Enhance video using distributed VEnhancer model.

        Args:
            model: Distributed VEnhancer model instance
            video: Path to input video
            sync_mode: Synchronization mode ('barrier' or 'gather')
            **kwargs: Enhancement parameters including:
                - prompt: Text prompt for enhancement
                - up_scale: Upscaling factor
                - target_fps: Target frame rate
                - noise_aug: Noise augmentation level

        Returns:
            Tuple[str]: Path to enhanced video

        Raises:
            RuntimeError: If GPU memory is insufficient or synchronization fails
            Exception: If enhancement fails
        """
        try:
            start_time = time.time()
            rank = dist.get_rank()
            world_size = dist.get_world_size()
            local_rank = model.dist_config.local_rank
            init_mem = torch.cuda.memory_allocated(local_rank)

            self.logger.info(f"Starting distributed enhancement on rank {rank}/{world_size-1}", 
                           extra={
                               "config": {
                                   "video": video,
                                   "prompt": kwargs.get("prompt"),
                                   "up_scale": kwargs.get("up_scale"),
                                   "target_fps": kwargs.get("target_fps"),
                                   "noise_aug": kwargs.get("noise_aug"),
                                   "sync_mode": sync_mode,
                                   "gpu": torch.cuda.get_device_name(local_rank)
                               }
                           })

            # Run distributed enhancement
            output_path = model.enhance_a_video(video_path=video, **kwargs)

            # Synchronize based on mode
            if sync_mode == "barrier":
                dist.barrier()
                sync_status = "barrier_sync"
            else:
                # Gather results if needed
                dist.gather(torch.tensor([1], device=f"cuda:{local_rank}"), 
                          dst=0 if rank == 0 else None)
                sync_status = "gather_sync"

            # Log metrics
            enhance_time = time.time() - start_time
            peak_mem = torch.cuda.max_memory_allocated(local_rank)
            mem_used = peak_mem - init_mem

            self.logger.success(f"Enhancement completed on rank {rank}", extra={
                "metrics": {
                    "total_time": f"{enhance_time:.2f}s",
                    "gpu_memory_used": f"{mem_used/1024/1024:.1f}MB",
                    "peak_gpu_memory": f"{peak_mem/1024/1024:.1f}MB",
                    "gpu_utilization": f"{torch.cuda.utilization(local_rank)}%",
                    "sync_mode": sync_status
                }
            })

            return (output_path,)

        except Exception as e:
            self.logger.exception(f"Enhancement failed on rank {rank}", extra={
                "error": str(e),
                "gpu_state": torch.cuda.memory_summary(local_rank)
            })
            # Try to clean up distributed resources
            try:
                dist.barrier()
            except:
                pass
            raise