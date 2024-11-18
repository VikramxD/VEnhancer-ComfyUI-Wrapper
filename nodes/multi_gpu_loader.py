"""Node for loading VEnhancer model in multi-GPU distributed mode."""

from typing import Dict, Any, Tuple
import time
import torch
import torch.distributed as dist
from loguru import logger
from VEnhancer.configs.distributred_venhancer_config import DistributedConfig
from VEnhancer.enhance_a_video_MultiGPU import DistributedVEnhancer


class MultiGPUVEnhancerLoader:
    """ComfyUI node for loading VEnhancer model in distributed multi-GPU mode."""

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        """Define input parameters for distributed model loading."""
        return {
            "required": {
                "version": (["v1", "v2"], {"default": "v1"}),
                "solver_mode": (["fast", "normal"], {"default": "fast"}),
                "steps": ("INT", {"default": 15, "min": 1, "max": 100}),
                "guide_scale": ("FLOAT", {"default": 7.5, "min": 1.0, "max": 20.0}),
                "s_cond": ("FLOAT", {"default": 8.0, "min": 1.0, "max": 20.0}),
                "world_size": ("INT", {"default": 1, "min": 1}),
                "rank": ("INT", {"default": 0, "min": 0}),
                "local_rank": ("INT", {"default": 0, "min": 0}),
            },
            "optional": {
                "model_path": ("STRING", {"default": ""}),
                "result_dir": ("STRING", {"default": "./results/"}),
            }
        }
    
    RETURN_TYPES = ("MULTI_GPU_MODEL",)
    FUNCTION = "load_model"
    CATEGORY = "loaders/venhancer"

    def __init__(self):
        """Initialize MultiGPUVEnhancerLoader with logging."""
        self.logger = logger.bind(node="MultiGPULoader")

    def load_model(self, world_size: int, rank: int, local_rank: int, **kwargs) -> Tuple[Any]:
        """
        Load VEnhancer model in distributed mode.

        Args:
            world_size: Total number of GPUs
            rank: Global rank of current process
            local_rank: Local GPU ID
            **kwargs: Additional configuration parameters

        Returns:
            Tuple[Any]: Tuple containing initialized distributed VEnhancer model

        Raises:
            RuntimeError: If CUDA is not available or distributed setup fails
            Exception: If model loading fails
        """
        try:
            start_time = time.time()
            init_mem = torch.cuda.memory_allocated(local_rank)

            self.logger.info(f"Initializing distributed setup on rank {rank}/{world_size-1}", 
                           extra={
                               "config": {
                                   "world_size": world_size,
                                   "rank": rank,
                                   "local_rank": local_rank,
                                   "gpu_name": torch.cuda.get_device_name(local_rank)
                               }
                           })
            
            dist_config = DistributedConfig(
                world_size=world_size,
                rank=rank,
                local_rank=local_rank
            )
            model = DistributedVEnhancer(dist_config)

            load_time = time.time() - start_time
            mem_used = torch.cuda.memory_allocated(local_rank) - init_mem

            self.logger.success(f"Model loaded on rank {rank}", extra={
                "metrics": {
                    "load_time": f"{load_time:.2f}s",
                    "gpu_memory": f"{mem_used/1024/1024:.1f}MB",
                    "gpu_utilization": f"{torch.cuda.utilization(local_rank)}%",
                    "version": kwargs.get("version"),
                    "solver_mode": kwargs.get("solver_mode")
                }
            })

            dist.barrier()  # Synchronize all processes
            return (model,)

        except Exception as e:
            self.logger.exception(f"Failed to load model on rank {rank}: {str(e)}", 
                                extra={
                                    "gpu_state": torch.cuda.memory_summary(local_rank)
                                })
            dist.destroy_process_group()
            raise