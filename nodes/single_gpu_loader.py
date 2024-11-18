"""Node for loading VEnhancer model in single GPU mode."""

from typing import Dict, Any, Tuple
import time
import torch
from loguru import logger
from VEnhancer.configs.venhnacer_config import VEnhancerConfig
from VEnhancer.enhance_a_video import VEnhancer


class SingleGPUVEnhancerLoader:
    """ComfyUI node for loading VEnhancer model in single GPU mode."""

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        """Define input parameters for model loading."""
        return {
            "required": {
                "version": (["v1", "v2"], {"default": "v1"}),
                "solver_mode": (["fast", "normal"], {"default": "fast"}),
                "steps": ("INT", {"default": 15, "min": 1, "max": 100}),
                "guide_scale": ("FLOAT", {"default": 7.5, "min": 1.0, "max": 20.0}),
                "s_cond": ("FLOAT", {"default": 8.0, "min": 1.0, "max": 20.0}),
            },
            "optional": {
                "model_path": ("STRING", {"default": ""}),
                "result_dir": ("STRING", {"default": "./results/"}),
            }
        }
    
    RETURN_TYPES = ("SINGLE_GPU_MODEL",)
    FUNCTION = "load_model"
    CATEGORY = "loaders/venhancer"

    def __init__(self):
        """Initialize SingleGPUVEnhancerLoader with logging."""
        self.logger = logger.bind(node="SingleGPULoader")

    def load_model(self, **kwargs) -> Tuple[Any]:
        """
        Load VEnhancer model with specified configuration.

        Args:
            **kwargs: Configuration parameters for VEnhancer model

        Returns:
            Tuple[Any]: Tuple containing initialized VEnhancer model

        Raises:
            RuntimeError: If CUDA is not available
            Exception: If model loading fails
        """
        try:
            start_time = time.time()
            init_mem = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0

            self.logger.info("Loading VEnhancer model", extra={"config": kwargs})
            
            model_config = VEnhancerConfig(**kwargs)
            model = VEnhancer(model_config)

            load_time = time.time() - start_time
            final_mem = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
            mem_used = final_mem - init_mem

            self.logger.success("Model loaded successfully", extra={
                "metrics": {
                    "load_time": f"{load_time:.2f}s",
                    "gpu_memory": f"{mem_used/1024/1024:.1f}MB",
                    "version": kwargs.get("version"),
                    "solver_mode": kwargs.get("solver_mode")
                }
            })

            return (model,)

        except Exception as e:
            self.logger.exception(f"Failed to load model: {str(e)}")
            raise