"""Node for loading VEnhancer model in single GPU mode."""

from typing import Dict, Any, Tuple
from src.VEnhancer.configs.venhnacer_config import VEnhancerConfig
from src.VEnhancer.enhance_a_video import VEnhancer

class SingleGPUVEnhancerLoader:
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
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

    def load_model(self, **kwargs) -> Tuple[Any]:
        model_config = VEnhancerConfig(**kwargs)
        model = VEnhancer(model_config)
        return (model,)
    

