from typing import Dict, Any, Tuple
from VEnhancer.configs.distributred_venhancer_config import DistributedConfig
from VEnhancer.enhance_a_video_MultiGPU import DistributedVEnhancer


class MultiGPUVEnhancerLoader:
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
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

    def load_model(self, world_size: int, rank: int, local_rank: int, **kwargs) -> Tuple[Any]:
        
        dist_config = DistributedConfig(
            world_size=world_size,
            rank=rank,
            local_rank=local_rank
        )
        model = DistributedVEnhancer(dist_config)
        return (model,)

