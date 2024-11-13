"""Node for running VEnhancer inference on single GPU."""

from typing import Dict, Any, Tuple

class SingleGPUInferenceNode:
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

    def enhance_video(self, model: Any, video: str, **kwargs) -> Tuple[str]:
        return (model.enhance_a_video(video_path=video, **kwargs),)