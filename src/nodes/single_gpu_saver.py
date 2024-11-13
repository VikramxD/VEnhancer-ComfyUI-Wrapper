    
import os
import shutil
from typing import Dict, Any

class SingleGPUSaverNode:
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "video": ("SINGLE_GPU_VIDEO",),
                "filename": ("STRING", {"default": "enhanced.mp4"}),
            },
            "optional": {
                "output_dir": ("STRING", {"default": "ComfyUI/output"}),
                "overwrite": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "save_video"
    CATEGORY = "savers/venhancer"
    OUTPUT_NODE = True

    def save_video(self, video: str, filename: str, output_dir: str = "ComfyUI/output", overwrite: bool = False) -> None:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, filename)
        
        if os.path.exists(output_path) and not overwrite:
            raise FileExistsError(f"Output file already exists: {output_path}")
        
        shutil.copy2(video, output_path)