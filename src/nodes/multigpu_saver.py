
"""Node for saving videos from multi-GPU inference."""

import os
import shutil
from typing import Dict, Any
import torch.distributed as dist
from VEnhancer.video_to_video.context_parallel import get_context_parallel_rank

class MultiGPUSaverNode:
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "video": ("MULTI_GPU_VIDEO",),
                "filename": ("STRING", {"default": "enhanced.mp4"}),
            },
            "optional": {
                "output_dir": ("STRING", {"default": "ComfyUI/output"}),
                "overwrite": ("BOOLEAN", {"default": False}),
                "only_rank_zero": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "save_video"
    CATEGORY = "savers/venhancer"
    OUTPUT_NODE = True

    def save_video(
        self,
        video: str,
        filename: str,
        output_dir: str = "ComfyUI/output",
        overwrite: bool = False,
        only_rank_zero: bool = True
    ) -> None:
        if only_rank_zero and get_context_parallel_rank() != 0:
            dist.barrier()  # Wait for rank 0 to finish saving
            return

        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, filename)
        
        if os.path.exists(output_path) and not overwrite:
            raise FileExistsError(f"Output file already exists: {output_path}")
        
        shutil.copy2(video, output_path)
        
        if only_rank_zero:
            dist.barrier()