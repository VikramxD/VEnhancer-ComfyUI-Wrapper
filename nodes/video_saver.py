
"""Node for saving videos from single GPU inference."""

from typing import Dict, Any, Tuple, Optional
import os
import time
import shutil
import torch
import cv2
from PIL import Image
import numpy as np
from loguru import logger


class VideoSaverNode:
    """ComfyUI node for saving video results from single GPU processing."""

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        """Define input parameters for video saving."""
        return {
            "required": {
                "video": ("SINGLE_GPU_VIDEO",),
                "filename": ("STRING", {"default": "enhanced.mp4"}),
            },
            "optional": {
                "output_dir": ("STRING", {"default": "ComfyUI/output"}),
                "generate_preview": ("BOOLEAN", {"default": True}),
                "preview_frames": ("INT", {"default": 4, "min": 1, "max": 10}),
            }
        }

    RETURN_TYPES = ("STRING", "IMAGE")
    FUNCTION = "save_video"
    CATEGORY = "savers/venhancer"
    OUTPUT_NODE = True

    def __init__(self):
        """Initialize saver node with logging."""
        self.logger = logger.bind(node="SingleGPUSaver")
        self.output_dir = os.path.join("ComfyUI", "output")
        os.makedirs(self.output_dir, exist_ok=True)

    def save_video(
        self,
        video: str,
        filename: str,
        output_dir: str = "ComfyUI/output",
        generate_preview: bool = True,
        preview_frames: int = 4
    ) -> Tuple[str, Optional[Image.Image]]:
        """Save enhanced video and generate preview."""
        try:
            start_time = time.time()
            output_path = os.path.join(output_dir, filename)
            preview = None

            # Ensure input video exists
            if not os.path.exists(video):
                raise FileNotFoundError(f"Enhanced video not found: {video}")

            # Create output directory
            os.makedirs(output_dir, exist_ok=True)

            # Generate preview if requested
            if generate_preview:
                cap = cv2.VideoCapture(video)
                if not cap.isOpened():
                    raise ValueError(f"Cannot open video for preview: {video}")

                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

                frames = []
                frame_indices = np.linspace(0, total_frames-1, preview_frames, dtype=int)

                for idx in frame_indices:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                    ret, frame = cap.read()
                    if ret:
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        frames.append(Image.fromarray(frame))

                cap.release()

                if frames:
                    grid_size = int(np.ceil(np.sqrt(preview_frames)))
                    preview = Image.new('RGB', (grid_size * width, grid_size * height))
                    
                    for idx, frame in enumerate(frames):
                        x = (idx % grid_size) * width
                        y = (idx // grid_size) * height
                        preview.paste(frame, (x, y))

            # Copy video to output location
            self.logger.info(f"Copying video to {output_path}")
            shutil.copy2(video, output_path)

            save_time = time.time() - start_time

            self.logger.success("Video saved successfully", extra={
                "metrics": {
                    "save_time": f"{save_time:.2f}s",
                    "file_size": f"{os.path.getsize(output_path)/1024/1024:.1f}MB",
                    "preview_generated": preview is not None
                }
            })

            return output_path, preview

        except Exception as e:
            self.logger.exception("Failed to save video", extra={
                "error": str(e),
                "paths": {
                    "input": video,
                    "output": output_path
                },
                "disk_space": shutil.disk_usage(output_dir).free
            })
            raise

    @classmethod
    def IS_CHANGED(cls, filename: str) -> bool:
        """Check if output file exists."""
        return not os.path.exists(os.path.join("ComfyUI", "output", filename))
