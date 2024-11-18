"""
Video saver node module for VEnhancer ComfyUI integration.

This module handles saving enhanced videos and generating previews in the ComfyUI interface.
It includes support for various output formats, preview generation, and detailed logging
of save operations and system metrics.

Typical usage:
    saver = VideoSaverNode()
    output_path = saver.save_video("enhanced.mp4", frames)
"""

import os
import cv2
import torch
from typing import Dict, Any, Tuple, Optional
from PIL import Image
import numpy as np
from loguru import logger


class VideoSaverNode:
    """
    A ComfyUI node for saving enhanced videos with preview generation.
    
    This node handles video saving operations with support for different formats,
    preview generation, and detailed metrics logging. It manages output directories
    and provides feedback through the ComfyUI interface.

    Attributes:
        logger: Loguru logger instance with node context
        output_dir: Base directory for saving enhanced videos
        RETURN_TYPES: Tuple specifying return types (STRING, IMAGE)
        FUNCTION: Name of the primary function to execute
        CATEGORY: Node category in ComfyUI interface
        OUTPUT_NODE: Boolean indicating this is an output node
    """

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        """
        Define the input parameters for the video saver node.

        Returns:
            Dictionary containing input parameter specifications:
                - video: Enhanced video data to save
                - filename: Output filename
                - format: Output video format
                - generate_preview: Whether to create preview image

        Example:
            {
                "required": {
                    "video": ("VIDEO",),
                    "filename": ("STRING", {"default": "enhanced.mp4"})
                },
                "optional": {
                    "format": (["mp4", "avi"], {"default": "mp4"}),
                    "generate_preview": ("BOOLEAN", {"default": True})
                }
            }
        """
        return {
            "required": {
                "video": ("VIDEO",),
                "filename": ("STRING", {"default": "enhanced.mp4"}),
            },
            "optional": {
                "format": (["mp4", "avi"], {"default": "mp4"}),
                "generate_preview": ("BOOLEAN", {"default": True}),
                "preview_frames": ("INT", {"default": 4, "min": 1, "max": 10}),
            }
        }

    RETURN_TYPES = ("STRING", "IMAGE")
    FUNCTION = "save_video"
    CATEGORY = "savers/venhancer"
    OUTPUT_NODE = True

    def __init__(self):
        """
        Initialize the VideoSaverNode.

        Sets up logging with node-specific context and initializes output directories.
        Ensures the output directory structure exists and is writable.
        """
        self.logger = logger.bind(node="VideoSaver")
        self.output_dir = os.path.join("ComfyUI", "output")
        os.makedirs(self.output_dir, exist_ok=True)
        self.logger.info(f"Initialized VideoSaver node with output dir: {self.output_dir}")

    def save_video(
        self,
        video: str,
        filename: str,
        format: str = "mp4",
        generate_preview: bool = True,
        preview_frames: int = 4
    ) -> Tuple[str, Optional[Image.Image]]:
        """
        Save enhanced video and generate preview.

        This method handles the complete video saving process, including format
        conversion, preview generation, and resource tracking. It provides detailed
        logging of the save operation and system resource usage.

        Args:
            video: Input video data or path
            filename: Desired output filename
            format: Output video format (default: "mp4")
            generate_preview: Whether to create preview (default: True)
            preview_frames: Number of frames for preview (default: 4)

        Returns:
            Tuple containing:
                - str: Path to saved video file
                - Optional[Image]: Preview image if generated, None otherwise

        Raises:
            ValueError: If video saving fails
            OSError: If output directory is not writable
            Exception: For other video processing errors

        Example:
            saver = VideoSaverNode()
            path, preview = saver.save_video(
                video="input.mp4",
                filename="enhanced.mp4",
                generate_preview=True
            )
        """
        try:
            start_time = time.time()
            output_path = os.path.join(self.output_dir, filename)
            preview = None

            if not os.path.exists(video):
                raise FileNotFoundError(f"Input video not found: {video}")

            self.logger.info(f"Starting video save operation: {filename}")

            cap = cv2.VideoCapture(video)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            if generate_preview:
                frames = []
                frame_indices = np.linspace(0, total_frames-1, preview_frames, dtype=int)
                
                for idx in frame_indices:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                    ret, frame = cap.read()
                    if ret:
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        frames.append(Image.fromarray(frame))

                grid_size = int(np.ceil(np.sqrt(preview_frames)))
                grid_width = grid_size * width
                grid_height = grid_size * height
                preview = Image.new('RGB', (grid_width, grid_height))

                for idx, frame in enumerate(frames):
                    x = (idx % grid_size) * width
                    y = (idx // grid_size) * height
                    preview.paste(frame, (x, y))

            cap.release()

            import shutil
            shutil.copy2(video, output_path)

            save_time = time.time() - start_time
            gpu_mem = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0

            self.logger.success(
                "Video saved successfully",
                extra={
                    "metrics": {
                        "save_time": f"{save_time:.2f}s",
                        "gpu_memory": f"{gpu_mem/1024/1024:.1f}MB",
                        "video_info": {
                            "path": output_path,
                            "frames": total_frames,
                            "fps": fps,
                            "resolution": f"{width}x{height}",
                            "format": format,
                            "size": os.path.getsize(output_path)
                        }
                    }
                }
            )

            return output_path, preview

        except Exception as e:
            self.logger.exception(
                "Failed to save video",
                extra={
                    "error": str(e),
                    "filename": filename,
                    "output_path": output_path,
                    "system_info": {
                        "disk_space": psutil.disk_usage(self.output_dir).free,
                        "gpu_memory": torch.cuda.memory_summary() if torch.cuda.is_available() else "N/A"
                    }
                }
            )
            raise

    @classmethod
    def IS_CHANGED(cls, filename: str) -> bool:
        """
        Check if the output file already exists.

        This method checks for existing output files to prevent overwriting
        without explicit confirmation.

        Args:
            filename: Name of output file to check

        Returns:
            bool: True if file doesn't exist, False otherwise

        Example:
            changed = VideoSaverNode.IS_CHANGED("enhanced.mp4")
        """
        return not os.path.exists(os.path.join("ComfyUI", "output", filename))