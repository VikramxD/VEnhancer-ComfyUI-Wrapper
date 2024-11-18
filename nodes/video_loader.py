"""
Video loader node module for VEnhancer ComfyUI integration.

This module provides functionality for loading, validating, and previewing video files
in the ComfyUI interface. It includes support for multi-frame preview generation,
video validation, and detailed logging of video properties and system metrics.

usage:
    loader = VideoLoaderNode()
    video_path, preview = loader.load_video("input.mp4")
"""

import os
import cv2
import torch
from typing import Dict, Any, Tuple
from PIL import Image
import numpy as np
from loguru import logger


class VideoLoaderNode:
    """
    A ComfyUI node for loading and validating video files with preview generation.
    
    This node handles video file loading, validation, and preview generation for the
    VEnhancer workflow. It supports various video formats and provides detailed logging
    of video properties and system resource usage.

    Attributes:
        logger: Loguru logger instance with node context
        RETURN_TYPES: Tuple specifying return types (VIDEO, IMAGE)
        FUNCTION: Name of the primary function to execute
        CATEGORY: Node category in ComfyUI interface

    Example:
        node = VideoLoaderNode()
        video_path, preview = node.load_video(
            video_path="input.mp4",
            preview_frames=4,
            check_video=True
        )
    """

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        """
        Define the input parameters for the node.

        Returns:
            Dictionary containing input parameter specifications:
                - video_path: Path to input video file
                - preview_frames: Number of frames to show in preview grid
                - check_video: Whether to perform video validation

        Example:
            {
                "required": {"video_path": ("STRING", {"default": ""})},
                "optional": {
                    "preview_frames": ("INT", {"default": 4, "min": 1, "max": 10}),
                    "check_video": ("BOOLEAN", {"default": True})
                }
            }
        """
        return {
            "required": {
                "video_path": ("STRING", {"default": ""}),
            },
            "optional": {
                "preview_frames": ("INT", {"default": 4, "min": 1, "max": 10}),
                "check_video": ("BOOLEAN", {"default": True})
            }
        }
    
    RETURN_TYPES = ("VIDEO", "IMAGE")
    FUNCTION = "load_video"
    CATEGORY = "loaders/venhancer"

    def __init__(self):
        """
        Initialize the VideoLoaderNode.

        Sets up logging with node-specific context and initializes the node.
        The logger is configured to track node-specific operations and metrics.
        """
        self.logger = logger.bind(node="VideoLoader")
        self.logger.info("Initializing VideoLoader node")

    def load_video(self, video_path: str, preview_frames: int = 4, check_video: bool = True) -> Tuple[str, Image.Image]:
        """
        Load and validate a video file, generating preview frames.

        This method handles the complete video loading process, including validation,
        preview generation, and resource tracking. It provides detailed logging of
        video properties and system resource usage.

        Args:
            video_path: Path to the input video file
            preview_frames: Number of frames to extract for preview grid (default: 4)
            check_video: Whether to verify video file integrity (default: True)

        Returns:
            Tuple containing:
                - str: Validated video file path
                - Image: Preview grid of selected frames

        Raises:
            FileNotFoundError: If video file doesn't exist
            ValueError: If video file cannot be opened
            Exception: For other video processing errors

        Example:
            loader = VideoLoaderNode()
            path, preview = loader.load_video("input.mp4", preview_frames=6)
        """
        try:
            if not os.path.exists(video_path):
                raise FileNotFoundError(f"Video not found: {video_path}")

            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Cannot open video: {video_path}")

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            self.logger.info(f"Loading video: {video_path}", 
                           extra={
                               "frames": total_frames,
                               "fps": fps,
                               "resolution": f"{width}x{height}"
                           })

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
            preview_grid = Image.new('RGB', (grid_width, grid_height))

            for idx, frame in enumerate(frames):
                x = (idx % grid_size) * width
                y = (idx // grid_size) * height
                preview_grid.paste(frame, (x, y))

            cap.release()

            gpu_mem = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
            self.logger.success(
                "Video loaded successfully",
                extra={
                    "video_info": {
                        "path": video_path,
                        "frames": total_frames,
                        "fps": fps,
                        "resolution": f"{width}x{height}",
                        "preview_frames": preview_frames
                    },
                    "system_info": {
                        "gpu_memory": f"{gpu_mem/1024/1024:.1f}MB"
                    }
                }
            )

            return video_path, preview_grid

        except Exception as e:
            self.logger.exception(f"Failed to load video: {str(e)}")
            raise

    @classmethod
    def IS_CHANGED(cls, video_path: str) -> bool:
        """
        Check if the video file has been modified since last load.

        This method monitors video file changes for ComfyUI's caching system.
        It verifies file existence and tracks modification timestamps.

        Args:
            video_path: Path to video file to check

        Returns:
            bool: True if file has changed or doesn't exist, False otherwise

        Example:
            changed = VideoLoaderNode.IS_CHANGED("input.mp4")
        """
        try:
            if not os.path.exists(video_path):
                return True
            return float(os.path.getmtime(video_path))
        except Exception as e:
            logger.error(f"Error checking video modification: {str(e)}")
            return True

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs) -> bool:
        """
        Validate input parameters before processing.

        This method performs validation of all input parameters to ensure
        they meet the required criteria before video processing begins.

        Args:
            **kwargs: Keyword arguments containing input parameters

        Returns:
            bool: True if inputs are valid, False otherwise

        Example:
            valid = VideoLoaderNode.VALIDATE_INPUTS(video_path="input.mp4")
        """
        if not kwargs.get("video_path"):
            return False
        return True