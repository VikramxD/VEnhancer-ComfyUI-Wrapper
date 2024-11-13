"""Node for loading video files into ComfyUI."""

import os
from typing import Dict, Any, Tuple


class VideoLoaderNode:
    """Node for loading and validating video files."""
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "video_path": ("STRING", {"default": ""}),
            },
            "optional": {
                "check_video": ("BOOLEAN", {"default": True}),
            }
        }
    
    RETURN_TYPES = ("VIDEO",)
    FUNCTION = "load_video"
    CATEGORY = "loaders/venhancer"

    def load_video(self, video_path: str, check_video: bool = True) -> Tuple[str]:
        """Load and validate video file.
        
        Args:
            video_path: Path to video file
            check_video: Whether to verify video can be opened

        Returns:
            Tuple containing validated video path

        Raises:
            AssertionError: If video file doesn't exist or can't be opened
            ValueError: If video path is empty
        """
        if not video_path:
            raise ValueError("Video path cannot be empty")

        if not os.path.exists(video_path):
            raise AssertionError(f"Video not found: {video_path}")

        if check_video:
            import cv2
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise AssertionError(f"Cannot open video: {video_path}")
            cap.release()

        return (video_path,)