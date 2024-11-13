"""Configuration settings for VEnhancer model."""

from pydantic import BaseSettings, Field


class VEnhancerConfig(BaseSettings):
    """Configuration settings for VEnhancer model.

    Attributes:
        result_dir: Directory to save enhanced videos
        version: Model version to use (v1 or v2)
        model_path: Custom path to model checkpoint
        solver_mode: Solver mode for inference (fast or normal)
        steps: Number of inference steps
        guide_scale: Guidance scale for text conditioning
        s_cond: Conditioning strength
        noise_aug: Noise augmentation level
        target_fps: Target frames per second
        up_scale: Upscaling factor
        repo_id: Hugging Face model repository ID
        seed: Random seed for reproducibility
    """

    result_dir: str = Field(default="./results/", description="Directory to save enhanced videos")
    version: str = Field(default="v1", description="Model version")
    model_path: str = Field(default="", description="Path to model checkpoint")
    solver_mode: str = Field(default="fast", description="Solver mode (fast or normal)")
    steps: int = Field(default=15, description="Number of inference steps")
    guide_scale: float = Field(default=7.5, description="Guidance scale")
    s_cond: float = Field(default=8.0, description="Conditioning strength")
    noise_aug: int = Field(default=200, ge=0, le=300, description="Noise augmentation level")
    target_fps: int = Field(default=24, ge=8, le=60, description="Target FPS")
    up_scale: float = Field(default=4.0, ge=1.0, le=8.0, description="Upscaling factor")
    repo_id: str = Field(default="jwhejwhe/VEnhancer", description="HuggingFace model repository")
    seed: int = Field(default=666, description="Random seed")

    class Config:
        env_prefix = "VENHANCER_"
