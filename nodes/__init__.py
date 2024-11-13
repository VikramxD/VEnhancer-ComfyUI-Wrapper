"""VEnhancer nodes for ComfyUI."""

from nodes.single_gpu_loader import SingleGPUVEnhancerLoader
from nodes.multi_gpu_loader import MultiGPUVEnhancerLoader
from nodes.single_gpu_inference import SingleGPUInferenceNode
from nodes.multigpu_inference import MultiGPUInferenceNode
from nodes.single_gpu_saver import SingleGPUSaverNode
from nodes.multigpu_saver import MultiGPUSaverNode
from nodes.video_loader import VideoLoaderNode

NODE_CLASS_MAPPINGS = {
    "SingleGPUVEnhancerLoader": SingleGPUVEnhancerLoader,
    "MultiGPUVEnhancerLoader": MultiGPUVEnhancerLoader,
    "SingleGPUInference": SingleGPUInferenceNode,
    "MultiGPUInference": MultiGPUInferenceNode,
    "SingleGPUSaver": SingleGPUSaverNode,
    "MultiGPUSaver": MultiGPUSaverNode,
    "VideoLoader": VideoLoaderNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SingleGPUVEnhancerLoader": "Load VEnhancer (Single GPU)",
    "MultiGPUVEnhancerLoader": "Load VEnhancer (Multi-GPU)",
    "SingleGPUInference": "Enhance Video (Single GPU)",
    "MultiGPUInference": "Enhance Video (Multi-GPU)",
    "SingleGPUSaver": "Save Video (Single GPU)",
    "MultiGPUSaver": "Save Video (Multi-GPU)",
    "VideoLoader": "Load Video",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]