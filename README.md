# VEnhancer ComfyUI Extension

<div align="center">


**ComfyUI extension for VEnhancer: A powerful video enhancement model that supports spatial super-resolution, temporal interpolation, and AI-guided refinement.**

[![Project License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![ComfyUI](https://img.shields.io/badge/ComfyUI-extension-green.svg)](https://github.com/comfyanonymous/ComfyUI)
[![Original Project](https://img.shields.io/badge/Original-VEnhancer-red.svg)](https://github.com/Vchitect/VEnhancer)

[Features](#features) •
[Installation](#installation) •
[Quick Start](#quick-start) •
[Documentation](#documentation) •

</div>

## Features

- 🎥 **High-Quality Video Enhancement**
  - Spatial super-resolution (up to 8x upscaling)
  - Temporal super-resolution through frame interpolation
  - AI-guided video refinement with text prompts

- 🚀 **Flexible Processing Options**
  - Single GPU inference for standard workloads
  - Multi-GPU support for large-scale processing
  - Adjustable enhancement parameters
  - Custom text prompting

- 🛠️ **ComfyUI Integration**
  - Intuitive node-based workflow
  - Real-time preview support
  - Progress tracking
  - Batch processing capabilities

## Installation

### Prerequisites
- ComfyUI installed and running
- Python 3.10 or higher
- CUDA-capable GPU with at least 12GB VRAM (24GB+ recommended)

### Setup

1. Install in ComfyUI custom nodes directory:
```bash
cd ComfyUI/custom_nodes/
git clone https://github.com/vikramxD/VEnhancer-ComfyUI-Wrapper
cd venhancer-comfyui
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Single GPU Enhancement

```python
from venhancer_comfyui.nodes import (
    VideoLoader, 
    SingleGPUVEnhancerLoader,
    SingleGPUInference,
    SingleGPUSaver
)

# Load video
video = VideoLoader().load_video("input.mp4")

# Initialize model
model = SingleGPUVEnhancerLoader().load_model(
    version="v2",
    solver_mode="fast"
)

# Enhance video
enhanced = SingleGPUInference().enhance_video(
    model=model,
    video=video,
    prompt="Enhance video quality with cinematic style",
    up_scale=4.0,
    target_fps=24
)

# Save result
SingleGPUSaver().save_video(enhanced, "enhanced.mp4")
```

### 2. Multi-GPU Enhancement

```bash
# Start ComfyUI with multiple GPUs
torchrun --nproc_per_node=4 main.py
```

```python
# Initialize distributed model
model = MultiGPUVEnhancerLoader().load_model(
    version="v2",
    world_size=4,
    rank=0,
    local_rank=0
)
```

## Documentation

### Available Models

| Model | Description | Download |
|-------|-------------|----------|
| v1 (paper) | Creative enhancement with strong refinement | [Download](https://huggingface.co/jwhejwhe/VEnhancer/resolve/main/venhancer_paper.pt) |
| v2 | Better texture preservation and identity consistency | [Download](https://huggingface.co/jwhejwhe/VEnhancer/resolve/main/venhancer_v2.pt) |

### Core Parameters

#### Enhancement Settings
```python
{
    "up_scale": 4.0,      # Spatial upscaling (1.0-8.0)
    "target_fps": 24,     # Target frame rate (8-60)
    "noise_aug": 200,     # Refinement strength (50-300)
    "solver_mode": "fast" # "fast" (15 steps) or "normal"
}
```

#### Model Configuration
```python
{
    "version": "v2",      # Model version (v1/v2)
    "guide_scale": 7.5,   # Text guidance strength
    "s_cond": 8.0,       # Conditioning strength
    "steps": 15          # Inference steps (fast mode)
}
```

## Troubleshooting

Common issues and solutions:

1. **CUDA Out of Memory**
   - Reduce `up_scale` value
   - Use multi-GPU processing
   - Process in smaller chunks

2. **Slow Processing**
   - Enable `solver_mode="fast"`
   - Use multi-GPU setup
   - Reduce video resolution

## Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

Based on [VEnhancer](https://github.com/Vchitect/VEnhancer) by Jingwen He et al. If you use this extension in your research, please cite:

```bibtex
@article{he2024venhancer,
  title={VEnhancer: Generative Space-Time Enhancement for Video Generation},
  author={He, Jingwen and Xue, Tianfan and Liu, Dongyang and Lin, Xinqi and 
          Gao, Peng and Lin, Dahua and Qiao, Yu and Ouyang, Wanli and Liu, Ziwei},
  journal={arXiv preprint arXiv:2407.07667},
  year={2024}
}
```

---
<div align="center">
Made by VikramxD
</div>
