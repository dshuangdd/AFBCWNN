# Installation Guide

## System Requirements

- **Python**: 3.8 - 3.10
- **CUDA**: 11.x or 12.x (for GPU support)
- **GPU**: NVIDIA GPU with compute capability 6.0+ (recommended)
- **RAM**: 8GB+ (16GB recommended)

## Quick Start

### 1. Create Virtual Environment (Recommended)

```bash
# Using conda
conda create -n afbcwnn python=3.10
conda activate afbcwnn

# Or using venv
python -m venv afbcwnn_env
source afbcwnn_env/bin/activate  # Linux/Mac
# afbcwnn_env\Scripts\activate  # Windows
```

### 2. Install PyTorch with CUDA

**For CUDA 12.1** (your current setup):
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**For CUDA 11.8**:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**For CPU only**:
```bash
pip install torch torchvision torchaudio
```

### 3. Install Other Dependencies

```bash
pip install -r requirements.txt
```

## Verify Installation

Run the following to verify your installation:

```python
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
```

Expected output (for your system):
```
PyTorch version: 2.5.1+cu121
CUDA available: True
CUDA version: 12.1
GPU: NVIDIA GeForce RTX 4060 Laptop GPU
```

## Troubleshooting

### CUDA Issues

If PyTorch doesn't detect your GPU:
1. Verify NVIDIA drivers are installed: `nvidia-smi`
2. Reinstall PyTorch with correct CUDA version
3. Check CUDA toolkit installation

### Memory Issues

If you encounter out-of-memory errors:
- Reduce `base_interval` or `time_max` in config
- Lower `max_cache_size` parameter
- Close other GPU-intensive applications

### Package Conflicts

If you encounter dependency conflicts:
```bash
pip install --upgrade pip
pip install -r requirements.txt --force-reinstall
```

## Development Setup

For development and testing:

```bash
# Install additional dev tools
pip install jupyter notebook ipython

# Install code quality tools (optional)
pip install black flake8 pylint
```

## Notes

- The code is optimized for NVIDIA GPUs with CUDA support
- CPU mode is supported but significantly slower
- Tested on Windows 10/11 with NVIDIA RTX 4060 Laptop GPU
- For different GPU architectures, PyTorch installation may vary