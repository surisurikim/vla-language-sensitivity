# Environment Setup

## Requirements

- NVIDIA GPU (RTX 3090 or similar, 24GB+ VRAM recommended)
- CUDA 12.x driver
- ~20GB disk space for model weights

## Installation

### 1. Miniconda (if not installed)

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p ~/miniconda3
~/miniconda3/bin/conda init bash
source ~/.bashrc
```

### 2. Create conda environment

```bash
conda env create -f environment.yaml
conda activate vla-lang
```

### 3. Install SimplerEnv

```bash
git clone https://github.com/simpler-env/SimplerEnv.git --recurse-submodules ~/SimplerEnv
cd ~/SimplerEnv
pip install -e .
cd ManiSkill2_real2sim && pip install -e . && cd ..
pip install -r requirements_full_install.txt
pip install "typing-extensions>=4.8.0" "opencv-python==4.9.0.80"
```

### 4. Install OpenVLA

```bash
git clone https://github.com/openvla/openvla.git ~/openvla
cd ~/openvla
pip install -e . --no-deps
```

### 5. Verify installation

```bash
python -c "
import torch
from transformers import AutoModelForVision2Seq, AutoProcessor
import simpler_env
print('torch:', torch.__version__, '| CUDA:', torch.cuda.is_available())
print('All OK')
"
```

## Known Issues

- TensorFlow cuDNN/cuFFT/cuBLAS factory warning on import: harmless, can be ignored
- `tf-agents` requires `typing-extensions==4.5.0` but torch requires `>=4.8.0`: keep the newer version
- `opencv-python>=4.13` is incompatible with `numpy==1.24.4` (forced by tensorflow): use `opencv-python==4.9.0.80`

## External Repositories

The following repos are cloned locally and installed in editable mode.
They are **not** included in this repo.

| Repo | Path | Purpose |
|---|---|---|
| [SimplerEnv](https://github.com/simpler-env/SimplerEnv) | `~/SimplerEnv` | Simulation environment for VLA evaluation |
| [OpenVLA](https://github.com/openvla/openvla) | `~/openvla` | VLA model |
