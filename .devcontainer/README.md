# Dev Container Configuration

## Overview

This dev container provides a complete Python development environment for the DSM-5 Classification project with PyTorch, Transformers, and ML libraries pre-installed.

## GPU Support

The dev container is configured to work **with or without GPU** support.

### Current Status: CPU Mode (Default)

The container is currently configured to run without GPU requirements. PyTorch will work but use CPU only.

### Enabling GPU Support

If you want to use your NVIDIA GPU inside the container:

1. **Install NVIDIA Container Toolkit** (one-time setup on host):
   ```bash
   .devcontainer/setup-nvidia-docker.sh
   ```

2. **Update devcontainer.json** to enable GPU:
   - Replace the `runArgs` section with:
     ```json
     "runArgs": [
       "--gpus", "all",
       "--shm-size=1g"
     ],
     ```
   - Optionally add GPU environment variables:
     ```json
     "containerEnv": {
       "NVIDIA_VISIBLE_DEVICES": "all",
       "NVIDIA_DRIVER_CAPABILITIES": "compute,utility",
       "PYTHONDONTWRITEBYTECODE": "1",
       "PYTHONUNBUFFERED": "1"
     },
     ```

3. **Rebuild the container**:
   - Command Palette → "Dev Containers: Rebuild Container"

### Verifying GPU Access

After rebuilding with GPU support, verify in the container:
```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}')"
```

## Container Features

- **Python 3.11** with PyTorch (CUDA 12.1 compatible)
- **Pre-installed libraries**: transformers, pandas, numpy, scikit-learn, optuna, hydra-core
- **Development tools**: black, flake8, isort, pytest, ruff
- **VSCode extensions**: Python, Pylance, Jupyter, Black formatter
- **Git and GitHub CLI** configured
- **Auto-formatting** on save with Black and isort

## Troubleshooting

### Container fails to start with GPU error

If you see `could not select device driver "" with capabilities: [[gpu]]`:
- NVIDIA Container Toolkit is not installed
- Run the setup script or disable GPU in devcontainer.json

### PyTorch not detecting GPU

- Verify NVIDIA drivers on host: `nvidia-smi`
- Check Docker runtime: `docker info | grep -i runtime` (should show `nvidia`)
- Rebuild container after installing NVIDIA Container Toolkit