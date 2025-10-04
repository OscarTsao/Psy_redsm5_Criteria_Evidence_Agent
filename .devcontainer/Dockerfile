FROM mcr.microsoft.com/devcontainers/python:1-3.11-bullseye

# Install additional system dependencies (PyTorch and CUDA already included)
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install --no-install-recommends \
    nano \
    htop \
    tree \
    zip \
    unzip \
    sudo \
    && apt-get autoremove -y && apt-get clean -y && rm -rf /var/lib/apt/lists/*

# Install Python development tools
RUN pip install --upgrade pip setuptools wheel

# Install core dependencies in batches to optimize layer caching
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu130
RUN pip install --no-cache-dir \
    transformers>=4.20.0 \
    pandas>=1.3.0 \
    numpy>=1.20.0 \
    scikit-learn>=0.24.0 \
    tqdm>=4.60.0

RUN pip install --no-cache-dir \
    matplotlib>=3.5.0 \
    seaborn>=0.11.0 \
    hydra-core>=1.3.2 \
    optuna>=3.4.0

RUN pip install --no-cache-dir \
    jupyter \
    black \
    flake8 \
    isort \
    pytest \
    ruff \
    nlpaug \
    textattack

# Set Python path
ENV PYTHONPATH="${PYTHONPATH}:/workspaces"

# Set the default shell
SHELL ["/bin/bash", "-c"]
