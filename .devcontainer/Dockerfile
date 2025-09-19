FROM mcr.microsoft.com/devcontainers/python:1-3.11-bullseye

# Install system dependencies for PyTorch and scientific computing
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install --no-install-recommends \
        build-essential \
        curl \
        git \
        wget \
        vim \
        htop \
        tree \
        zip \
        unzip \
    && apt-get autoremove -y && apt-get clean -y && rm -rf /var/lib/apt/lists/*

# Install Python development tools
RUN pip install --upgrade pip setuptools wheel

# Pre-install common ML/NLP packages to speed up container startup
# Install PyTorch with CUDA support
RUN pip install --no-cache-dir \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 \
    transformers \
    pandas \
    numpy \
    scikit-learn \
    tqdm \
    jupyter \
    matplotlib \
    seaborn \
    black \
    flake8 \
    isort \
    pytest

# Set Python path
ENV PYTHONPATH="${PYTHONPATH}:/workspaces"

# Create a non-root user
ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# Configure git to handle the workspace safely
RUN git config --system --add safe.directory '*'

# Set the default shell
SHELL ["/bin/bash", "-c"]