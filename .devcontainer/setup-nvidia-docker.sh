#!/bin/bash
# Script to install NVIDIA Container Toolkit for GPU support in Docker
# Run this script if you want GPU support in your dev container

set -e

echo "Installing NVIDIA Container Toolkit..."

# Add NVIDIA package repository
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# Install nvidia-container-toolkit
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# Configure Docker to use nvidia runtime
sudo nvidia-ctk runtime configure --runtime=docker

# Restart Docker
sudo systemctl restart docker

echo "NVIDIA Container Toolkit installed successfully!"
echo "You can now enable GPU support in devcontainer.json by using:"
echo '  "runArgs": ["--gpus", "all", "--shm-size=1g"]'