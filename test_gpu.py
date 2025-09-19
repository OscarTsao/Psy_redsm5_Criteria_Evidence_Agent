#!/usr/bin/env python3
"""
Quick test script to verify GPU access in the dev container
"""
import torch

def test_gpu_access():
    print("=== GPU Access Test ===")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

        # Test tensor operations on GPU
        try:
            x = torch.randn(3, 3).cuda()
            y = torch.randn(3, 3).cuda()
            z = torch.matmul(x, y)
            print("✅ GPU tensor operations working!")
            print(f"Current GPU device: {torch.cuda.current_device()}")
        except Exception as e:
            print(f"❌ GPU tensor operations failed: {e}")
    else:
        print("❌ CUDA not available")

    print("======================")

if __name__ == "__main__":
    test_gpu_access()