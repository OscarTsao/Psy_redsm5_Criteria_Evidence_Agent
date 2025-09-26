#!/usr/bin/env python3
"""
Quick test script for the optimized training setup.
"""

import torch
import json
from pathlib import Path
from omegaconf import OmegaConf

# Test configuration loading
def test_config_loading():
    """Test that configurations load properly."""
    print("Testing configuration loading...")

    try:
        # Load default config
        config_path = Path("configs/training/default.yaml")
        if config_path.exists():
            cfg = OmegaConf.load(config_path)
            print(f"✓ Default config loaded successfully")
            print(f"  - Model: {cfg.model.model_name}")
            print(f"  - Batch size: {cfg.train_loader.batch_size}")
            print(f"  - Max epochs: {cfg.training.num_epochs}")
            print(f"  - Early stopping patience: {cfg.training.get('early_stopping_patience', 'not set')}")
            print(f"  - Max checkpoints: {cfg.training.get('max_checkpoints', 'not set')}")
            return True
        else:
            print(f"✗ Config file not found: {config_path}")
            return False
    except Exception as e:
        print(f"✗ Config loading failed: {e}")
        return False

def test_model_instantiation():
    """Test model instantiation with optimized settings."""
    print("\nTesting model instantiation...")

    try:
        from model import get_pairwise_model

        model, device = get_pairwise_model(
            model_name='bert-base-uncased',
            dropout=0.1
        )

        print(f"✓ Model created successfully")
        print(f"  - Device: {device}")
        print(f"  - Model type: {type(model).__name__}")
        print(f"  - Parameters: {sum(p.numel() for p in model.parameters()):,}")

        # Test forward pass
        batch_size = 2
        seq_len = 128
        input_ids = torch.randint(0, 1000, (batch_size, seq_len)).to(device)
        attention_mask = torch.ones(batch_size, seq_len).to(device)

        with torch.no_grad():
            outputs = model(input_ids, attention_mask)

        print(f"  - Forward pass output shape: {outputs.shape}")
        print(f"  - Output range: [{outputs.min().item():.3f}, {outputs.max().item():.3f}]")

        return True

    except Exception as e:
        print(f"✗ Model instantiation failed: {e}")
        return False

def test_hardware_optimizations():
    """Test hardware optimization settings."""
    print("\nTesting hardware optimizations...")

    try:
        print(f"✓ CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  - CUDA device: {torch.cuda.get_device_name()}")
            print(f"  - CUDA version: {torch.version.cuda}")

            # Test TF32 settings
            print(f"  - TF32 matmul: {torch.backends.cuda.matmul.allow_tf32}")
            print(f"  - TF32 cuDNN: {torch.backends.cudnn.allow_tf32}")
            print(f"  - cuDNN benchmark: {torch.backends.cudnn.benchmark}")

            # Test bfloat16 support
            try:
                bf16_supported = torch.cuda.is_bf16_supported()
                print(f"  - BFloat16 supported: {bf16_supported}")
            except:
                print(f"  - BFloat16 supported: Unknown")

        # Test mixed precision
        print(f"✓ Mixed precision available: {hasattr(torch.cuda.amp, 'GradScaler')}")

        return True

    except Exception as e:
        print(f"✗ Hardware optimization test failed: {e}")
        return False

def test_data_paths():
    """Test that required data files exist."""
    print("\nTesting data file paths...")

    data_files = [
        "Data/redsm5/redsm5_posts.csv",
        "Data/redsm5/redsm5_annotations.csv",
        "Data/DSM-5/DSM_Criteria_Array_Fixed_Major_Depressive.json"
    ]

    all_exist = True
    for data_file in data_files:
        path = Path(data_file)
        if path.exists():
            print(f"✓ Found: {data_file}")
        else:
            print(f"✗ Missing: {data_file}")
            all_exist = False

    return all_exist

def test_checkpoint_management():
    """Test checkpoint management functionality."""
    print("\nTesting checkpoint management...")

    try:
        from train import cleanup_old_checkpoints

        # Create a temporary directory for testing
        test_dir = Path("test_checkpoints")
        test_dir.mkdir(exist_ok=True)

        # Create some fake checkpoint files
        for i in range(7):
            checkpoint_file = test_dir / f"checkpoint_epoch_{i+1}.pt"
            checkpoint_file.write_text(f"fake checkpoint {i+1}")

        print(f"  - Created 7 test checkpoints")

        # Test cleanup function
        cleanup_old_checkpoints(test_dir, max_checkpoints=5)

        remaining_checkpoints = list(test_dir.glob("checkpoint_epoch_*.pt"))
        print(f"  - Remaining checkpoints: {len(remaining_checkpoints)}")

        # Cleanup test directory
        import shutil
        shutil.rmtree(test_dir)

        if len(remaining_checkpoints) <= 5:
            print("✓ Checkpoint cleanup working correctly")
            return True
        else:
            print("✗ Checkpoint cleanup failed")
            return False

    except Exception as e:
        print(f"✗ Checkpoint management test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 Testing Optimized Training Setup")
    print("=" * 50)

    tests = [
        test_config_loading,
        test_model_instantiation,
        test_hardware_optimizations,
        test_data_paths,
        test_checkpoint_management,
    ]

    results = []
    for test in tests:
        result = test()
        results.append(result)

    print("\n" + "=" * 50)
    print("📊 Test Results:")

    passed = sum(results)
    total = len(results)

    print(f"✅ Passed: {passed}/{total}")
    if passed == total:
        print("🎉 All tests passed! The optimized setup is ready.")
    else:
        print("⚠️  Some tests failed. Please check the issues above.")

    return passed == total

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)