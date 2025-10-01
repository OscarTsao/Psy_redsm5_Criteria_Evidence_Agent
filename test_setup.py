#!/usr/bin/env python3
"""
Simple test to validate pairwise data loading and model initialization.
"""

import torch
from data import make_pairwise_datasets_from_groundtruth
from model import get_pairwise_model

def test_setup():
    print("Testing pairwise data loading and model setup...")

    # Test data loading
    try:
        train_ds, val_ds, test_ds, criteria_map = make_pairwise_datasets_from_groundtruth(
            'Data/groundtruth/redsm5_ground_truth.json',
            'Data/DSM-5/DSM_Criteria_Array_Fixed_Major_Depressive.json',
            tokenizer_name='bert-base-uncased'
        )
        print(f"✓ Data loading successful")
        print(f"  - Train pairs: {len(train_ds)}")
        print(f"  - Val pairs: {len(val_ds)}")
        print(f"  - Test pairs: {len(test_ds)}")
        print(f"  - Criteria count: {len(criteria_map)}")

        # Test a single batch
        sample = train_ds[0]
        print(f"  - Sample input shape: {sample['input_ids'].shape}")
        print(f"  - Sample label: {sample['labels'].item()}")
        print(f"  - Sample criterion idx: {sample['criterion_idx'].item()}")

    except Exception as e:
        print(f"✗ Data loading failed: {e}")
        return False

    # Test model initialization
    try:
        model, device = get_pairwise_model('bert-base-uncased')
        print(f"✓ Model initialization successful")
        print(f"  - Device: {device}")
        print(f"  - Model type: {type(model).__name__}")

        # Test forward pass
        with torch.no_grad():
            logits = model(sample['input_ids'].unsqueeze(0).to(device),
                          sample['attention_mask'].unsqueeze(0).to(device))
            print(f"  - Forward pass output shape: {logits.shape}")

    except Exception as e:
        print(f"✗ Model initialization failed: {e}")
        return False

    print("✓ All tests passed!")
    return True

if __name__ == '__main__':
    test_setup()