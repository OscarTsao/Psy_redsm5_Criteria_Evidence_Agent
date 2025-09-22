#!/usr/bin/env python3
"""
Test script to verify checkpoint naming and evaluation output functionality.
"""

import os
import tempfile
import torch
import pandas as pd
import numpy as np
from train import Trainer
from model import BERTForDSM5Classification
from data_preprocessing import prepare_data, split_data, create_datasets

def test_checkpoint_naming():
    """Test that checkpoint naming includes metrics correctly"""
    print("Testing enhanced checkpoint naming...")

    # Create a minimal test config
    config = {
        'output_dir': tempfile.mkdtemp(),
        'threshold': 0.5,
        'max_grad_norm': 1.0,
        'patience': 3,
        'num_epochs': 1,
        'use_amp': False,
        'gradient_accumulation_steps': 1
    }

    # Create mock metrics
    test_metrics = {
        'f1_macro': 0.7854,
        'f1_micro': 0.8123,
        'exact_match_ratio': 0.6789,
        'hamming_loss': 0.2341,
        'precision_macro': 0.7654,
        'recall_macro': 0.8012
    }

    # Create a dummy model and trainer
    model = BERTForDSM5Classification('google-bert/bert-base-uncased', num_criteria=9)
    device = torch.device('cpu')

    # Create dummy optimizer and scheduler for testing
    from torch.optim import AdamW
    from transformers import get_linear_schedule_with_warmup

    optimizer = AdamW(model.parameters(), lr=1e-5)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=10, num_training_steps=100)

    # Create minimal trainer instance (we'll only test save_checkpoint)
    trainer = Trainer(
        model=model,
        device=device,
        train_loader=None,
        val_loader=None,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=None,
        config=config
    )

    # Test checkpoint saving
    trainer.save_checkpoint(epoch=5, metrics=test_metrics)

    # Check if files were created correctly
    output_files = os.listdir(config['output_dir'])
    print(f"Created files: {output_files}")

    # Look for the enhanced checkpoint filename
    enhanced_checkpoint = None
    for filename in output_files:
        if filename.startswith('best_model_epoch5_f1macro0.7854'):
            enhanced_checkpoint = filename
            break

    if enhanced_checkpoint:
        print(f"✅ Enhanced checkpoint naming works: {enhanced_checkpoint}")

        # Verify the filename contains all expected metrics
        expected_parts = ['epoch5', 'f1macro0.7854', 'f1micro0.8123', 'exact0.6789', 'hamming0.2341']
        for part in expected_parts:
            if part in enhanced_checkpoint:
                print(f"  ✅ Contains {part}")
            else:
                print(f"  ❌ Missing {part}")
    else:
        print("❌ Enhanced checkpoint naming failed")

    # Check if generic checkpoint also exists
    if 'best_model.pt' in output_files:
        print("✅ Generic checkpoint (best_model.pt) also created for compatibility")
    else:
        print("❌ Generic checkpoint not found")

    # Cleanup
    import shutil
    shutil.rmtree(config['output_dir'])

    print("Checkpoint naming test completed.\n")

def test_prediction_csv_format():
    """Test the enhanced CSV output format"""
    print("Testing enhanced CSV prediction format...")

    # Create mock data for testing
    n_samples = 5
    n_criteria = 9

    # Mock test dataframe
    test_df = pd.DataFrame({
        'post_id': [f'test_post_{i}' for i in range(n_samples)],
        'text': [f'This is test post {i} with some sample text for testing.' for i in range(n_samples)]
    })

    # Mock predictions, labels, and probabilities
    np.random.seed(42)
    predictions = np.random.randint(0, 2, (n_samples, n_criteria))
    labels = np.random.randint(0, 2, (n_samples, n_criteria))
    probs = np.random.uniform(0, 1, (n_samples, n_criteria))

    # Import the save_predictions function
    from predict import save_predictions

    # Test saving predictions
    output_dir = tempfile.mkdtemp()
    output_path = os.path.join(output_dir, 'test_predictions.csv')

    try:
        results_df = save_predictions(test_df, predictions, labels, probs, output_path, threshold=0.5)

        print(f"✅ Predictions saved successfully")
        print(f"  - Main file: test_predictions.csv")
        print(f"  - Simplified file: test_predictions_simplified.csv")
        print(f"  - Error analysis file: test_predictions_errors.csv")

        # Check column structure
        expected_cols = ['post_id', 'full_text', 'text_preview', 'post_exact_match',
                        'post_accuracy', 'predicted_criteria_summary', 'actual_criteria_summary']

        missing_cols = [col for col in expected_cols if col not in results_df.columns]
        if not missing_cols:
            print("✅ All expected columns present in results")
        else:
            print(f"❌ Missing columns: {missing_cols}")

        # Check for criterion-specific columns
        criterion_cols = [col for col in results_df.columns if '_outcome' in col]
        print(f"✅ Found {len(criterion_cols)} criterion outcome columns")

        # Verify files were created
        created_files = os.listdir(output_dir)
        expected_files = ['test_predictions.csv', 'test_predictions_simplified.csv', 'test_predictions_errors.csv']

        for expected_file in expected_files:
            if expected_file in created_files:
                print(f"✅ Created {expected_file}")
            else:
                print(f"❌ Missing {expected_file}")

    except Exception as e:
        print(f"❌ Error in prediction CSV format test: {e}")

    finally:
        # Cleanup
        import shutil
        shutil.rmtree(output_dir)

    print("CSV format test completed.\n")

if __name__ == "__main__":
    print("="*60)
    print("TESTING ENHANCED CHECKPOINT AND EVALUATION FEATURES")
    print("="*60)

    test_checkpoint_naming()
    test_prediction_csv_format()

    print("="*60)
    print("All tests completed!")
    print("="*60)