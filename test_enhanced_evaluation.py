#!/usr/bin/env python3
"""
Test script for enhanced evaluation metrics functionality.
Validates that the new prediction and evaluation features work correctly.
"""

import numpy as np
import pandas as pd
import os
import tempfile
import json
from unittest.mock import MagicMock

# Import the functions we want to test
from predict import calculate_detailed_metrics, save_predictions, save_confusion_matrix_plots

def create_mock_data():
    """Create mock data for testing"""
    # Create synthetic labels, predictions, and probabilities for 5 samples, 9 criteria
    np.random.seed(42)

    labels = np.random.randint(0, 2, (5, 9))
    probs = np.random.rand(5, 9)
    predictions = (probs > 0.5).astype(int)

    # Create mock test dataframe
    test_df = pd.DataFrame({
        'post_id': [f'post_{i}' for i in range(5)],
        'text': [f'This is test post {i} with some sample text for testing purposes.' for i in range(5)]
    })

    symptom_names = [
        'DEPRESSED_MOOD', 'ANHEDONIA', 'APPETITE_CHANGE', 'SLEEP_ISSUES',
        'PSYCHOMOTOR', 'FATIGUE', 'WORTHLESSNESS', 'COGNITIVE_ISSUES', 'SUICIDAL_THOUGHTS'
    ]

    return labels, predictions, probs, test_df, symptom_names

def test_enhanced_metrics():
    """Test the enhanced metrics calculation"""
    print("Testing enhanced metrics calculation...")

    labels, predictions, probs, test_df, symptom_names = create_mock_data()

    metrics = calculate_detailed_metrics(labels, predictions, probs, symptom_names)

    # Validate structure
    assert 'overall' in metrics
    assert 'per_criteria' in metrics
    assert 'confusion_matrices' in metrics

    # Validate overall metrics
    overall = metrics['overall']
    required_overall_keys = [
        'hamming_loss', 'exact_match_ratio', 'subset_accuracy', 'label_accuracy',
        'precision_macro', 'recall_macro', 'f1_macro',
        'precision_micro', 'recall_micro', 'f1_micro'
    ]

    for key in required_overall_keys:
        assert key in overall, f"Missing overall metric: {key}"
        assert isinstance(overall[key], (int, float)), f"Metric {key} should be numeric"

    # Validate per-criteria metrics
    assert len(metrics['per_criteria']) == 9, "Should have 9 criteria"

    required_criteria_keys = [
        'criteria', 'symptom', 'accuracy', 'precision', 'recall', 'f1',
        'support', 'predicted_positive', 'true_positive', 'true_negative',
        'false_positive', 'false_negative', 'specificity', 'sensitivity'
    ]

    for i, criteria in enumerate(metrics['per_criteria']):
        for key in required_criteria_keys:
            assert key in criteria, f"Missing criteria metric: {key} in criteria {i}"

        assert criteria['criteria'] == f'A.{i+1}', f"Incorrect criteria ID: {criteria['criteria']}"
        assert criteria['symptom'] == symptom_names[i], f"Incorrect symptom name"

    # Validate confusion matrices
    assert len(metrics['confusion_matrices']) == 9, "Should have 9 confusion matrices"

    print("✓ Enhanced metrics calculation test passed!")

def test_enhanced_predictions_saving():
    """Test the enhanced prediction saving functionality"""
    print("Testing enhanced prediction saving...")

    labels, predictions, probs, test_df, symptom_names = create_mock_data()

    with tempfile.TemporaryDirectory() as temp_dir:
        output_path = os.path.join(temp_dir, 'test_predictions.csv')

        results_df = save_predictions(test_df, predictions, labels, probs, output_path, threshold=0.5)

        # Validate the results dataframe
        assert len(results_df) == 5, "Should have 5 prediction rows"

        required_columns = [
            'post_id', 'text', 'post_exact_match', 'post_accuracy',
            'num_predicted_criteria', 'num_actual_criteria', 'threshold_used'
        ]

        for col in required_columns:
            assert col in results_df.columns, f"Missing column: {col}"

        # Check that criteria-specific columns exist
        for i in range(9):
            criteria_id = f'A.{i+1}'
            assert f'{criteria_id}_predicted' in results_df.columns
            assert f'{criteria_id}_groundtruth' in results_df.columns
            assert f'{criteria_id}_probability' in results_df.columns
            assert f'{criteria_id}_symptom' in results_df.columns
            assert f'{criteria_id}_correct' in results_df.columns

        # Validate that the CSV file was created
        assert os.path.exists(output_path), "CSV file should be created"

        # Load and validate the CSV
        loaded_df = pd.read_csv(output_path)
        assert len(loaded_df) == 5, "Loaded CSV should have 5 rows"

        print("✓ Enhanced prediction saving test passed!")

def test_confusion_matrix_plots():
    """Test confusion matrix plot generation (without actually creating plots)"""
    print("Testing confusion matrix plot generation...")

    labels, predictions, probs, test_df, symptom_names = create_mock_data()
    metrics = calculate_detailed_metrics(labels, predictions, probs, symptom_names)

    with tempfile.TemporaryDirectory() as temp_dir:
        # Mock matplotlib to avoid actually creating plots in testing
        import matplotlib.pyplot as plt
        original_savefig = plt.savefig
        original_close = plt.close

        # Track if savefig was called
        savefig_called = False
        def mock_savefig(*args, **kwargs):
            nonlocal savefig_called
            savefig_called = True

        plt.savefig = mock_savefig
        plt.close = lambda: None

        try:
            save_confusion_matrix_plots(metrics, temp_dir)

            # Validate that confusion matrix JSON was created
            cm_json_path = os.path.join(temp_dir, 'confusion_matrices.json')
            assert os.path.exists(cm_json_path), "Confusion matrix JSON should be created"

            # Validate JSON content
            with open(cm_json_path, 'r') as f:
                cm_data = json.load(f)

            assert len(cm_data) == 9, "Should have 9 confusion matrices"

            for i in range(9):
                key = f'A.{i+1}_{symptom_names[i]}'
                assert key in cm_data, f"Missing confusion matrix: {key}"

                cm_entry = cm_data[key]
                required_keys = ['true_negative', 'false_positive', 'false_negative', 'true_positive', 'matrix']
                for k in required_keys:
                    assert k in cm_entry, f"Missing key in confusion matrix: {k}"

            print("✓ Confusion matrix generation test passed!")

        finally:
            # Restore original functions
            plt.savefig = original_savefig
            plt.close = original_close

def main():
    """Run all tests"""
    print("Starting enhanced evaluation system tests...\n")

    try:
        test_enhanced_metrics()
        test_enhanced_predictions_saving()
        test_confusion_matrix_plots()

        print(f"\n✅ All enhanced evaluation tests passed!")
        print(f"Enhanced evaluation system is ready for use.")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        raise

if __name__ == "__main__":
    main()