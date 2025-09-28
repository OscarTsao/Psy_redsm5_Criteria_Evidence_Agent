#!/usr/bin/env python3
"""
Test script to demonstrate multi-binary classification for DSM-5 criteria.
This script shows how each criterion is treated as an independent binary classifier.
"""

import torch
import numpy as np
from model import SpanBERTForDSM5Classification
from data_preprocessing import create_symptom_mapping

def demonstrate_multi_binary_classification():
    """
    Demonstrate how the model performs independent binary classification for each criterion.
    """
    print("="*60)
    print("MULTI-BINARY CLASSIFICATION DEMONSTRATION")
    print("="*60)

    # Initialize model
    model = SpanBERTForDSM5Classification(num_criteria=9)
    model.eval()

    # Create dummy input (batch_size=3 for demonstration)
    batch_size = 3
    seq_length = 512
    input_ids = torch.randint(1, 1000, (batch_size, seq_length))
    attention_mask = torch.ones(batch_size, seq_length)

    print(f"Input shape: {input_ids.shape}")
    print(f"Processing {batch_size} posts...")

    # Get model predictions
    with torch.no_grad():
        predictions, probs = model.predict(input_ids, attention_mask, threshold=0.5)

    print(f"\nOutput predictions shape: {predictions.shape}")
    print(f"Output probabilities shape: {probs.shape}")

    # Map criteria to symptoms
    symptom_mapping = create_symptom_mapping()
    criteria_names = list(symptom_mapping.keys())
    symptom_names = list(symptom_mapping.values())

    print(f"\nIndependent Binary Classifications:")
    print("-" * 80)
    print(f"{'Criterion':<10} {'Symptom':<20} {'Post 1':<10} {'Post 2':<10} {'Post 3':<10}")
    print("-" * 80)

    for i, (criterion, symptom) in enumerate(zip(criteria_names, symptom_names)):
        post1_pred = "Present" if predictions[0][i] == 1 else "Absent"
        post2_pred = "Present" if predictions[1][i] == 1 else "Absent"
        post3_pred = "Present" if predictions[2][i] == 1 else "Absent"

        print(f"{criterion:<10} {symptom:<20} {post1_pred:<10} {post2_pred:<10} {post3_pred:<10}")

    print("\nProbability Scores (0.0 = Definitely Absent, 1.0 = Definitely Present):")
    print("-" * 80)
    print(f"{'Criterion':<10} {'Symptom':<20} {'Post 1':<10} {'Post 2':<10} {'Post 3':<10}")
    print("-" * 80)

    for i, (criterion, symptom) in enumerate(zip(criteria_names, symptom_names)):
        prob1 = f"{probs[0][i]:.3f}"
        prob2 = f"{probs[1][i]:.3f}"
        prob3 = f"{probs[2][i]:.3f}"

        print(f"{criterion:<10} {symptom:<20} {prob1:<10} {prob2:<10} {prob3:<10}")

    print(f"\nKey Points:")
    print(f"• Each post can have MULTIPLE criteria present simultaneously")
    print(f"• Each criterion is classified independently (binary: present/absent)")
    print(f"• Predictions are based on threshold (0.5 by default)")
    print(f"• This allows for realistic multi-symptom depression classification")

    # Show example multi-label scenario
    print(f"\nExample Multi-Label Scenario:")
    print(f"Post 1 criteria present: {[criteria_names[i] for i in range(9) if predictions[0][i] == 1]}")
    print(f"Post 2 criteria present: {[criteria_names[i] for i in range(9) if predictions[1][i] == 1]}")
    print(f"Post 3 criteria present: {[criteria_names[i] for i in range(9) if predictions[2][i] == 1]}")

if __name__ == "__main__":
    demonstrate_multi_binary_classification()