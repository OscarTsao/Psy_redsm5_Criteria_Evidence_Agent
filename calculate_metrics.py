#!/usr/bin/env python3

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score
import sys

def calculate_metrics_for_criterion(df_criterion):
    """Calculate metrics for a single criterion."""
    y_true = df_criterion['true_label'].values
    y_pred = df_criterion['prediction'].values
    y_prob = df_criterion['probability'].values

    # Handle edge case where all labels are the same class
    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = np.nan

    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)

    return {
        'accuracy': accuracy,
        'auc': auc,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'total_samples': len(y_true),
        'positive_samples': sum(y_true),
        'predicted_positive': sum(y_pred)
    }

def main():
    # Load the CSV file
    try:
        df = pd.read_csv('outputs/test_raw_pairs.csv')
    except FileNotFoundError:
        print("Error: outputs/test_raw_pairs.csv not found!")
        sys.exit(1)

    print("=" * 80)
    print("METRICS CALCULATION FOR DSM-5 CRITERIA CLASSIFICATION")
    print("=" * 80)
    print(f"Total pairs analyzed: {len(df)}")
    print()

    # Calculate per-criterion metrics
    criteria_metrics = {}
    criterion_names = {
        'A.1': 'DEPRESSED_MOOD',
        'A.2': 'ANHEDONIA',
        'A.3': 'APPETITE_CHANGE',
        'A.4': 'SLEEP_ISSUES',
        'A.5': 'PSYCHOMOTOR',
        'A.6': 'FATIGUE',
        'A.7': 'WORTHLESSNESS',
        'A.8': 'COGNITIVE_ISSUES',
        'A.9': 'SUICIDAL_THOUGHTS'
    }

    print("PER-CRITERION METRICS")
    print("-" * 80)
    print(f"{'Criterion':<12} {'Name':<18} {'Acc':<6} {'AUC':<6} {'F1':<6} {'Prec':<6} {'Rec':<6} {'Pos/Tot':<8}")
    print("-" * 80)

    for criterion_id in sorted(df['criterion_id'].unique()):
        df_criterion = df[df['criterion_id'] == criterion_id]
        metrics = calculate_metrics_for_criterion(df_criterion)
        criteria_metrics[criterion_id] = metrics

        criterion_name = criterion_names.get(criterion_id, 'UNKNOWN')
        pos_tot = f"{metrics['positive_samples']}/{metrics['total_samples']}"

        auc_str = f"{metrics['auc']:.3f}" if not np.isnan(metrics['auc']) else "N/A"

        print(f"{criterion_id:<12} {criterion_name:<18} "
              f"{metrics['accuracy']:.3f}  {auc_str:<6} {metrics['f1']:.3f}  "
              f"{metrics['precision']:.3f}  {metrics['recall']:.3f}  {pos_tot:<8}")

    # Calculate overall metrics
    print()
    print("OVERALL METRICS")
    print("-" * 80)

    y_true_all = df['true_label'].values
    y_pred_all = df['prediction'].values
    y_prob_all = df['probability'].values

    overall_metrics = {
        'accuracy': accuracy_score(y_true_all, y_pred_all),
        'auc': roc_auc_score(y_true_all, y_prob_all),
        'f1': f1_score(y_true_all, y_pred_all),
        'precision': precision_score(y_true_all, y_pred_all, zero_division=0),
        'recall': recall_score(y_true_all, y_pred_all, zero_division=0)
    }

    print(f"Overall Accuracy:  {overall_metrics['accuracy']:.4f}")
    print(f"Overall AUC:       {overall_metrics['auc']:.4f}")
    print(f"Overall F1:        {overall_metrics['f1']:.4f}")
    print(f"Overall Precision: {overall_metrics['precision']:.4f}")
    print(f"Overall Recall:    {overall_metrics['recall']:.4f}")

    # Summary statistics
    print()
    print("SUMMARY STATISTICS")
    print("-" * 80)
    total_positive = sum(y_true_all)
    total_samples = len(y_true_all)
    print(f"Total positive samples: {total_positive}")
    print(f"Total samples: {total_samples}")
    print(f"Positive rate: {total_positive/total_samples:.4f}")
    print(f"Class distribution: {total_samples - total_positive} negative, {total_positive} positive")

    # Per-criterion summary
    print()
    print("PER-CRITERION SUMMARY")
    print("-" * 80)
    valid_aucs = [m['auc'] for m in criteria_metrics.values() if not np.isnan(m['auc'])]
    if valid_aucs:
        print(f"Mean AUC across criteria: {np.mean(valid_aucs):.4f}")

    f1_scores = [m['f1'] for m in criteria_metrics.values()]
    print(f"Mean F1 across criteria:  {np.mean(f1_scores):.4f}")

    accuracies = [m['accuracy'] for m in criteria_metrics.values()]
    print(f"Mean Accuracy across criteria: {np.mean(accuracies):.4f}")

    print("=" * 80)

if __name__ == "__main__":
    main()