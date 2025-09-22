import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from tqdm import tqdm
import argparse
import os
import json
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, hamming_loss, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

from data_preprocessing import prepare_data, split_data, create_datasets, create_symptom_mapping
from model import BERTForDSM5Classification

def load_checkpoint(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    return checkpoint

def evaluate_model(model, test_loader, device, threshold=0.5):
    """
    Evaluate model using multi-binary classification.
    Each criterion is independently classified as present/absent using the threshold.
    """
    model.eval()
    all_predictions = []
    all_probs = []
    all_labels = []

    with torch.no_grad():
        progress_bar = tqdm(test_loader, desc='Evaluating Multi-Binary Classification')
        for batch_idx, batch in enumerate(progress_bar):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # Get logits for all 9 criteria
            logits = model(input_ids, attention_mask)

            # Apply sigmoid to get probabilities for each criterion independently
            probs = torch.sigmoid(logits)

            # Apply threshold independently to each criterion
            predictions = (probs > threshold).float()

            all_predictions.extend(predictions.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return np.array(all_predictions), np.array(all_probs), np.array(all_labels)

def calculate_detailed_metrics(labels, predictions, probs, symptom_names):
    metrics = {}

    # Overall metrics
    metrics['overall'] = {
        'hamming_loss': hamming_loss(labels, predictions),
        'exact_match_ratio': accuracy_score(labels, predictions),
        'subset_accuracy': accuracy_score(labels, predictions)  # Same as exact_match_ratio
    }

    # Macro and micro averaged metrics
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        labels, predictions, average='macro', zero_division=0
    )
    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
        labels, predictions, average='micro', zero_division=0
    )

    metrics['overall'].update({
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
        'precision_micro': precision_micro,
        'recall_micro': recall_micro,
        'f1_micro': f1_micro
    })

    # AUC scores
    try:
        auc_macro = roc_auc_score(labels, probs, average='macro')
        auc_micro = roc_auc_score(labels, probs, average='micro')
        metrics['overall']['auc_macro'] = auc_macro
        metrics['overall']['auc_micro'] = auc_micro
    except:
        pass

    # Label-wise accuracy (percentage of correctly predicted labels across all samples)
    label_accuracy = np.mean(labels == predictions)
    metrics['overall']['label_accuracy'] = label_accuracy

    # Per-criteria metrics and confusion matrices
    metrics['per_criteria'] = []
    metrics['confusion_matrices'] = {}

    for i, symptom in enumerate(symptom_names):
        precision, recall, f1, support = precision_recall_fscore_support(
            labels[:, i], predictions[:, i], average='binary', zero_division=0
        )

        # Calculate accuracy for this criterion
        criterion_accuracy = accuracy_score(labels[:, i], predictions[:, i])

        # Confusion matrix for this criterion
        cm = confusion_matrix(labels[:, i], predictions[:, i])
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

        criteria_metrics = {
            'criteria': f'A.{i+1}',
            'symptom': symptom,
            'accuracy': criterion_accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'support': int(np.sum(labels[:, i])),
            'predicted_positive': int(np.sum(predictions[:, i])),
            'true_positive': int(tp),
            'true_negative': int(tn),
            'false_positive': int(fp),
            'false_negative': int(fn),
            'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0.0,
            'sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0.0  # Same as recall
        }

        try:
            auc = roc_auc_score(labels[:, i], probs[:, i])
            criteria_metrics['auc'] = auc
        except:
            criteria_metrics['auc'] = None

        metrics['per_criteria'].append(criteria_metrics)

        # Store confusion matrix
        metrics['confusion_matrices'][f'A.{i+1}_{symptom}'] = {
            'true_negative': int(tn),
            'false_positive': int(fp),
            'false_negative': int(fn),
            'true_positive': int(tp),
            'matrix': cm.tolist()
        }

    return metrics

def save_predictions(test_df, predictions, labels, probs, output_path, threshold=0.5):
    symptom_mapping = create_symptom_mapping()
    results = []

    for idx, (_, row) in enumerate(test_df.iterrows()):
        # Calculate per-post accuracy metrics
        post_predictions = predictions[idx]
        post_labels = labels[idx]
        post_probs = probs[idx]

        # Post-level exact match (all criteria correctly predicted)
        exact_match = np.array_equal(post_predictions, post_labels)

        # Post-level accuracy (percentage of correctly predicted criteria)
        post_accuracy = np.mean(post_predictions == post_labels)

        # Number of predicted vs actual criteria
        num_predicted_criteria = int(np.sum(post_predictions))
        num_actual_criteria = int(np.sum(post_labels))

        # Calculate additional metrics for human evaluation
        true_positives = int(np.sum((post_predictions == 1) & (post_labels == 1)))
        false_positives = int(np.sum((post_predictions == 1) & (post_labels == 0)))
        false_negatives = int(np.sum((post_predictions == 0) & (post_labels == 1)))
        true_negatives = int(np.sum((post_predictions == 0) & (post_labels == 0)))

        # Calculate post-level precision, recall, F1
        post_precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
        post_recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
        post_f1 = 2 * (post_precision * post_recall) / (post_precision + post_recall) if (post_precision + post_recall) > 0 else 0.0

        # Create summary strings for easier human review
        predicted_criteria_list = [f"A.{i+1}" for i, pred in enumerate(post_predictions) if pred == 1]
        actual_criteria_list = [f"A.{i+1}" for i, label in enumerate(post_labels) if label == 1]

        post_result = {
            'post_id': row['post_id'],
            'full_text': row['text'],  # Full text for human evaluation
            'text_preview': row['text'][:500] + '...' if len(row['text']) > 500 else row['text'],
            'text_length': len(row['text']),
            'post_exact_match': int(exact_match),
            'post_accuracy': float(post_accuracy),
            'post_precision': float(post_precision),
            'post_recall': float(post_recall),
            'post_f1': float(post_f1),
            'num_predicted_criteria': num_predicted_criteria,
            'num_actual_criteria': num_actual_criteria,
            'predicted_criteria_summary': '; '.join(predicted_criteria_list) if predicted_criteria_list else 'None',
            'actual_criteria_summary': '; '.join(actual_criteria_list) if actual_criteria_list else 'None',
            'threshold_used': threshold,
            'evaluation_outcome': 'CORRECT' if exact_match else 'INCORRECT'
        }

        # Add individual criteria predictions with human-readable format
        for i, (criteria_id, symptom) in enumerate(symptom_mapping.items()):
            pred = int(post_predictions[i])
            truth = int(post_labels[i])
            prob = float(post_probs[i])
            correct = int(pred == truth)

            post_result[f'{criteria_id}_predicted'] = pred
            post_result[f'{criteria_id}_groundtruth'] = truth
            post_result[f'{criteria_id}_probability'] = prob
            post_result[f'{criteria_id}_symptom'] = symptom
            post_result[f'{criteria_id}_correct'] = correct

            # Add human-readable evaluation for each criterion
            if pred == truth:
                outcome = 'CORRECT'
            elif pred == 1 and truth == 0:
                outcome = 'FALSE_POSITIVE'
            else:  # pred == 0 and truth == 1
                outcome = 'FALSE_NEGATIVE'
            post_result[f'{criteria_id}_outcome'] = outcome

        results.append(post_result)

    results_df = pd.DataFrame(results)

    # Save main predictions file
    results_df.to_csv(output_path, index=False)
    print(f"Detailed predictions saved to {output_path}")

    # Save a simplified version for quick human review
    base_path = output_path.replace('.csv', '')
    simplified_path = f"{base_path}_simplified.csv"

    simplified_cols = [
        'post_id', 'text_preview', 'evaluation_outcome', 'post_accuracy', 'post_f1',
        'predicted_criteria_summary', 'actual_criteria_summary',
        'num_predicted_criteria', 'num_actual_criteria'
    ]

    results_df[simplified_cols].to_csv(simplified_path, index=False)
    print(f"Simplified predictions for human review saved to {simplified_path}")

    # Save error analysis file (only incorrect predictions)
    errors_path = f"{base_path}_errors.csv"
    error_df = results_df[results_df['post_exact_match'] == 0].copy()

    if len(error_df) > 0:
        # Sort by post_accuracy (worst first) for easier review
        error_df = error_df.sort_values('post_accuracy')
        error_df.to_csv(errors_path, index=False)
        print(f"Error analysis file saved to {errors_path} ({len(error_df)} incorrect predictions)")
    else:
        print("No prediction errors found!")

    return results_df

def save_confusion_matrix_plots(metrics, output_dir):
    """Save confusion matrix plots for each DSM-5 criterion"""
    plt.style.use('default')

    # Create a figure with subplots for all criteria
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    fig.suptitle('Confusion Matrices for DSM-5 Criteria Classification', fontsize=16)

    for i, criteria_data in enumerate(metrics['per_criteria']):
        row = i // 3
        col = i % 3
        ax = axes[row, col]

        # Get confusion matrix
        cm_key = f"A.{i+1}_{criteria_data['symptom']}"
        cm_data = metrics['confusion_matrices'][cm_key]
        cm_matrix = np.array(cm_data['matrix'])

        # Create heatmap
        sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='Blues', ax=ax,
                   xticklabels=['Predicted: 0', 'Predicted: 1'],
                   yticklabels=['Actual: 0', 'Actual: 1'])

        ax.set_title(f"A.{i+1}: {criteria_data['symptom'][:15]}...", fontsize=10)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')

    plt.tight_layout()
    confusion_matrix_path = os.path.join(output_dir, 'confusion_matrices.png')
    plt.savefig(confusion_matrix_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Confusion matrix plots saved to {confusion_matrix_path}")

    # Also save individual confusion matrices as JSON for easy access
    cm_json_path = os.path.join(output_dir, 'confusion_matrices.json')
    with open(cm_json_path, 'w') as f:
        json.dump(metrics['confusion_matrices'], f, indent=2)
    print(f"Confusion matrix data saved to {cm_json_path}")

def print_metrics_summary(metrics):
    print("\n" + "="*80)
    print("MULTI-BINARY CLASSIFICATION EVALUATION SUMMARY")
    print("="*80)
    print("Each DSM-5 criterion evaluated as independent binary classification")

    print("\nOverall Performance (Across All Criteria):")
    print("-"*50)
    for key, value in metrics['overall'].items():
        if isinstance(value, float):
            print(f"{key:30s}: {value:.4f}")
        else:
            print(f"{key:30s}: {value}")

    print("\nPer-Criterion Binary Classification Performance:")
    print("-"*120)
    print(f"{'Criteria':<8} {'Symptom':<20} {'Acc':<6} {'Prec':<6} {'Rec':<6} {'F1':<6} {'AUC':<6} {'Spec':<6} {'Supp':<6} {'TP':<4} {'TN':<4} {'FP':<4} {'FN':<4}")
    print("-"*120)

    for criteria in metrics['per_criteria']:
        auc_str = f"{criteria['auc']:.3f}" if criteria['auc'] is not None else "N/A"
        print(f"{criteria['criteria']:<8} {criteria['symptom'][:18]:<20} "
              f"{criteria['accuracy']:<6.3f} {criteria['precision']:<6.3f} "
              f"{criteria['recall']:<6.3f} {criteria['f1']:<6.3f} "
              f"{auc_str:<6} {criteria['specificity']:<6.3f} "
              f"{criteria['support']:<6} {criteria['true_positive']:<4} "
              f"{criteria['true_negative']:<4} {criteria['false_positive']:<4} "
              f"{criteria['false_negative']:<4}")

    print(f"\nAbbreviations:")
    print(f"Acc=Accuracy, Prec=Precision, Rec=Recall, Spec=Specificity, Supp=Support")
    print(f"TP=True Positive, TN=True Negative, FP=False Positive, FN=False Negative")
    print(f"\nNote: Each criterion is independently classified as Present (1) or Absent (0)")
    print(f"Multiple criteria can be predicted as present for a single post.")

def create_evaluation_report(checkpoint, metrics, args, output_dir):
    """Create a comprehensive evaluation report with model and checkpoint information"""
    from datetime import datetime

    report_path = os.path.join(output_dir, 'evaluation_report.txt')

    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("COMPREHENSIVE EVALUATION REPORT\n")
        f.write("DSM-5 Major Depressive Disorder Criteria Classification\n")
        f.write("="*80 + "\n\n")

        # Model and checkpoint information
        f.write("MODEL INFORMATION:\n")
        f.write("-"*40 + "\n")
        f.write(f"Checkpoint file: {os.path.basename(args.checkpoint_path)}\n")
        f.write(f"Model architecture: BERT Large\n")
        f.write(f"Base model: {checkpoint.get('config', {}).get('model_name', 'N/A')}\n")
        f.write(f"Training epoch: {checkpoint.get('epoch', 'N/A')}\n")
        f.write(f"Prediction threshold: {args.threshold}\n")
        f.write(f"Evaluation date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        # Training configuration
        if 'config' in checkpoint:
            config = checkpoint['config']
            f.write("TRAINING CONFIGURATION:\n")
            f.write("-"*40 + "\n")
            f.write(f"Batch size: {config.get('batch_size', 'N/A')}\n")
            f.write(f"Learning rate: {config.get('learning_rate', 'N/A')}\n")
            f.write(f"Number of epochs: {config.get('num_epochs', 'N/A')}\n")
            f.write(f"Patience: {config.get('patience', 'N/A')}\n")
            f.write(f"Gradient accumulation steps: {config.get('gradient_accumulation_steps', 'N/A')}\n")
            f.write(f"Mixed precision: {config.get('use_amp', 'N/A')}\n\n")

        # Validation metrics from training
        if 'metrics' in checkpoint:
            val_metrics = checkpoint['metrics']
            f.write("VALIDATION METRICS (from training):\n")
            f.write("-"*40 + "\n")
            f.write(f"F1 Macro: {val_metrics.get('f1_macro', 'N/A'):.4f}\n")
            f.write(f"F1 Micro: {val_metrics.get('f1_micro', 'N/A'):.4f}\n")
            f.write(f"Exact Match Ratio: {val_metrics.get('exact_match_ratio', 'N/A'):.4f}\n")
            f.write(f"Hamming Loss: {val_metrics.get('hamming_loss', 'N/A'):.4f}\n\n")

        # Test set evaluation results
        f.write("TEST SET EVALUATION RESULTS:\n")
        f.write("-"*40 + "\n")
        f.write("Overall Performance:\n")
        for key, value in metrics['overall'].items():
            if isinstance(value, float):
                f.write(f"  {key:25s}: {value:.4f}\n")
            else:
                f.write(f"  {key:25s}: {value}\n")

        f.write("\nPer-Criterion Performance:\n")
        f.write("  " + "-"*80 + "\n")
        f.write(f"  {'Criteria':<8} {'Symptom':<20} {'Acc':<6} {'Prec':<6} {'Rec':<6} {'F1':<6} {'AUC':<6} {'Spec':<6} {'Supp':<6}\n")
        f.write("  " + "-"*80 + "\n")

        for criteria in metrics['per_criteria']:
            auc_str = f"{criteria['auc']:.3f}" if criteria['auc'] is not None else "N/A"
            f.write(f"  {criteria['criteria']:<8} {criteria['symptom'][:18]:<20} "
                   f"{criteria['accuracy']:<6.3f} {criteria['precision']:<6.3f} "
                   f"{criteria['recall']:<6.3f} {criteria['f1']:<6.3f} "
                   f"{auc_str:<6} {criteria['specificity']:<6.3f} "
                   f"{criteria['support']:<6}\n")

        f.write("\n" + "="*80 + "\n")
        f.write("Generated by DSM-5 Criteria Classification System\n")
        f.write("="*80 + "\n")

    print(f"Comprehensive evaluation report saved to {report_path}")

def main():
    parser = argparse.ArgumentParser(description='Predict DSM-5 Criteria using trained BERT model')
    parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--posts_path', type=str, default='Data/redsm5/redsm5_posts.csv')
    parser.add_argument('--annotations_path', type=str, default='Data/redsm5/redsm5_annotations.csv')
    parser.add_argument('--criteria_path', type=str, default='Data/DSM-5/DSM_Criteria_Array_Fixed_Major_Depressive.json')
    parser.add_argument('--model_name', type=str, default='google-bert/bert-large-uncased-whole-word-masking-finetuned-squad')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--output_dir', type=str, default='outputs')
    parser.add_argument('--test_only', action='store_true', help='Evaluate on test set only')

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print("Loading checkpoint...")
    checkpoint = load_checkpoint(args.checkpoint_path, device)

    print("Loading model...")
    model = BERTForDSM5Classification(args.model_name, num_criteria=9)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)

    print("Loading and preprocessing data...")
    df = prepare_data(args.posts_path, args.annotations_path, args.criteria_path)
    train_df, val_df, test_df = split_data(df)

    _, _, test_dataset, tokenizer = create_datasets(
        train_df, val_df, test_df, args.model_name
    )

    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    print("Evaluating model...")
    predictions, probs, labels = evaluate_model(model, test_loader, device, args.threshold)

    symptom_names = list(create_symptom_mapping().values())
    metrics = calculate_detailed_metrics(labels, predictions, probs, symptom_names)

    print_metrics_summary(metrics)

    os.makedirs(args.output_dir, exist_ok=True)

    predictions_path = os.path.join(args.output_dir, 'predictions.csv')
    results_df = save_predictions(test_df, predictions, labels, probs, predictions_path, args.threshold)

    # Save confusion matrix visualizations
    save_confusion_matrix_plots(metrics, args.output_dir)

    # Create comprehensive evaluation report
    create_evaluation_report(checkpoint, metrics, args, args.output_dir)

    metrics_path = os.path.join(args.output_dir, 'evaluation_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to {metrics_path}")

    summary_path = os.path.join(args.output_dir, 'evaluation_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("EVALUATION METRICS SUMMARY\n")
        f.write("="*80 + "\n\n")
        f.write("Overall Performance:\n")
        f.write("-"*50 + "\n")
        for key, value in metrics['overall'].items():
            if isinstance(value, float):
                f.write(f"{key:30s}: {value:.4f}\n")
            else:
                f.write(f"{key:30s}: {value}\n")

        f.write("\nPer-Criteria Performance:\n")
        f.write("-"*80 + "\n")
        f.write(f"{'Criteria':<8} {'Symptom':<20} {'Acc':<6} {'Prec':<6} {'Rec':<6} {'F1':<6} {'AUC':<6} {'Spec':<6} {'Support':<8}\n")
        f.write("-"*80 + "\n")
        for criteria in metrics['per_criteria']:
            auc_str = f"{criteria['auc']:.3f}" if criteria['auc'] is not None else "N/A"
            f.write(f"{criteria['criteria']:<8} {criteria['symptom'][:18]:<20} "
                   f"{criteria['accuracy']:<6.3f} {criteria['precision']:<6.3f} "
                   f"{criteria['recall']:<6.3f} {criteria['f1']:<6.3f} "
                   f"{auc_str:<6} {criteria['specificity']:<6.3f} "
                   f"{criteria['support']:<8}\n")

        f.write("\nConfusion Matrix Details:\n")
        f.write("-"*50 + "\n")
        for criteria in metrics['per_criteria']:
            f.write(f"\n{criteria['criteria']} - {criteria['symptom']}:\n")
            f.write(f"  Accuracy: {criteria['accuracy']:.3f}\n")
            f.write(f"  Precision: {criteria['precision']:.3f}\n")
            f.write(f"  Recall: {criteria['recall']:.3f}\n")
            f.write(f"  F1: {criteria['f1']:.3f}\n")
            f.write(f"  Specificity: {criteria['specificity']:.3f}\n")
            f.write(f"  AUC: {criteria['auc']:.3f if criteria['auc'] is not None else 'N/A'}\n")
            f.write(f"  Support: {criteria['support']}\n")
            f.write(f"  True Positive: {criteria['true_positive']}\n")
            f.write(f"  True Negative: {criteria['true_negative']}\n")
            f.write(f"  False Positive: {criteria['false_positive']}\n")
            f.write(f"  False Negative: {criteria['false_negative']}\n")

    print(f"Summary saved to {summary_path}")

if __name__ == "__main__":
    main()