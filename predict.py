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

from data_preprocessing import prepare_data, split_data, create_datasets, create_symptom_mapping
from model import SpanBERTForDSM5Classification

def load_checkpoint(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    return checkpoint

def evaluate_model(model, test_loader, device, threshold=0.5):
    model.eval()
    all_predictions = []
    all_probs = []
    all_labels = []
    all_post_ids = []

    with torch.no_grad():
        progress_bar = tqdm(test_loader, desc='Evaluating')
        for batch_idx, batch in enumerate(progress_bar):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            logits = model(input_ids, attention_mask)
            probs = torch.sigmoid(logits)
            predictions = (probs > threshold).float()

            all_predictions.extend(predictions.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return np.array(all_predictions), np.array(all_probs), np.array(all_labels)

def calculate_detailed_metrics(labels, predictions, probs, symptom_names):
    metrics = {}

    metrics['overall'] = {
        'hamming_loss': hamming_loss(labels, predictions),
        'exact_match_ratio': accuracy_score(labels, predictions)
    }

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

    try:
        auc_macro = roc_auc_score(labels, probs, average='macro')
        auc_micro = roc_auc_score(labels, probs, average='micro')
        metrics['overall']['auc_macro'] = auc_macro
        metrics['overall']['auc_micro'] = auc_micro
    except:
        pass

    metrics['per_criteria'] = []
    for i, symptom in enumerate(symptom_names):
        precision, recall, f1, support = precision_recall_fscore_support(
            labels[:, i], predictions[:, i], average='binary', zero_division=0
        )

        criteria_metrics = {
            'criteria': f'A.{i+1}',
            'symptom': symptom,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'support': int(np.sum(labels[:, i])),
            'predicted_positive': int(np.sum(predictions[:, i])),
            'true_positive': int(np.sum((labels[:, i] == 1) & (predictions[:, i] == 1))),
            'false_positive': int(np.sum((labels[:, i] == 0) & (predictions[:, i] == 1))),
            'false_negative': int(np.sum((labels[:, i] == 1) & (predictions[:, i] == 0)))
        }

        try:
            auc = roc_auc_score(labels[:, i], probs[:, i])
            criteria_metrics['auc'] = auc
        except:
            criteria_metrics['auc'] = None

        metrics['per_criteria'].append(criteria_metrics)

    return metrics

def save_predictions(test_df, predictions, labels, probs, output_path):
    symptom_mapping = create_symptom_mapping()
    results = []

    for idx, (_, row) in enumerate(test_df.iterrows()):
        post_result = {
            'post_id': row['post_id'],
            'text': row['text'][:200] + '...' if len(row['text']) > 200 else row['text']
        }

        for i, (criteria_id, symptom) in enumerate(symptom_mapping.items()):
            post_result[f'{criteria_id}_predicted'] = int(predictions[idx][i])
            post_result[f'{criteria_id}_groundtruth'] = int(labels[idx][i])
            post_result[f'{criteria_id}_probability'] = float(probs[idx][i])
            post_result[f'{criteria_id}_symptom'] = symptom

        results.append(post_result)

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")

    return results_df

def print_metrics_summary(metrics):
    print("\n" + "="*50)
    print("EVALUATION METRICS SUMMARY")
    print("="*50)

    print("\nOverall Performance:")
    print("-"*30)
    for key, value in metrics['overall'].items():
        if isinstance(value, float):
            print(f"{key:20s}: {value:.4f}")
        else:
            print(f"{key:20s}: {value}")

    print("\nPer-Criteria Performance:")
    print("-"*30)
    print(f"{'Criteria':<10} {'Symptom':<20} {'Precision':<10} {'Recall':<10} {'F1':<10} {'Support':<10}")
    print("-"*70)

    for criteria in metrics['per_criteria']:
        print(f"{criteria['criteria']:<10} {criteria['symptom']:<20} "
              f"{criteria['precision']:<10.3f} {criteria['recall']:<10.3f} "
              f"{criteria['f1']:<10.3f} {criteria['support']:<10}")

def main():
    parser = argparse.ArgumentParser(description='Predict DSM-5 Criteria using trained SpanBERT model')
    parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--posts_path', type=str, default='Data/redsm5/redsm5_posts.csv')
    parser.add_argument('--annotations_path', type=str, default='Data/redsm5/redsm5_annotations.csv')
    parser.add_argument('--criteria_path', type=str, default='Data/DSM-5/DSM_Criteria_Array_Fixed_Major_Depressive.json')
    parser.add_argument('--model_name', type=str, default='SpanBERT/spanbert-base-cased')
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
    model = SpanBERTForDSM5Classification(args.model_name, num_criteria=9)
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
    results_df = save_predictions(test_df, predictions, labels, probs, predictions_path)

    metrics_path = os.path.join(args.output_dir, 'evaluation_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to {metrics_path}")

    summary_path = os.path.join(args.output_dir, 'evaluation_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("EVALUATION METRICS SUMMARY\n")
        f.write("="*50 + "\n\n")
        f.write("Overall Performance:\n")
        f.write("-"*30 + "\n")
        for key, value in metrics['overall'].items():
            f.write(f"{key:20s}: {value:.4f}\n")

        f.write("\nPer-Criteria Performance:\n")
        f.write("-"*30 + "\n")
        for criteria in metrics['per_criteria']:
            f.write(f"\n{criteria['criteria']} - {criteria['symptom']}:\n")
            f.write(f"  Precision: {criteria['precision']:.3f}\n")
            f.write(f"  Recall: {criteria['recall']:.3f}\n")
            f.write(f"  F1: {criteria['f1']:.3f}\n")
            f.write(f"  Support: {criteria['support']}\n")

    print(f"Summary saved to {summary_path}")

if __name__ == "__main__":
    main()