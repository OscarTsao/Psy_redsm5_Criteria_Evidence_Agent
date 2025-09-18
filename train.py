import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
import numpy as np
from tqdm import tqdm
import argparse
import os
import json
from datetime import datetime
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, hamming_loss, roc_auc_score

from data_preprocessing import prepare_data, split_data, create_datasets, create_symptom_mapping
from model import SpanBERTForDSM5Classification, FocalLoss, get_model

class Trainer:
    def __init__(self, model, device, train_loader, val_loader, optimizer, scheduler, criterion, config):
        self.model = model
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.config = config
        self.best_val_f1 = 0
        self.patience_counter = 0

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        all_predictions = []
        all_labels = []

        progress_bar = tqdm(self.train_loader, desc='Training')
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)

            self.optimizer.zero_grad()

            logits = self.model(input_ids, attention_mask)
            loss = self.criterion(logits, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['max_grad_norm'])

            self.optimizer.step()
            self.scheduler.step()

            total_loss += loss.item()

            with torch.no_grad():
                predictions = torch.sigmoid(logits) > self.config['threshold']
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

            progress_bar.set_postfix({'loss': loss.item()})

        avg_loss = total_loss / len(self.train_loader)
        metrics = self.calculate_metrics(np.array(all_labels), np.array(all_predictions))

        return avg_loss, metrics

    def validate(self):
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_labels = []
        all_probs = []

        with torch.no_grad():
            progress_bar = tqdm(self.val_loader, desc='Validation')
            for batch in progress_bar:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                logits = self.model(input_ids, attention_mask)
                loss = self.criterion(logits, labels)

                total_loss += loss.item()

                probs = torch.sigmoid(logits)
                predictions = probs > self.config['threshold']

                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        avg_loss = total_loss / len(self.val_loader)
        metrics = self.calculate_metrics(np.array(all_labels), np.array(all_predictions), np.array(all_probs))

        return avg_loss, metrics

    def calculate_metrics(self, labels, predictions, probs=None):
        precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='macro', zero_division=0)

        precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(labels, predictions, average='micro', zero_division=0)

        h_loss = hamming_loss(labels, predictions)

        exact_match = accuracy_score(labels, predictions)

        metrics = {
            'precision_macro': precision,
            'recall_macro': recall,
            'f1_macro': f1,
            'precision_micro': precision_micro,
            'recall_micro': recall_micro,
            'f1_micro': f1_micro,
            'hamming_loss': h_loss,
            'exact_match_ratio': exact_match
        }

        if probs is not None:
            try:
                auc_macro = roc_auc_score(labels, probs, average='macro')
                auc_micro = roc_auc_score(labels, probs, average='micro')
                metrics['auc_macro'] = auc_macro
                metrics['auc_micro'] = auc_micro
            except:
                pass

        per_class_metrics = []
        symptom_names = list(create_symptom_mapping().values())
        for i in range(labels.shape[1]):
            p, r, f, _ = precision_recall_fscore_support(labels[:, i], predictions[:, i], average='binary', zero_division=0)
            per_class_metrics.append({
                'symptom': symptom_names[i] if i < len(symptom_names) else f'Criterion_{i+1}',
                'precision': p,
                'recall': r,
                'f1': f
            })

        metrics['per_class'] = per_class_metrics

        return metrics

    def train(self):
        train_history = []
        val_history = []

        for epoch in range(self.config['num_epochs']):
            print(f"\nEpoch {epoch + 1}/{self.config['num_epochs']}")

            train_loss, train_metrics = self.train_epoch()
            val_loss, val_metrics = self.validate()

            train_history.append({
                'epoch': epoch + 1,
                'loss': train_loss,
                **train_metrics
            })

            val_history.append({
                'epoch': epoch + 1,
                'loss': val_loss,
                **val_metrics
            })

            print(f"Train Loss: {train_loss:.4f}, F1 Macro: {train_metrics['f1_macro']:.4f}")
            print(f"Val Loss: {val_loss:.4f}, F1 Macro: {val_metrics['f1_macro']:.4f}")

            if val_metrics['f1_macro'] > self.best_val_f1:
                self.best_val_f1 = val_metrics['f1_macro']
                self.save_checkpoint(epoch + 1, val_metrics)
                self.patience_counter = 0
            else:
                self.patience_counter += 1

            if self.patience_counter >= self.config['patience']:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                break

        return train_history, val_history

    def save_checkpoint(self, epoch, metrics):
        checkpoint_path = os.path.join(self.config['output_dir'], 'best_model.pt')
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_f1': self.best_val_f1,
            'metrics': metrics,
            'config': self.config
        }, checkpoint_path)
        print(f"Model checkpoint saved with F1: {self.best_val_f1:.4f}")

def main():
    parser = argparse.ArgumentParser(description='Train SpanBERT for DSM-5 Criteria Classification')
    parser.add_argument('--posts_path', type=str, default='Data/redsm5/redsm5_posts.csv')
    parser.add_argument('--annotations_path', type=str, default='Data/redsm5/redsm5_annotations.csv')
    parser.add_argument('--criteria_path', type=str, default='Data/DSM-5/DSM_Criteria_Array_Fixed_Major_Depressive.json')
    parser.add_argument('--model_name', type=str, default='SpanBERT/spanbert-base-cased')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--learning_rate', type=float, default=2e-5)
    parser.add_argument('--warmup_ratio', type=float, default=0.1)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--use_focal_loss', action='store_true', help='Use Focal Loss instead of BCEWithLogitsLoss')
    parser.add_argument('--output_dir', type=str, default='outputs')
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    print("Loading and preprocessing data...")
    df = prepare_data(args.posts_path, args.annotations_path, args.criteria_path)
    train_df, val_df, test_df = split_data(df)

    print(f"Data split - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    train_dataset, val_dataset, test_dataset, tokenizer = create_datasets(
        train_df, val_df, test_df, args.model_name
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    print("Initializing model...")
    model, device = get_model(args.model_name, num_criteria=9)

    if args.use_focal_loss:
        criterion = FocalLoss()
    else:
        criterion = nn.BCEWithLogitsLoss()

    optimizer = AdamW(model.parameters(), lr=args.learning_rate)

    total_steps = len(train_loader) * args.num_epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    config = {
        'num_epochs': args.num_epochs,
        'learning_rate': args.learning_rate,
        'batch_size': args.batch_size,
        'max_grad_norm': args.max_grad_norm,
        'threshold': args.threshold,
        'patience': args.patience,
        'output_dir': args.output_dir,
        'model_name': args.model_name
    }

    trainer = Trainer(model, device, train_loader, val_loader, optimizer, scheduler, criterion, config)

    print("Starting training...")
    train_history, val_history = trainer.train()

    history_path = os.path.join(args.output_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump({
            'train': train_history,
            'validation': val_history,
            'config': config
        }, f, indent=2)

    print(f"Training completed. Best validation F1: {trainer.best_val_f1:.4f}")

if __name__ == "__main__":
    main()