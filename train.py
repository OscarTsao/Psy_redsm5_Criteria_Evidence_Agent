import argparse
import json
import os
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, accuracy_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import BERTForPairwiseClassification, get_pairwise_model, FocalLoss
from data import make_pairwise_datasets, load_dsm5_criteria


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict:
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except Exception:
        auc = None
    acc = accuracy_score(y_true, y_pred)
    return {
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'auc': None if auc is None else float(auc),
        'accuracy': float(acc),
    }


def evaluate(model, loader, device, threshold: float = 0.5, use_amp: bool = True):
    model.eval()
    all_y, all_pred, all_prob, all_cidx = [], [], [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc='Eval'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            cidx = batch['criterion_idx'].cpu().numpy()

            if use_amp:
                with autocast():
                    logits = model(input_ids, attention_mask)
            else:
                logits = model(input_ids, attention_mask)

            probs = torch.sigmoid(logits)
            preds = (probs > threshold).float()

            all_y.append(labels.cpu().numpy())
            all_pred.append(preds.cpu().numpy())
            all_prob.append(probs.cpu().numpy())
            all_cidx.append(cidx)

    y = np.concatenate(all_y)
    y_hat = np.concatenate(all_pred)
    y_prob = np.concatenate(all_prob)
    cidx = np.concatenate(all_cidx)

    overall = compute_metrics(y, y_hat, y_prob)

    # Per-criterion metrics
    per_criteria = []
    for i in range(9):
        m = compute_metrics(y[cidx == i], y_hat[cidx == i], y_prob[cidx == i])
        m['criteria_idx'] = i
        per_criteria.append(m)

    return overall, per_criteria


def main():
    parser = argparse.ArgumentParser(description='Train BERT for DSM-5 criteria matching (pairwise).')
    parser.add_argument('--posts_path', type=str, default='Data/redsm5/redsm5_posts.csv')
    parser.add_argument('--annotations_path', type=str, default='Data/redsm5/redsm5_annotations.csv')
    parser.add_argument('--criteria_path', type=str, default='Data/DSM-5/DSM_Criteria_Array_Fixed_Major_Depressive.json')
    parser.add_argument('--model_name', type=str, default='bert-base-uncased')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--learning_rate', type=float, default=2e-5)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--use_focal_loss', action='store_true')
    parser.add_argument('--use_amp', action='store_true', default=True, help='Use Automatic Mixed Precision')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='Number of steps to accumulate gradients')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
    parser.add_argument('--use_compile', action='store_true', help='Use torch.compile for optimization')
    parser.add_argument('--output_dir', type=str, default='outputs')
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print('Building datasets...')
    train_ds, val_ds, test_ds, criteria_map = make_pairwise_datasets(
        args.posts_path, args.annotations_path, args.criteria_path, tokenizer_name=args.model_name
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True if args.num_workers > 0 else False
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True if args.num_workers > 0 else False
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True if args.num_workers > 0 else False
    )

    print('Initializing model...')
    model, device = get_pairwise_model(args.model_name)

    # Enable gradient checkpointing for memory efficiency
    if hasattr(model.encoder, 'gradient_checkpointing_enable'):
        model.encoder.gradient_checkpointing_enable()

    # Compile model for optimization (PyTorch 2.0+)
    if args.use_compile and hasattr(torch, 'compile'):
        print("Compiling model with torch.compile...")
        model = torch.compile(model, mode='max-autotune')

    if args.use_focal_loss:
        criterion = FocalLoss()
    else:
        criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    scaler = GradScaler() if args.use_amp else None

    best_f1 = 0.0
    history = []

    for epoch in range(1, args.num_epochs + 1):
        model.train()
        running_loss = 0.0
        accumulated_loss = 0.0

        for step, batch in enumerate(tqdm(train_loader, desc=f'Epoch {epoch}/{args.num_epochs}')):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            if scaler:  # Mixed precision training
                with autocast():
                    logits = model(input_ids, attention_mask)
                    loss = criterion(logits, labels)
                    loss = loss / args.gradient_accumulation_steps

                scaler.scale(loss).backward()
                accumulated_loss += loss.item()

                if (step + 1) % args.gradient_accumulation_steps == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

                    running_loss += accumulated_loss
                    accumulated_loss = 0.0
            else:  # Standard training
                logits = model(input_ids, attention_mask)
                loss = criterion(logits, labels)
                loss = loss / args.gradient_accumulation_steps

                loss.backward()
                accumulated_loss += loss.item()

                if (step + 1) % args.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()

                    running_loss += accumulated_loss
                    accumulated_loss = 0.0

        train_loss = running_loss / max(1, len(train_loader) // args.gradient_accumulation_steps)
        val_overall, val_per = evaluate(model, val_loader, device, threshold=args.threshold, use_amp=args.use_amp)

        history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_overall': val_overall,
        })

        print(f"Epoch {epoch}: train_loss={train_loss:.4f} val_f1={val_overall['f1']:.4f} val_auc={val_overall['auc']}")

        if val_overall['f1'] > best_f1:
            best_f1 = val_overall['f1']
            ckpt = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'config': vars(args),
                'val_overall': val_overall,
                'val_per_criteria': val_per,
            }
            torch.save(ckpt, os.path.join(args.output_dir, 'best_model.pt'))
            with open(os.path.join(args.output_dir, 'history.json'), 'w') as f:
                json.dump(history, f, indent=2)
            print('Saved new best checkpoint.')

    print('Evaluating on test set...')
    test_overall, test_per = evaluate(model, test_loader, device, threshold=args.threshold, use_amp=args.use_amp)
    with open(os.path.join(args.output_dir, 'test_metrics.json'), 'w') as f:
        json.dump({'overall': test_overall, 'per_criteria': test_per}, f, indent=2)
    print('Done.')


if __name__ == '__main__':
    main()
