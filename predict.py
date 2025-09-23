import argparse
import json
import os
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import BERTForPairwiseClassification
from data import make_pairwise_datasets, load_dsm5_criteria


def load_checkpoint(path: str, device: torch.device):
    ckpt = torch.load(path, map_location=device)
    return ckpt


def score_pairs(model, loader, device, threshold: float = 0.5):
    model.eval()
    probs_all, preds_all, labels_all, cidx_all = [], [], [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc='Scoring'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            logits = model(input_ids, attention_mask)
            probs = torch.sigmoid(logits)
            preds = (probs > threshold).float()
            probs_all.append(probs.cpu().numpy())
            preds_all.append(preds.cpu().numpy())
            labels_all.append(labels.cpu().numpy())
            cidx_all.append(batch['criterion_idx'].cpu().numpy())

    return (
        np.concatenate(probs_all),
        np.concatenate(preds_all),
        np.concatenate(labels_all),
        np.concatenate(cidx_all),
    )


def main():
    parser = argparse.ArgumentParser(description='Predict with pairwise criteria-matching model.')
    parser.add_argument('--checkpoint_path', type=str, required=True)
    parser.add_argument('--posts_path', type=str, default='Data/redsm5/redsm5_posts.csv')
    parser.add_argument('--annotations_path', type=str, default='Data/redsm5/redsm5_annotations.csv')
    parser.add_argument('--criteria_path', type=str, default='Data/DSM-5/DSM_Criteria_Array_Fixed_Major_Depressive.json')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--output_dir', type=str, default='outputs')

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt = load_checkpoint(args.checkpoint_path, device)

    config = ckpt.get('config', {})
    model_name = config.get('model_name', 'bert-base-uncased')

    model = BERTForPairwiseClassification(model_name)
    model.load_state_dict(ckpt['model_state_dict'])
    model = model.to(device)

    # Build datasets (we use test for scoring by default)
    _, _, test_ds, criteria_map = make_pairwise_datasets(
        args.posts_path, args.annotations_path, args.criteria_path, tokenizer_name=model_name
    )
    test_loader = DataLoader(test_ds, batch_size=args.batch_size)

    probs, preds, labels, cidx = score_pairs(model, test_loader, device, threshold=args.threshold)

    # Aggregate to per-post 9 scores/preds by order of criteria id A.1..A.9
    crit_ids_sorted = sorted(criteria_map.keys(), key=lambda x: int(x.split('.')[1]))
    crit_index_map = {i: cid for i, cid in enumerate(crit_ids_sorted)}

    # Rebuild post-level view
    records = []
    # We need post ids; rebuild by recomputing pairs df order from dataset inputs size
    # The current dataset doesn’t return post_id; to export post-level CSV, we’d extend dataset to include it.
    # For now, just write per-criterion aggregates.

    out = {
        'probs': probs.tolist(),
        'preds': preds.astype(int).tolist(),
        'labels': labels.astype(int).tolist(),
        'criterion_idx': cidx.tolist(),
    }
    with open(os.path.join(args.output_dir, 'test_raw_pairs.json'), 'w') as f:
        json.dump(out, f)

    print('Saved pairwise raw outputs to test_raw_pairs.json')


if __name__ == '__main__':
    main()
