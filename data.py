import json
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


def load_dsm5_criteria(criteria_path: str) -> Dict[str, str]:
    """Load DSM-5 criteria JSON into a mapping {"A.1": text, ...}."""
    with open(criteria_path, "r") as f:
        data = json.load(f)

    criteria_map: Dict[str, str] = {}
    for item in data:
        for criterion in item.get("criteria", []):
            criteria_map[criterion["id"]] = criterion["text"]
    return criteria_map


def create_symptom_mapping() -> Dict[str, str]:
    """Map criteria IDs to dataset symptom labels used in annotations."""
    return {
        "A.1": "DEPRESSED_MOOD",
        "A.2": "ANHEDONIA",
        "A.3": "APPETITE_CHANGE",
        "A.4": "SLEEP_ISSUES",
        "A.5": "PSYCHOMOTOR",
        "A.6": "FATIGUE",
        "A.7": "WORTHLESSNESS",
        "A.8": "COGNITIVE_ISSUES",
        "A.9": "SUICIDAL_THOUGHTS",
    }


def build_post_level_labels(posts_path: str, annotations_path: str) -> pd.DataFrame:
    """Return a dataframe with columns: post_id, text, and a 9-dim labels array."""
    posts_df = pd.read_csv(posts_path)
    ann_df = pd.read_csv(annotations_path)

    symptom_mapping = create_symptom_mapping()
    symptom_to_idx = {symptom: int(cid.split(".")[1]) - 1 for cid, symptom in symptom_mapping.items()}

    out_rows = []
    for post_id, post_text in posts_df[["post_id", "text"]].itertuples(index=False):
        labels = np.zeros(9, dtype=np.float32)
        subset = ann_df[(ann_df["post_id"] == post_id) & (ann_df["status"] == 1)]
        for _, r in subset.iterrows():
            symptom = str(r["DSM5_symptom"]).upper()
            if symptom in symptom_to_idx:
                labels[symptom_to_idx[symptom]] = 1.0
        out_rows.append({"post_id": post_id, "text": post_text, "labels": labels})

    return pd.DataFrame(out_rows)


def expand_to_pairs(df: pd.DataFrame, criteria_map: Dict[str, str]) -> pd.DataFrame:
    """Expand each post to 9 (post, criterion) rows with binary label y for that criterion.

    Returns columns: post_id, text, criteria_id, criteria_text, y
    """
    rows = []
    criteria_ids = sorted(criteria_map.keys(), key=lambda x: int(x.split(".")[1]))

    for _, row in df.iterrows():
        labels = row["labels"]
        for i, cid in enumerate(criteria_ids):
            rows.append({
                "post_id": row["post_id"],
                "text": row["text"],
                "criteria_id": cid,
                "criteria_text": criteria_map[cid],
                "criteria_idx": i,
                "y": float(labels[i]),
            })
    return pd.DataFrame(rows)


class CriteriaPairDataset(Dataset):
    """Tokenize (post, criterion) as a pair: [CLS] post [SEP] criterion [SEP]."""

    def __init__(self, posts: List[str], criteria: List[str], labels: List[float], criterion_indices: List[int], post_ids: List[str], tokenizer_name: str,
                 max_length: int = 512):
        self.posts = posts
        self.criteria = criteria
        self.labels = labels
        self.criterion_indices = criterion_indices
        self.post_ids = post_ids
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        text = str(self.posts[idx])
        criterion = str(self.criteria[idx])
        label = float(self.labels[idx])
        cidx = int(self.criterion_indices[idx])
        post_id = str(self.post_ids[idx])

        enc = self.tokenizer(
            text,
            criterion,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.float32),
            "criterion_idx": torch.tensor(cidx, dtype=torch.long),
            "post_id": post_id,
        }


def make_pairwise_datasets(posts_path: str,
                           annotations_path: str,
                           criteria_path: str,
                           tokenizer_name: str = "bert-base-uncased",
                           train_frac: float = 0.8,
                           val_frac: float = 0.1,
                           seed: int = 42,
                           max_length: int = 512):
    """Build train/val/test CriteriaPairDataset objects from inputs."""
    rng = np.random.default_rng(seed)

    post_df = build_post_level_labels(posts_path, annotations_path)
    criteria_map = load_dsm5_criteria(criteria_path)
    pairs_df = expand_to_pairs(post_df, criteria_map)

    # Shuffle by post to avoid leakage across splits
    unique_posts = post_df["post_id"].values
    rng.shuffle(unique_posts)
    n = len(unique_posts)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    train_ids = set(unique_posts[:n_train])
    val_ids = set(unique_posts[n_train:n_train + n_val])
    test_ids = set(unique_posts[n_train + n_val:])

    def to_ds(sub_df: pd.DataFrame):
        return CriteriaPairDataset(
            posts=sub_df["text"].tolist(),
            criteria=sub_df["criteria_text"].tolist(),
            labels=sub_df["y"].tolist(),
            criterion_indices=sub_df["criteria_idx"].tolist(),
            post_ids=sub_df["post_id"].tolist(),
            tokenizer_name=tokenizer_name,
            max_length=max_length,
        )

    train_df = pairs_df[pairs_df["post_id"].isin(train_ids)].reset_index(drop=True)
    val_df = pairs_df[pairs_df["post_id"].isin(val_ids)].reset_index(drop=True)
    test_df = pairs_df[pairs_df["post_id"].isin(test_ids)].reset_index(drop=True)

    return to_ds(train_df), to_ds(val_df), to_ds(test_df), criteria_map
