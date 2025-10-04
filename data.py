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


def build_post_level_labels_from_groundtruth(groundtruth_path: str) -> pd.DataFrame:
    """Load groundtruth data and return dataframe with columns: post_id, text, and a 9-dim labels array."""
    # Load data from JSON (more efficient based on testing)
    import json
    rows = []
    with open(groundtruth_path, 'r') as f:
        for line in f:
            rows.append(json.loads(line))
    df = pd.DataFrame(rows)

    # Define the symptom order matching the original mapping
    symptom_columns = [
        "DEPRESSED_MOOD",    # A.1
        "ANHEDONIA",         # A.2
        "APPETITE_CHANGE",   # A.3
        "SLEEP_ISSUES",      # A.4
        "PSYCHOMOTOR",       # A.5
        "FATIGUE",           # A.6
        "WORTHLESSNESS",     # A.7
        "COGNITIVE_ISSUES",  # A.8
        "SUICIDAL_THOUGHTS"  # A.9
    ]

    out_rows = []
    for _, row in df.iterrows():
        # Extract labels as numpy array in the correct order
        labels = np.array([float(row[col]) for col in symptom_columns], dtype=np.float32)
        out_rows.append({
            "post_id": row["post_id"],
            "text": row["text"],
            "labels": labels
        })

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


def make_pairwise_datasets_from_groundtruth(groundtruth_path: str,
                                          criteria_path: str,
                                          tokenizer_name: str = "bert-base-uncased",
                                          train_frac: float = 0.8,
                                          val_frac: float = 0.1,
                                          seed: int = 42,
                                          max_length: int = 512):
    """Build train/val/test CriteriaPairDataset objects from groundtruth data."""
    rng = np.random.default_rng(seed)

    post_df = build_post_level_labels_from_groundtruth(groundtruth_path)
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


def make_pairwise_datasets(groundtruth_path: str = None,
                           criteria_path: str = None,
                           posts_path: str = None,
                           annotations_path: str = None,
                           tokenizer_name: str = "bert-base-uncased",
                           train_frac: float = 0.8,
                           val_frac: float = 0.1,
                           seed: int = 42,
                           max_length: int = 512):
    """Create pairwise datasets. Now uses groundtruth data by default."""
    # Legacy arguments stay for backward compatibility; groundtruth JSON is now canonical
    if groundtruth_path is None:
        groundtruth_path = "Data/groundtruth/redsm5_ground_truth.json"
    if criteria_path is None:
        criteria_path = "Data/DSM-5/DSM_Criteria_Array_Fixed_Major_Depressive.json"

    return make_pairwise_datasets_from_groundtruth(
        groundtruth_path=groundtruth_path,
        criteria_path=criteria_path,
        tokenizer_name=tokenizer_name,
        train_frac=train_frac,
        val_frac=val_frac,
        seed=seed,
        max_length=max_length
    )
