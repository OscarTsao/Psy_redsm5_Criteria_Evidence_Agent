import pandas as pd
import json
import numpy as np
from typing import Dict, List, Tuple
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split

class DSM5CriteriaDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        labels = self.labels[idx]

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.FloatTensor(labels)
        }

def load_dsm5_criteria(json_path: str) -> Dict[str, str]:
    with open(json_path, 'r') as f:
        data = json.load(f)

    criteria_map = {}
    for item in data:
        for criterion in item['criteria']:
            criteria_map[criterion['id']] = criterion['text']

    return criteria_map

def create_symptom_mapping() -> Dict[str, str]:
    return {
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

def prepare_data(posts_path: str, annotations_path: str, criteria_path: str):
    posts_df = pd.read_csv(posts_path)
    annotations_df = pd.read_csv(annotations_path)
    criteria_map = load_dsm5_criteria(criteria_path)
    symptom_mapping = create_symptom_mapping()

    symptom_to_criteria = {v: k for k, v in symptom_mapping.items()}

    processed_data = []

    for post_id in posts_df['post_id'].unique():
        post_text = posts_df[posts_df['post_id'] == post_id]['text'].values[0]

        labels = np.zeros(9)

        post_annotations = annotations_df[
            (annotations_df['post_id'] == post_id) &
            (annotations_df['status'] == 1)
        ]

        for _, ann in post_annotations.iterrows():
            symptom = ann['DSM5_symptom'].upper()
            if symptom in symptom_to_criteria:
                criteria_id = symptom_to_criteria[symptom]
                criteria_idx = int(criteria_id.split('.')[1]) - 1
                labels[criteria_idx] = 1

        processed_data.append({
            'post_id': post_id,
            'text': post_text,
            'labels': labels
        })

    return pd.DataFrame(processed_data)

def split_data(df: pd.DataFrame, test_size: float = 0.2, val_size: float = 0.1, random_state: int = 42):
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)
    train_df, val_df = train_test_split(train_df, test_size=val_size/(1-test_size), random_state=random_state)

    return train_df, val_df, test_df

def create_datasets(train_df, val_df, test_df, tokenizer_name='SpanBERT/spanbert-base-cased'):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    train_dataset = DSM5CriteriaDataset(
        train_df['text'].values,
        np.vstack(train_df['labels'].values),
        tokenizer
    )

    val_dataset = DSM5CriteriaDataset(
        val_df['text'].values,
        np.vstack(val_df['labels'].values),
        tokenizer
    )

    test_dataset = DSM5CriteriaDataset(
        test_df['text'].values,
        np.vstack(test_df['labels'].values),
        tokenizer
    )

    return train_dataset, val_dataset, test_dataset, tokenizer

if __name__ == "__main__":
    posts_path = "Data/redsm5/redsm5_posts.csv"
    annotations_path = "Data/redsm5/redsm5_annotations.csv"
    criteria_path = "Data/DSM-5/DSM_Criteria_Array_Fixed_Major_Depressive.json"

    df = prepare_data(posts_path, annotations_path, criteria_path)
    print(f"Total posts: {len(df)}")
    print(f"Label distribution:\n{np.sum(np.vstack(df['labels'].values), axis=0)}")

    train_df, val_df, test_df = split_data(df)
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")