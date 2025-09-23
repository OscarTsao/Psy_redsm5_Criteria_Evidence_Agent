# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a BERT-based pairwise classification system for matching DSM-5 Major Depressive Disorder criteria to Reddit posts from the RedSM5 dataset. The system uses a pairwise approach with input format [CLS]post[SEP]criteria[SEP] to classify each post-criteria pair into 9 DSM-5 criteria (A.1 to A.9) for depression diagnosis.

## Essential Commands

### Setup and Installation
```bash
pip install -r requirements.txt
```

### Testing Setup
```bash
python test_setup.py
```
This validates pairwise data loading, model initialization, and forward pass with the cleaned structure.

### Training
```bash
python train.py \
    --posts_path Data/redsm5/redsm5_posts.csv \
    --annotations_path Data/redsm5/redsm5_annotations.csv \
    --criteria_path Data/DSM-5/DSM_Criteria_Array_Fixed_Major_Depressive.json \
    --num_epochs 20 \
    --batch_size 16 \
    --learning_rate 2e-5 \
    --use_focal_loss \
    --output_dir outputs
```

### Prediction and Evaluation
```bash
python predict.py \
    --checkpoint_path outputs/best_model.pt \
    --posts_path Data/redsm5/redsm5_posts.csv \
    --annotations_path Data/redsm5/redsm5_annotations.csv \
    --criteria_path Data/DSM-5/DSM_Criteria_Array_Fixed_Major_Depressive.json \
    --threshold 0.5 \
    --output_dir outputs
```

## Architecture Overview

### Core Components
- **`model.py`**: BERT model with pairwise classification head (2-layer MLP + dropout)
- **`data.py`**: Pairwise data loading, preprocessing, and dataset creation utilities
- **`train.py`**: Training pipeline with early stopping and checkpointing
- **`predict.py`**: Inference and evaluation with detailed metrics

### Data Flow
1. Raw data (`redsm5_posts.csv`, `redsm5_annotations.csv`) → preprocessing
2. DSM-5 criteria mapping from JSON → symptom labels
3. Expand to (post, criterion) pairs → tokenization as [CLS]post[SEP]criteria[SEP]
4. BERT embeddings → pairwise binary classification for each (post, criterion) pair

### Model Architecture
- Base: BERT (bert-base-uncased)
- Classification Head: Linear(768→256) → ReLU → Dropout → Linear(256→1)
- Loss: Binary Cross-Entropy or Focal Loss (for class imbalance)
- Output: Sigmoid probability for each (post, criterion) pair

### Key Features
- Pairwise classification for 9 DSM-5 criteria using [CLS]post[SEP]criteria[SEP] format
- Focal loss support for imbalanced datasets
- Early stopping with patience-based training
- Comprehensive evaluation metrics (F1, AUC, precision, recall, accuracy)
- Per-criterion evaluation and aggregation
- Mixed precision training with gradient accumulation
- Gradient clipping and model compilation support

## Data Structure

```
Data/
├── redsm5/
│   ├── redsm5_posts.csv          # Reddit posts text data
│   └── redsm5_annotations.csv    # Ground truth annotations
├── DSM-5/
│   └── DSM_Criteria_Array_Fixed_Major_Depressive.json  # Criteria definitions
└── groundtruth/                  # Additional reference data
```

## Output Files

### Training Outputs
- `best_model.pt`: Best model checkpoint based on validation F1 score
- `history.json`: Per-epoch training/validation metrics and loss

### Evaluation Outputs
- `test_raw_pairs.json`: Raw pairwise predictions, probabilities, and labels
- `test_metrics.json`: Overall and per-criterion evaluation metrics

## DSM-5 Criteria Mapping

1. A.1: Depressed mood → `DEPRESSED_MOOD`
2. A.2: Anhedonia → `ANHEDONIA`
3. A.3: Appetite change → `APPETITE_CHANGE`
4. A.4: Sleep issues → `SLEEP_ISSUES`
5. A.5: Psychomotor changes → `PSYCHOMOTOR`
6. A.6: Fatigue → `FATIGUE`
7. A.7: Worthlessness → `WORTHLESSNESS`
8. A.8: Cognitive issues → `COGNITIVE_ISSUES`
9. A.9: Suicidal thoughts → `SUICIDAL_THOUGHTS`

## Development Environment

### VSCode Dev Container Setup

This project includes a complete dev container configuration for consistent development environments.

**Prerequisites:**
- Docker installed and running
- VSCode with Dev Containers extension

**Getting Started:**
1. Open project in VSCode
2. Command Palette (Ctrl+Shift+P) → "Dev Containers: Reopen in Container"
3. Container builds automatically with all dependencies

**Container Features:**
- Python 3.11 with PyTorch (CUDA 12.1), Transformers, and ML libraries pre-installed
- GPU support with NVIDIA Docker runtime
- VSCode extensions for Python development (Pylance, Black, Jupyter)
- Git and GitHub CLI configured
- Jupyter server on port 8888
- Auto-formatting on save with Black and isort

**GPU Requirements:**
- NVIDIA GPU with CUDA support
- NVIDIA Docker runtime installed on host
- Docker configured with GPU access

### Git Workflow

**Initial Setup:**
```bash
git add .
git commit -m "Add dev container and Git setup"
git push origin spanbert-criteria-classification
```

**Development Workflow:**
```bash
# Create feature branch
git checkout -b feature/experiment-name

# Make changes and test
python test_setup.py
python train.py [args]

# Commit changes
git add .
git commit -m "Experiment: description of changes"

# Push to remote
git push origin feature/experiment-name

# Create PR when ready
gh pr create --title "Experiment: description" --body "Details..."
```

**Quick Commands:**
```bash
# Check repository status
git status

# View recent commits
git log --oneline -10

# Push current branch
git push origin $(git branch --show-current)
```