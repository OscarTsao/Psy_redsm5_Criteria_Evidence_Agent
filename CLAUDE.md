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

### Training (Optimized)
```bash
# Basic training with optimized defaults
python train.py

# Training with hyperparameter overrides
python train.py training.num_epochs=50 train_loader.batch_size=32

# Optuna hyperparameter optimization
python run_maxed_hpo.py

# Use optimized configuration from HPO
python train.py --config-path=outputs/optimization/TIMESTAMP_study/production_config.yaml
```

### Prediction and Evaluation (Enhanced)
```bash
# Predict using best model from a training run
python predict.py --run outputs/training/20231215_143022

# Predict with specific checkpoint
python predict.py --run outputs/training/20231215_143022 --checkpoint checkpoint_epoch_15.pt

# Predict different data split
python predict.py --run outputs/training/20231215_143022 --split val
```

### Git Workflow for Experiments
```bash
# Create new experiment branch
./scripts/git_workflow.sh new-experiment focal-loss-optimization

# Commit experimental changes
./scripts/git_workflow.sh commit "Implement adaptive focal loss with early stopping"

# Push experiment branch
./scripts/git_workflow.sh push

# Check project status
./scripts/git_workflow.sh status

# Clean old training outputs
./scripts/git_workflow.sh clean-outputs
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
- Advanced loss functions: Focal Loss, Adaptive Focal Loss, and Hybrid Loss
- Intelligent checkpoint management (keeps only 5 most recent + best model)
- Early stopping with configurable patience (default: 10 epochs)
- Learning rate scheduling with ReduceLROnPlateau
- Comprehensive evaluation metrics (F1, AUC, precision, recall, accuracy)
- Per-criterion evaluation and aggregation
- Mixed precision training with gradient accumulation and TF32 optimization
- Gradient clipping and model compilation support
- Optuna hyperparameter optimization with best configuration saving
- Git-based experiment workflow management
- Hardware optimization for CUDA, cuDNN benchmark, and bfloat16 support

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
- `checkpoint_epoch_N.pt`: Regular epoch checkpoints (max 5 kept automatically)
- `history.json`: Per-epoch training/validation metrics and loss

### Evaluation Outputs
- `test_raw_pairs.csv`: Raw pairwise predictions with post_id as first column
- `test_metrics.json`: Overall and per-criterion evaluation metrics

### Optimization Outputs
- `outputs/optimization/TIMESTAMP_study/`: Optuna study results
  - `best_config.yaml`: Best hyperparameter configuration
  - `production_config.yaml`: Production-ready configuration
  - `optimization_results.json`: Complete optimization history
  - `best_trial_artifacts/`: Best model and training outputs

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
