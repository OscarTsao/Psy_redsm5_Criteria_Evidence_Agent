# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a SpanBERT-based multi-label classification system for matching DSM-5 Major Depressive Disorder criteria to Reddit posts from the RedSM5 dataset. The system classifies text into 9 DSM-5 criteria (A.1 to A.9) for depression diagnosis.

## Essential Commands

### Setup and Installation
```bash
pip install -r requirements.txt
```

### Testing Setup
```bash
python test_setup.py
```
This validates data loading, model initialization, and dataset creation with a small subset of data.

### GPU Setup Verification
```bash
python test_gpu.py
```
This tests GPU access and CUDA functionality within the container.

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
- **`model.py`**: SpanBERT model with classification head (2-layer MLP + dropout)
- **`data_preprocessing.py`**: Data loading, preprocessing, and dataset creation utilities
- **`train.py`**: Training pipeline with early stopping and checkpointing
- **`predict.py`**: Inference and evaluation with detailed metrics

### Data Flow
1. Raw data (`redsm5_posts.csv`, `redsm5_annotations.csv`) → preprocessing
2. DSM-5 criteria mapping from JSON → symptom labels
3. Text tokenization (max_length=512) → SpanBERT embeddings
4. Multi-label classification → 9 criterion predictions

### Model Architecture
- Base: SpanBERT (spanbert-base-cased)
- Classification Head: Linear(768→256) → ReLU → Dropout → Linear(256→9)
- Loss: Binary Cross-Entropy or Focal Loss (for class imbalance)
- Output: Sigmoid probabilities for each DSM-5 criterion

### Key Features
- Multi-label classification for 9 DSM-5 criteria
- Focal loss support for imbalanced datasets
- Early stopping with patience-based training
- Comprehensive evaluation metrics (F1, AUC, Hamming loss, exact match)
- Gradient clipping and learning rate scheduling

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

Training and prediction generate:
- `predictions.csv`: Post IDs, predicted labels, ground truth, probabilities
- `evaluation_metrics.json`: Detailed performance metrics
- `evaluation_summary.txt`: Human-readable metrics summary
- `training_history.json`: Per-epoch training/validation metrics
- `best_model.pt`: Best performing model checkpoint

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