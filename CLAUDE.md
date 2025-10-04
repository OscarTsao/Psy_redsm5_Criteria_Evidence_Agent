# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an advanced BERT-based pairwise classification system for automated identification of DSM-5 Major Depressive Disorder criteria evidence in Reddit posts from the RedSM5 dataset. The system employs a sophisticated pairwise learning approach that processes each (post, criterion) pair independently to determine criterion relevance.

### Core Architecture

**Model**: BERT-based binary classifier with 2-layer MLP head
- Base Model: BERT-base-uncased (768 hidden dimensions)
- Classification Head: 768 → 256 → ReLU → Dropout → 256 → 1
- Input Format: `[CLS]post_text[SEP]criterion_text[SEP]`
- Output: Single logit per pair (sigmoid for probability)

**Advanced Loss Functions**:
- Binary Cross-Entropy (BCE)
- Focal Loss (α, γ parameters for hard example focus)
- Adaptive Focal Loss (additional δ parameter for dynamic weighting)
- Hybrid Loss (BCE + Focal/Adaptive Focal combinations)
- Weighted BCE (class imbalance handling)

**Training Optimizations**:
- Mixed precision training (AMP) with configurable dtypes
- Gradient accumulation and clipping
- Model compilation (PyTorch 2.0+)
- Early stopping with patience
- Learning rate scheduling (Plateau, Cosine, Linear, Exponential)
- Checkpoint management (best + last 5 epochs)

### Research Method

**Problem**: Automated detection of DSM-5 depression criteria evidence in social media text
**Approach**: Pairwise binary classification treating each (post, criterion) as independent classification task
**Target**: 9 DSM-5 Major Depressive Disorder criteria (A.1-A.9)
**Dataset**: RedSM5 groundtruth annotations with post-level symptom labels

## Workflow and Process

### 1. Data Processing Pipeline
```
Raw Data → Pairwise Expansion → Tokenization → Training/Validation/Test Splits
```

**Data Sources**:
- `Data/groundtruth/redsm5_ground_truth.json`: Post-level annotations with symptom labels
- `Data/DSM-5/DSM_Criteria_Array_Fixed_Major_Depressive.json`: Criterion text definitions

**Processing Steps**:
1. Load groundtruth posts with symptom annotations
2. Expand each post into 9 (post, criterion) pairs
3. Create binary labels: 1 if post contains evidence for criterion, 0 otherwise
4. Tokenize pairs as `[CLS]post[SEP]criterion[SEP]` format
5. Split into train/val/test (configurable ratios)

### 2. Model Training Process
```
Data Loading → Model Init → Training Loop → Validation → Checkpointing → Best Model Selection
```

**Training Features**:
- Hydra configuration management for reproducible experiments
- Optuna hyperparameter optimization with pruning
- Early stopping based on validation F1 score
- Mixed precision training for memory efficiency
- Gradient accumulation for large effective batch sizes
- Learning rate scheduling with multiple strategies

### 3. Evaluation and Analysis
```
Model Inference → Pairwise Predictions → Aggregation → Metrics Calculation → Detailed Analysis
```

**Evaluation Metrics**:
- Binary classification: Precision, Recall, F1, Accuracy, AUC
- Per-criterion analysis and overall aggregated performance
- Confusion matrices and detailed error analysis

## Essential Commands

### Setup and Installation
```bash
pip install -r requirements.txt
```

### Testing and Validation
```bash
# Validate complete setup and data pipeline
python test_setup.py

# Test training loop functionality
python test_training.py
```

### Training Workflows

#### Basic Training
```bash
# Standard training with default configuration (Adaptive Focal Loss)
python train.py

# Override specific parameters
python train.py training.num_epochs=50 train_loader.batch_size=32 optimizer.lr=2e-5

# Train with Binary Cross-Entropy (BCE) loss
python train.py --config-name=bce

# Train with BCE loss using manual override
python train.py loss._target_=model.DynamicLossFactory.create_loss +loss.loss_type=bce

# Train with other specific loss functions (requires proper configuration)
python train.py loss._target_=model.DynamicLossFactory.create_loss +loss.loss_type=focal +loss.alpha=0.25 +loss.gamma=2.0
```

#### Hyperparameter Optimization
```bash
# Quick HPO with Hydra integration (30 trials)
python train.py training=hpo

# Comprehensive HPO with 500 trials and advanced features
python run_maxed_hpo.py

# Resume existing HPO study
python run_maxed_hpo.py optuna.study_name=existing_study_name

# Use best configuration from HPO
python train.py --config-path=outputs/optimization/TIMESTAMP_study/production_config.yaml
```

### Prediction and Analysis
```bash
# Generate predictions for test set
python predict.py --run outputs/training/20231215_143022

# Use specific checkpoint
python predict.py --run outputs/training/20231215_143022 --checkpoint checkpoint_epoch_15.pt

# Predict on different data split
python predict.py --run outputs/training/20231215_143022 --split val

# Detailed metrics analysis
python calculate_metrics.py
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

## Detailed Architecture and Implementation

### Core Components

#### Model Architecture (`model.py`)
```
Input: [CLS]post_text[SEP]criterion_text[SEP]
    ↓
BERT Encoder (bert-base-uncased, 768 dim)
    ↓
Pooler Output ([CLS] token representation)
    ↓
Dropout Layer (configurable rate)
    ↓
MLP Head: Linear(768→256) → ReLU → Dropout → Linear(256→1)
    ↓
Single Logit Output (sigmoid for probability)
```

**Loss Functions Available**:
1. **BCE**: Standard binary cross-entropy
2. **Weighted BCE**: Class-balanced with automatic pos_weight calculation
3. **Focal Loss**: α(1-p_t)^γ * BCE for hard example focus
4. **Adaptive Focal Loss**: Adds δ parameter for dynamic weighting
5. **Hybrid Losses**: BCE + Focal/Adaptive Focal combinations

#### Data Pipeline (`data.py`)
```python
# Core data flow
def create_pairwise_dataset():
    1. Load posts from groundtruth JSON
    2. Load DSM-5 criteria definitions
    3. Create symptom mapping (A.1→DEPRESSED_MOOD, etc.)
    4. Expand: 1 post → 9 (post, criterion) pairs
    5. Generate binary labels based on symptom annotations
    6. Tokenize with format: [CLS]post[SEP]criterion[SEP]
    7. Return PyTorch Dataset with input_ids, attention_mask, labels
```

#### Training Pipeline (`train.py`)
**Features**:
- Hydra configuration system with automatic timestamped outputs
- Multi-strategy learning rate scheduling
- Mixed precision training (fp16/bf16/fp32)
- Gradient accumulation and clipping
- Early stopping with validation F1 monitoring
- Checkpoint management (best + rolling window of 5)
- Model compilation for PyTorch 2.0+ acceleration

#### Hyperparameter Optimization
**Basic HPO** (`train.py training=hpo`):
- 30 trials with median pruning
- Search space: batch size, learning rate, dropout, loss parameters

**Advanced HPO** (`run_maxed_hpo.py`):
- 500 trials with hyperband pruning
- Comprehensive search space: optimizers, schedulers, architectures
- Advanced artifact management and production config generation

### Technical Implementation Details

#### Memory Optimization
- Gradient checkpointing for reduced memory usage
- Mixed precision training with automatic loss scaling
- Dynamic batch sizing based on available GPU memory
- Pin memory for faster CPU→GPU transfers

#### Hardware Acceleration
- TF32 acceleration on Ampere GPUs
- cuDNN benchmark optimization
- Multi-GPU support (experimental)
- Automatic fallback to CPU if CUDA unavailable

#### Reproducibility
- Fixed random seeds across NumPy, PyTorch, Python
- Deterministic CUDA operations (when possible)
- Configuration versioning with Hydra
- Git commit tracking in output directories

## Data Structure

```
Data/
├── groundtruth/
│   ├── redsm5_ground_truth.json  # Optimized groundtruth data (JSON format)
│   └── redsm5_ground_truth.csv   # Alternative CSV format (slower)
└── DSM-5/
    └── DSM_Criteria_Array_Fixed_Major_Depressive.json  # Criteria definitions
```

## Output Structure and File Organization

### Training Outputs (`outputs/training/YYYYMMDD_HHMMSS/`)
```
outputs/training/20231215_143022/
├── best_model.pt                    # Best checkpoint (highest val F1)
├── checkpoint_epoch_N.pt            # Rolling checkpoint window (max 5)
├── history.json                     # Per-epoch metrics and loss curves
├── test_metrics.json                # Final test set evaluation
├── test_raw_pairs.csv              # Raw pairwise predictions
├── config.yaml                     # Final resolved configuration
└── hydra/                          # Hydra configuration artifacts
    ├── config.yaml
    ├── overrides.yaml
    └── hydra.yaml
```

### Hyperparameter Optimization Outputs
#### Basic HPO (`outputs/optuna/YYYYMMDD_HHMMSS/`)
```
outputs/optuna/20231215_143022/
├── trial_*/                        # Individual trial outputs
├── best_trial/                     # Best performing trial
└── optuna_study.db                 # SQLite study database
```

#### Advanced HPO (`outputs/optimization/YYYYMMDD_HHMMSS_study/`)
```
outputs/optimization/20231215_143022_maxed_comprehensive_hpo_v2/
├── best_config.yaml                # Best hyperparameters only
├── production_config.yaml          # Production-ready full config
├── base_config.yaml               # Original configuration
├── optimization_results.json       # Complete trial history
├── all_trials.csv                 # Tabular view of all trials
└── best_trial_artifacts/          # Best model checkpoints
    ├── best_model.pt
    ├── history.json
    └── test_metrics.json
```

### Prediction Outputs
- **`test_raw_pairs.csv`**: Raw pairwise predictions with columns:
  - `post_id`: Original post identifier
  - `criterion_id`: DSM-5 criterion (A.1-A.9)
  - `criterion_text`: Full criterion description
  - `probability`: Model prediction probability [0,1]
  - `prediction`: Binary prediction (threshold=0.5)
  - `true_label`: Ground truth label

### Metrics and Analysis
- **`test_metrics.json`**: Comprehensive evaluation metrics:
  - Overall performance: F1, Precision, Recall, Accuracy, AUC
  - Per-criterion breakdown
  - Confusion matrices
  - Class distribution statistics

## DSM-5 Criteria Mapping and Clinical Context

### Criterion-to-Symptom Mapping
The system maps 9 DSM-5 Major Depressive Disorder criteria to symptom labels:

| Criterion ID | Clinical Description | Symptom Label | Examples |
|--------------|---------------------|---------------|----------|
| A.1 | Depressed mood most of the day | `DEPRESSED_MOOD` | "feeling sad", "down", "empty" |
| A.2 | Markedly diminished interest/pleasure | `ANHEDONIA` | "nothing is fun anymore", "lost interest" |
| A.3 | Significant weight/appetite change | `APPETITE_CHANGE` | "can't eat", "eating too much" |
| A.4 | Insomnia or hypersomnia | `SLEEP_ISSUES` | "can't sleep", "sleeping all day" |
| A.5 | Psychomotor agitation/retardation | `PSYCHOMOTOR` | "restless", "moving slowly" |
| A.6 | Fatigue or loss of energy | `FATIGUE` | "exhausted", "no energy" |
| A.7 | Feelings of worthlessness/guilt | `WORTHLESSNESS` | "I'm useless", "everything is my fault" |
| A.8 | Concentration/decision difficulties | `COGNITIVE_ISSUES` | "can't focus", "indecisive" |
| A.9 | Recurrent thoughts of death/suicide | `SUICIDAL_THOUGHTS` | "wish I was dead", "suicidal" |

### Clinical Significance
- **Pairwise Approach**: Each (post, criterion) pair is evaluated independently
- **Binary Classification**: Model determines if post contains evidence for specific criterion
- **Multi-label Nature**: Posts can exhibit evidence for multiple criteria simultaneously
- **Clinical Relevance**: 5+ criteria indicate Major Depressive Episode (DSM-5 diagnostic criteria)

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

## Performance Expectations and Results

### Expected Model Performance
- **Baseline F1 Score**: 0.75-0.80 (with default hyperparameters)
- **Optimized F1 Score**: 0.82-0.88 (after hyperparameter optimization)
- **Training Time**: 15-30 minutes per epoch (RTX 3080, batch_size=32)
- **Memory Usage**: 8-16GB GPU memory (varies by batch size and precision)

### Typical Hyperparameter Optimization Results
Based on historical runs:
- **Best Learning Rate**: 1e-5 to 3e-5
- **Optimal Batch Size**: 32-96 (balance of speed and performance)
- **Best Loss Function**: Adaptive Focal Loss or Hybrid BCE+Adaptive Focal
- **Optimal Dropout**: 0.1-0.3
- **Weight Decay**: 1e-3 to 1e-2

### Hardware Requirements
- **Minimum**: 8GB GPU memory, 16GB RAM
- **Recommended**: 16GB+ GPU memory, 32GB+ RAM
- **Optimal**: RTX 4080/4090 or A100, 64GB+ RAM

## Usage Guidelines

### Quick Start Workflow
1. **Setup**: `pip install -r requirements.txt`
2. **Validate**: `python test_setup.py`
3. **Train**: `python train.py`
4. **Predict**: `python predict.py --run outputs/training/TIMESTAMP`
5. **Analyze**: `python calculate_metrics.py`

### Advanced Usage
1. **Hyperparameter Optimization**: `python run_maxed_hpo.py`
2. **Production Training**: Use best config from HPO
3. **Detailed Analysis**: Examine per-criterion performance
4. **Experiment Tracking**: Use Git workflow scripts

### Troubleshooting
- **CUDA OOM**: Reduce batch size or enable gradient checkpointing
- **Slow Training**: Check data loading parallelization (`num_workers`)
- **Poor Performance**: Try different loss functions or run HPO
- **Reproducibility Issues**: Verify seed settings and deterministic flags
- **BCE Training Fails**: Use `--config-name=bce` or manual override with `loss._target_=model.DynamicLossFactory.create_loss +loss.loss_type=bce`
- **Loss Type Override Error**: Use `+loss.loss_type=X` (with +) when changing loss functions manually
