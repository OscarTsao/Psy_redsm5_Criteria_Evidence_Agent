# Psy REDSM5 Criteria Evidence Agent

## Overview

- BERT-based pairwise classifier for Reddit posts and DSM-5 Major Depressive Disorder criteria.
- Uses hybrid BCE + adaptive focal loss and supports mixed precision, gradient checkpointing, and torch.compile.
- Hydra configuration and Optuna hyperparameter optimization with per-run/timestamped outputs.

## Project Layout

- `data.py` – dataset construction and tokenization utilities
- `model.py` – model architecture and hybrid loss definitions
- `train.py` – Hydra-driven training script with Optuna HPO
- `predict.py` – inference on the test split and CSV export
- `calculate_metrics.py` – post-processing metrics script
- `configs/` – Hydra configuration hierarchy (default + HPO overrides)
- `outputs/` – per-run artifacts and metrics

## Environment Setup

- Python 3.10 or newer recommended.
- Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```

## Data Requirements

- Expect CSV data under `Data/redsm5/` with posts and annotations and DSM criteria JSON in `Data/DSM-5/`.
- Default paths in configs:
  - `Data/redsm5/redsm5_posts.csv`
  - `Data/redsm5/redsm5_annotations.csv`
  - `Data/DSM-5/DSM_Criteria_Array_Fixed_Major_Depressive.json`

## Hydra Configuration

Primary config: `configs/config.yaml` with timestamped output directories.
Default training settings: `configs/training/default.yaml`.
Key sections:

- `model`: BERT backbone name, device, dropout.
- `train_loader`/`val_loader`/`test_loader`: batch size, shuffle, workers, prefetching, etc.
- `training`: epochs, gradient accumulation, grad clipping, AMP dtype, gradient checkpointing, compile flag.
- `optimizer`: optimizer type, learning rate, weight decay.
- `loss`: hybrid loss weighting and adaptive focal parameters.
- `hardware`: TF32, cuDNN benchmark, bf16 fallback.
- `search_space`: Optuna-tunable parameters (batch size, optimizer, dropout, amp dtype, etc.).

## Basic Training

Run with defaults:

```bash
python train.py
```

Override parameters inline:

```bash
python train.py training.num_epochs=20 optimizer.lr=1e-5 model.model_name=roberta-base
```

## Optuna Hyperparameter Optimization

Activate HPO using the provided Hydra override:

```bash
python train.py training=hpo
```

- Uses the search space defined in `configs/training/hpo.yaml`.
- Default: 30 trials maximizing validation `f1`.
- Customize trials, timeout, study storage:
  ```bash
  python train.py training=hpo optuna.n_trials=50
  python train.py training=hpo optuna.timeout=3600
  python train.py training=hpo optuna.storage="sqlite:///optuna.db"
  ```
- Per-trial outputs saved under `outputs/optuna/<timestamp>/`.

For long-running, fully expanded searches with advanced artifact export use `run_maxed_hpo.py`:

```bash
python run_maxed_hpo.py optuna.n_trials=500 optuna.storage="sqlite:///optuna_maxed_study_v3.db"
```

- Automatically resumes existing studies or auto-increments the study name when re-running.
- Writes `best_config.yaml`, `production_config.yaml`, `base_config.yaml`, and `all_trials.csv` into `outputs/optimization/<timestamp>_<study>/`.
- Mirrors Optuna pruning configuration defined in `configs/training/maxed_hpo.yaml`.

### Tuned Parameters

Optuna considers:

- Loss weights: `bce_weight`, `alpha`, `gamma`, `delta`
- Optimization: `learning_rate`, `optimizer_name`, `weight_decay`
- Data loaders: train/val/test batch sizes
- Model: classifier dropout
- Training loop: gradient accumulation steps, gradient clipping, AMP dtype, compile flag, epoch count

## Outputs and Artifacts

Each run creates a directory `outputs/YYYYMMDD_HHMMSS/` containing:

- `history.json`
- `checkpoint_epoch_<N>.pt`
- `best_model.pt`
- `test_metrics.json`
- `hydra/` configuration snapshot

## Evaluation & Metrics

- Validation metrics printed each epoch (precision, recall, f1, accuracy, AUC).
- Test metrics stored in `test_metrics.json`. Use `calculate_metrics.py` for detailed CSV analysis:
  ```bash
  python calculate_metrics.py
  ```

## Inference

Run the prediction script on the test split:

```bash
python predict.py \
  --checkpoint_path outputs/<run>/best_model.pt \
  --output_dir outputs/<run>/predictions
```

Produces `test_raw_pairs.csv` containing probabilities, predictions, and criterion IDs.

## Hardware Acceleration Tips

- Enable `training.use_compile=true` for torch.compile (PyTorch ≥ 2.0).
- Adjust `train_loader.num_workers` for I/O parallelism; defaults to CPU count.
- Use `hardware.enable_tf32=true` (default) on Ampere GPUs for faster matmuls.
- Automatic bf16 fallback when available if `hardware.use_bfloat16_if_available=true`.

## Reproducibility

- Set random seed via `seed` in Hydra config (default 42).
- Hydra logs the final resolved configuration in each run directory.

## Next Steps

- Inspect Optuna results to identify best hyperparameters; consider pinning them in a new Hydra config.
- Extend dataset or loss functions by editing `data.py` and `model.py`.
