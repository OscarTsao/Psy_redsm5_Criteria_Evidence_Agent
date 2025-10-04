# Maxed Out HPO Configuration Guide

## Overview

This configuration provides comprehensive hyperparameter optimization with 500 trials, advanced pruning, and extensive search space covering:

- **Loss functions**: BCE, Weighted BCE, Focal, Adaptive Focal, Hybrid BCE+Focal variants
- **Optimizers**: AdamW and Adam with tuned betas/eps
- **Schedulers**: Plateau, Cosine, Linear, Exponential
- **Architecture**: Dropout, batch sizes, gradient settings
- **Training**: Early stopping, gradient checkpointing, accumulation

## Quick Start

### 1. Basic Maxed HPO Run
```bash
python run_maxed_hpo.py
```

### 2. Resume Previous Study
```bash
python run_maxed_hpo.py optuna.study_name=your_previous_study_name
```

### 3. Customize Settings
```bash
# Run with fewer trials for testing
python run_maxed_hpo.py optuna.n_trials=50

# Change timeout (7 days = 604800 seconds)
python run_maxed_hpo.py optuna.timeout=172800  # 2 days

# Disable pruning for complete trials
python run_maxed_hpo.py optuna.pruning.enabled=false
```

## Time & Resource Estimates

### Expected Performance
- **Per trial**: 10-25 minutes (with pruning)
- **500 trials**: 2-7 days total runtime
- **Pruning efficiency**: ~40-60% of trials pruned early
- **Memory usage**: 8-16GB GPU memory per trial

### Hardware Recommendations
- **GPU**: RTX 3080/4080 or better (16GB+ VRAM recommended)
- **CPU**: 8+ cores for data loading
- **RAM**: 32GB+ system memory
- **Storage**: 100GB+ free space for outputs

### Optimization Strategy
1. **Phase 1** (Trials 1-50): Exploration phase, all trials run
2. **Phase 2** (Trials 51-200): Aggressive pruning starts
3. **Phase 3** (Trials 201-500): Fine-tuning around best regions

## Search Space Details

### Loss Functions Explored
- **BCE**: Standard binary cross-entropy
- **Weighted BCE**: Class-balanced BCE with pos_weight tuning
- **Focal**: α and γ tuning for hard example focus
- **Adaptive Focal**: Additional δ parameter for dynamic focusing
- **Hybrid BCE Focal**: BCE + Focal combination with weight tuning
- **Hybrid BCE Adaptive Focal**: Adds delta term for adaptive focusing while keeping BCE stability

### Key Parameters Optimized
- **Batch sizes**: 8-256 (train), 32-320 (eval)
- **Learning rate**: 5e-7 to 1e-4 (log-uniform)
- **Weight decay**: 1e-7 to 5e-2 (log-uniform)
- **Dropout**: 0.0 to 0.5
- **Early stopping**: 5-25 epochs patience

## Output Structure

```
outputs/optimization/YYYYMMDD_HHMMSS_maxed_comprehensive_hpo_v2/
├── best_config.yaml              # Best hyperparameters
├── production_config.yaml        # Production-ready config
├── optimization_results.json     # Full optimization history
├── all_trials.csv                # Tabular view of every trial (number, params, metrics)
├── base_config.yaml             # Original configuration
└── best_trial_artifacts/        # Best model checkpoints
    ├── best_model.pt
    ├── history.json
    └── test_metrics.json
```

## Monitoring Progress

### Real-time Monitoring
```bash
# Watch optimization progress
tail -f outputs/optimization/*/optimization_results.json

# Monitor GPU usage
watch -n 1 nvidia-smi
```

### Intermediate Results
- Check `optimization_results.json` for trial progress
- Best parameters updated after each trial
- Pruned trials logged with reasons

## Production Usage

After optimization completes:

```bash
# Train with best configuration
python train.py --config-path=outputs/optimization/TIMESTAMP_study/production_config.yaml

# Or use best config directly
python train.py --config-path=outputs/optimization/TIMESTAMP_study/best_config.yaml
```

## Troubleshooting

### Common Issues
1. **CUDA OOM**: Reduce batch sizes in search space
2. **Slow trials**: Check if pruning is enabled
3. **Study corruption**: Use SQLite storage for persistence

### Performance Tuning
- Enable `use_compile=true` for PyTorch 2.0+
- Use `amp_dtype=bfloat16` on supported hardware
- Set `dataloader_pin_memory=true` for faster data loading

## Advanced Features

### Custom Pruning
```yaml
optuna:
  pruning:
    enabled: true
    pruner: HyperbandPruner  # More aggressive
    n_startup_trials: 30
    n_warmup_steps: 3
```

### Distributed HPO
```bash
# Multi-GPU setup (experimental)
python run_maxed_hpo.py optuna.storage=postgresql://user:pass@host/db
```

## Expected Results

### Performance Improvements
- **Baseline F1**: ~0.75-0.80
- **Expected optimized F1**: 0.82-0.88
- **Improvement**: 5-10% relative gain

### Best Parameter Ranges (Historical)
- **Learning rate**: 1e-5 to 3e-5
- **Batch size**: 32-96 (optimal balance)
- **Loss function**: Adaptive Focal or Hybrid BCE Adaptive Focal (typically best)
- **Dropout**: 0.1-0.3
- **Weight decay**: 1e-3 to 1e-2
