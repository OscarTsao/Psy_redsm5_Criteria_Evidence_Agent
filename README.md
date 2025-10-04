# DSM-5 Criteria Evidence Detection System

## Overview

An advanced BERT-based pairwise classification system for automated identification of DSM-5 Major Depressive Disorder criteria evidence in Reddit posts. This research system employs sophisticated machine learning techniques to analyze social media text for clinical depression symptoms.

### Key Features

- **Pairwise Classification Architecture**: Treats each (post, criterion) pair as independent binary classification
- **Advanced Loss Functions**: BCE, Focal, Adaptive Focal, and Hybrid loss variants for handling class imbalance
- **Comprehensive Hyperparameter Optimization**: Optuna-based HPO with 500+ trial support
- **Production-Ready**: Mixed precision training, model compilation, and enterprise-grade configuration management
- **Clinical Relevance**: Maps to 9 DSM-5 Major Depressive Disorder criteria (A.1-A.9)

## Project Architecture

### Core Components

| Component | Description | Key Features |
|-----------|-------------|--------------|
| **`model.py`** | BERT-based pairwise classifier | 2-layer MLP head, multiple loss functions |
| **`data.py`** | Data pipeline and preprocessing | Pairwise expansion, tokenization, dataset creation |
| **`train.py`** | Training orchestration | Hydra configs, early stopping, checkpointing |
| **`predict.py`** | Inference and evaluation | Test set prediction, metrics calculation |
| **`run_maxed_hpo.py`** | Advanced hyperparameter optimization | 500 trials, comprehensive search space |
| **`configs/`** | Configuration management | Hydra-based, reproducible experiments |
| **`Data/`** | Dataset storage | Groundtruth JSON, DSM-5 criteria definitions |

### Model Architecture

```
Input: [CLS]post_text[SEP]criterion_text[SEP]
    ↓
BERT Encoder (bert-base-uncased, 768 dimensions)
    ↓
Classification Head: 768→256→ReLU→Dropout→256→1
    ↓
Binary Classification Output (sigmoid probability)
```

## Quick Start

### 1. Environment Setup

**Requirements**: Python 3.10+, CUDA-capable GPU (recommended)

```bash
# Install dependencies
pip install -r requirements.txt

# Validate setup
python test_setup.py
```

### 2. Data Structure

The system uses two main data sources:

```
Data/
├── groundtruth/
│   └── redsm5_ground_truth.json    # Post-level symptom annotations
└── DSM-5/
    └── DSM_Criteria_Array_Fixed_Major_Depressive.json  # Criteria definitions
```

### 3. Basic Training

```bash
# Train with default settings
python train.py

# Override hyperparameters
python train.py training.num_epochs=50 train_loader.batch_size=32

# Monitor training
tail -f outputs/training/$(ls outputs/training/ | tail -1)/history.json
```

### 4. Hyperparameter Optimization

```bash
# Quick HPO (30 trials)
python train.py training=hpo

# Comprehensive HPO (500 trials)
python run_maxed_hpo.py

# Use optimized configuration
python train.py --config-path=outputs/optimization/TIMESTAMP_study/production_config.yaml
```

### 5. Prediction and Evaluation

```bash
# Generate predictions
python predict.py --run outputs/training/TIMESTAMP

# Detailed analysis
python calculate_metrics.py
```

## DSM-5 Criteria Mapping

The system classifies evidence for 9 Major Depressive Disorder criteria:

| ID | Clinical Description | Symptom Label | Examples |
|----|---------------------|---------------|----------|
| A.1 | Depressed mood | `DEPRESSED_MOOD` | "feeling sad", "empty" |
| A.2 | Loss of interest/pleasure | `ANHEDONIA` | "nothing is fun", "lost interest" |
| A.3 | Weight/appetite change | `APPETITE_CHANGE` | "can't eat", "eating too much" |
| A.4 | Sleep disturbance | `SLEEP_ISSUES` | "can't sleep", "sleeping all day" |
| A.5 | Psychomotor changes | `PSYCHOMOTOR` | "restless", "moving slowly" |
| A.6 | Fatigue/energy loss | `FATIGUE` | "exhausted", "no energy" |
| A.7 | Worthlessness/guilt | `WORTHLESSNESS` | "I'm useless", "my fault" |
| A.8 | Concentration problems | `COGNITIVE_ISSUES` | "can't focus", "indecisive" |
| A.9 | Suicidal thoughts | `SUICIDAL_THOUGHTS` | "wish I was dead" |

## Advanced Configuration

### Hydra Configuration System

The project uses Hydra for reproducible experiment management:

```
configs/
├── config.yaml                     # Main configuration
├── training/
│   ├── default.yaml                # Standard training settings
│   ├── hpo.yaml                    # Basic HPO configuration
│   └── maxed_hpo.yaml             # Advanced HPO settings
```

**Key Configuration Sections**:
- **`model`**: BERT variant, dropout, device settings
- **`training`**: Epochs, accumulation, early stopping, compilation
- **`optimizer`**: Learning rate, weight decay, scheduler type
- **`loss`**: Loss function type and parameters (α, γ, δ)
- **`hardware`**: GPU optimizations, precision settings

## Performance and Results

### Expected Performance Metrics

| Metric | Baseline (Default) | Optimized (HPO) | Hardware (RTX 3080) |
|--------|-------------------|-----------------|---------------------|
| **F1 Score** | 0.75-0.80 | 0.82-0.88 | 15-30 min/epoch |
| **Training Time** | - | - | batch_size=32 |
| **GPU Memory** | 8-16GB | 8-16GB | Mixed precision |

### Typical Optimization Results

**Best Hyperparameters** (from historical runs):
- **Learning Rate**: 1e-5 to 3e-5
- **Batch Size**: 32-96 (optimal balance)
- **Loss Function**: Adaptive Focal or Hybrid BCE+Adaptive Focal
- **Dropout**: 0.1-0.3
- **Weight Decay**: 1e-3 to 1e-2

## Hardware Requirements

| Tier | GPU Memory | System RAM | Performance |
|------|------------|------------|------------|
| **Minimum** | 8GB | 16GB | Basic training |
| **Recommended** | 16GB+ | 32GB+ | HPO + large batches |
| **Optimal** | RTX 4090/A100 | 64GB+ | Maximum throughput |

## Output Structure

### Training Outputs
```
outputs/training/YYYYMMDD_HHMMSS/
├── best_model.pt              # Best checkpoint (highest val F1)
├── checkpoint_epoch_N.pt      # Rolling window (last 5 epochs)
├── history.json              # Training curves and metrics
├── test_metrics.json         # Final evaluation results
└── config.yaml              # Resolved configuration
```

### HPO Outputs
```
outputs/optimization/YYYYMMDD_HHMMSS_study/
├── best_config.yaml           # Best hyperparameters
├── production_config.yaml     # Production-ready config
├── all_trials.csv            # Complete trial history
└── best_trial_artifacts/     # Best model files
```

## Troubleshooting

### Common Issues and Solutions

| Problem | Cause | Solution |
|---------|-------|----------|
| **CUDA OOM** | Batch size too large | Reduce batch size or enable gradient checkpointing |
| **Slow training** | I/O bottleneck | Increase `num_workers` in data loaders |
| **Poor performance** | Suboptimal hyperparameters | Run HPO or try different loss functions |
| **Crashes during HPO** | Resource exhaustion | Reduce search space or trial count |

### Performance Optimization Tips

```bash
# Enable model compilation (PyTorch 2.0+)
python train.py training.use_compile=true

# Use mixed precision training
python train.py training.amp_dtype=bfloat16

# Optimize data loading
python train.py train_loader.num_workers=8 train_loader.pin_memory=true

# Enable hardware optimizations
python train.py hardware.enable_tf32=true hardware.enable_cudnn_benchmark=true
```

## Research Applications

### Clinical Research
- **Depression Symptom Detection**: Automated screening in social media
- **Longitudinal Studies**: Track symptom evolution over time
- **Population Health**: Large-scale mental health monitoring

### Technical Applications
- **Multi-label Classification**: Template for clinical NLP tasks
- **Pairwise Learning**: Architecture for relationship modeling
- **HPO Best Practices**: Comprehensive optimization framework

## Contributing and Development

### Development Workflow
1. **Create feature branch**: `git checkout -b feature/experiment-name`
2. **Test changes**: `python test_setup.py && python test_training.py`
3. **Run experiments**: Use HPO for parameter validation
4. **Document results**: Update configs and documentation
5. **Submit PR**: Include performance metrics and analysis

### Citation

If you use this system in research, please cite:
```bibtex
@software{dsm5_criteria_detection,
  title={DSM-5 Criteria Evidence Detection System},
  author={[Your Name]},
  year={2024},
  url={https://github.com/[username]/[repository]}
}
```

---

## License

This project is licensed under the MIT License - see the LICENSE file for details.

**Disclaimer**: This system is for research purposes only and should not be used for clinical diagnosis or treatment decisions. Always consult qualified healthcare professionals for mental health concerns.
