#!/usr/bin/env python3
"""
Enhanced Optuna HPO script with comprehensive hyperparameter search space.
Handles complex conditional parameters and advanced search strategies.
"""

from __future__ import annotations

import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

import hydra
import optuna
import yaml
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from train import run_training


def suggest_hyperparameters(trial: optuna.Trial, search_space: DictConfig) -> Dict[str, Any]:
    """Suggest hyperparameters based on the search space configuration."""
    params = {}

    for param_name, param_config in search_space.items():
        method = param_config.method

        if method == "categorical":
            params[param_name] = trial.suggest_categorical(param_name, param_config.choices)
        elif method == "uniform":
            params[param_name] = trial.suggest_uniform(param_name, param_config.low, param_config.high)
        elif method == "loguniform":
            params[param_name] = trial.suggest_loguniform(param_name, param_config.low, param_config.high)
        elif method == "int":
            params[param_name] = trial.suggest_int(param_name, param_config.low, param_config.high)
        else:
            raise ValueError(f"Unknown sampling method: {method}")

    return params


def apply_hyperparameters(cfg: DictConfig, params: Dict[str, Any]) -> DictConfig:
    """Apply suggested hyperparameters to the configuration."""
    # Create a copy to avoid modifying the original
    trial_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    OmegaConf.set_struct(trial_cfg, False)

    # Apply basic parameters
    for param_name, value in params.items():
        if param_name == "train_batch_size":
            trial_cfg.train_loader.batch_size = int(value)
        elif param_name == "eval_batch_size":
            trial_cfg.eval_batch_size = int(value)
            trial_cfg.val_loader.batch_size = int(value)
            trial_cfg.test_loader.batch_size = int(value)
        elif param_name == "learning_rate":
            trial_cfg.optimizer.lr = float(value)
        elif param_name == "weight_decay":
            trial_cfg.optimizer.weight_decay = float(value)
        elif param_name == "dropout":
            trial_cfg.model.dropout = float(value)
        elif param_name == "clip_grad_norm":
            trial_cfg.training.clip_grad_norm = float(value)
        elif param_name == "threshold":
            trial_cfg.training.threshold = float(value)
        elif param_name == "gradient_accumulation_steps":
            trial_cfg.training.gradient_accumulation_steps = int(value)
        elif param_name == "early_stopping_patience":
            trial_cfg.training.early_stopping_patience = int(value)
        elif param_name == "use_gradient_checkpointing":
            trial_cfg.training.use_grad_checkpointing = bool(value)
        elif param_name == "max_steps_per_epoch":
            trial_cfg.training.max_steps_per_epoch = value if value != "null" else None

    # Handle optimizer configuration
    if "optimizer_type" in params:
        optimizer_type = params["optimizer_type"]
        if optimizer_type == "adamw":
            trial_cfg.optimizer._target_ = "torch.optim.AdamW"
        elif optimizer_type == "adam":
            trial_cfg.optimizer._target_ = "torch.optim.Adam"
        elif optimizer_type == "rmsprop":
            trial_cfg.optimizer._target_ = "torch.optim.RMSprop"

        # Apply optimizer-specific parameters
        for param in ["beta1", "beta2", "eps"]:
            if param in params:
                trial_cfg.optimizer[param] = float(params[param])

    # Handle scheduler configuration
    if "scheduler_type" in params:
        scheduler_type = params["scheduler_type"]
        if scheduler_type == "plateau":
            trial_cfg.scheduler._target_ = "torch.optim.lr_scheduler.ReduceLROnPlateau"
            trial_cfg.scheduler.mode = "max"
            if "scheduler_patience" in params:
                trial_cfg.scheduler.patience = int(params["scheduler_patience"])
            if "scheduler_factor" in params:
                trial_cfg.scheduler.factor = float(params["scheduler_factor"])
        elif scheduler_type == "cosine":
            trial_cfg.scheduler._target_ = "torch.optim.lr_scheduler.CosineAnnealingLR"
            trial_cfg.scheduler.T_max = trial_cfg.training.num_epochs
        elif scheduler_type == "linear":
            trial_cfg.scheduler._target_ = "torch.optim.lr_scheduler.LinearLR"
            if "warmup_steps" in params:
                trial_cfg.scheduler.start_factor = 0.1
                trial_cfg.scheduler.total_iters = int(params["warmup_steps"])
        elif scheduler_type == "exponential":
            trial_cfg.scheduler._target_ = "torch.optim.lr_scheduler.ExponentialLR"
            trial_cfg.scheduler.gamma = 0.9

    # Handle loss function configuration
    if "loss_function" in params:
        loss_type = params["loss_function"]
        trial_cfg.loss._target_ = "model.DynamicLossFactory.create_loss"
        trial_cfg.loss.loss_type = loss_type

        # Apply loss-specific parameters based on loss type
        if loss_type in ["focal", "adaptive_focal", "hybrid"]:
            if "alpha" in params:
                trial_cfg.loss.alpha = float(params["alpha"])
            if "gamma" in params:
                trial_cfg.loss.gamma = float(params["gamma"])

        if loss_type == "adaptive_focal" and "delta" in params:
            trial_cfg.loss.delta = float(params["delta"])

        if loss_type == "hybrid" and "bce_weight" in params:
            trial_cfg.loss.bce_weight = float(params["bce_weight"])

        if loss_type == "weighted_bce" and "pos_weight" in params:
            trial_cfg.loss.pos_weight = float(params["pos_weight"])

    OmegaConf.set_struct(trial_cfg, True)
    return trial_cfg


def run_training_with_trial(cfg: DictConfig, trial: optuna.Trial) -> float:
    """Wrapper for run_training that handles trial injection and parameter suggestion."""
    # Suggest hyperparameters
    params = suggest_hyperparameters(trial, cfg.search_space)

    # Apply hyperparameters to config
    trial_cfg = apply_hyperparameters(cfg, params)

    # Inject trial object for pruning
    object.__setattr__(trial_cfg, '_trial_obj', trial)

    # Store the original get method
    original_get = trial_cfg.get

    # Create a custom get method that handles trial
    def custom_get(key, default_value=None):
        if key == 'trial':
            return getattr(trial_cfg, '_trial_obj', None)
        return original_get(key, default_value)

    # Monkey-patch the get method
    object.__setattr__(trial_cfg, 'get', custom_get)

    try:
        # Store trial output directory for artifact collection
        result = run_training(trial_cfg)
        if hasattr(trial_cfg, 'output_dir'):
            trial.set_user_attr('output_dir', str(trial_cfg.output_dir))
        return result
    finally:
        # Cleanup
        object.__setattr__(trial_cfg, 'get', original_get)
        if hasattr(trial_cfg, '_trial_obj'):
            object.__delattr__(trial_cfg, '_trial_obj')


@hydra.main(version_base=None, config_path='configs/training', config_name='maxed_hpo')
def main(cfg: DictConfig) -> None:
    """Main HPO entry point with enhanced configuration."""

    if not cfg.optuna.enabled:
        print("❌ Optuna optimization is not enabled in the config.")
        return

    # Create optimization output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    study_name = cfg.optuna.study_name
    optimization_dir = Path(f"outputs/optimization/{timestamp}_{study_name}")
    optimization_dir.mkdir(parents=True, exist_ok=True)

    # Configure pruner
    pruner = None
    if cfg.optuna.get('pruning', {}).get('enabled', False):
        pruning_cfg = cfg.optuna.pruning
        pruner_type = pruning_cfg.get('pruner', 'MedianPruner')

        if pruner_type == 'MedianPruner':
            pruner = optuna.pruners.MedianPruner(
                n_startup_trials=pruning_cfg.get('n_startup_trials', 10),
                n_warmup_steps=pruning_cfg.get('n_warmup_steps', 5),
                interval_steps=pruning_cfg.get('interval_steps', 1)
            )
        elif pruner_type == 'SuccessiveHalvingPruner':
            pruner = optuna.pruners.SuccessiveHalvingPruner()
        elif pruner_type == 'HyperbandPruner':
            pruner = optuna.pruners.HyperbandPruner()

        print(f"🔧 Using pruner: {pruner_type}")

    # Create study
    study = optuna.create_study(
        study_name=study_name,
        direction=cfg.optuna.direction,
        storage=cfg.optuna.get('storage'),
        load_if_exists=cfg.optuna.get('load_if_exists', True),
        pruner=pruner,
    )

    # Set up objective function
    def objective(trial: optuna.Trial) -> float:
        try:
            return run_training_with_trial(cfg, trial)
        except Exception as e:
            print(f"Trial {trial.number} failed: {e}")
            import traceback
            traceback.print_exc()
            raise optuna.exceptions.TrialPruned()

    # Print optimization info
    print(f"🚀 Starting maxed out Optuna optimization...")
    print(f"📁 Output directory: {optimization_dir}")
    print(f"🎯 Direction: {cfg.optuna.direction}")
    print(f"🔄 Number of trials: {cfg.optuna.n_trials}")
    print(f"⏱️  Timeout: {cfg.optuna.get('timeout', 'None')} seconds")
    print(f"💾 Storage: {cfg.optuna.get('storage', 'In-memory')}")

    search_space_summary = {}
    for param, config in cfg.search_space.items():
        if config.method == "categorical":
            search_space_summary[param] = f"{len(config.choices)} choices"
        else:
            search_space_summary[param] = f"{config.method}({config.get('low', '')}, {config.get('high', '')})"

    print(f"🔍 Search space: {len(cfg.search_space)} parameters")
    for param, summary in search_space_summary.items():
        print(f"   - {param}: {summary}")

    # Run optimization
    study.optimize(
        objective,
        n_trials=cfg.optuna.n_trials,
        timeout=cfg.optuna.get('timeout'),
        n_jobs=1,  # Single GPU training
    )

    # Save results
    from optuna_optimize import save_best_configuration, copy_best_trial_artifacts
    save_best_configuration(study, cfg, optimization_dir)
    copy_best_trial_artifacts(study, optimization_dir)

    print(f"\n✅ Maxed out optimization completed!")
    print(f"🏆 Best performance: {study.best_value:.4f}")
    print(f"📊 Total trials: {len(study.trials)}")
    print(f"🎯 Best trial: {study.best_trial.number}")


if __name__ == '__main__':
    main()