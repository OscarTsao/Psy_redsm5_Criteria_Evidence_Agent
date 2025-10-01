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
import pandas as pd
import yaml
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from train import run_training


def save_best_configuration(study: optuna.Study, base_cfg: DictConfig, output_dir: Path) -> None:
    """Save the best trial configuration for production use."""
    if len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]) == 0:
        print("⚠️  No completed trials found. Cannot save best configuration.")
        return

    best_trial = study.best_trial
    best_params = best_trial.params

    # Apply the best parameters to the base config
    best_cfg = apply_hyperparameters(base_cfg, best_params)

    # Convert to regular dict for saving
    best_config_dict = OmegaConf.to_container(best_cfg, resolve=True)

    # Save the best configuration
    best_config_path = output_dir / "best_config.yaml"
    with open(best_config_path, 'w') as f:
        yaml.dump(best_config_dict, f, default_flow_style=False, sort_keys=False)

    base_config_path = output_dir / "base_config.yaml"
    with open(base_config_path, 'w') as f:
        yaml.dump(OmegaConf.to_container(base_cfg, resolve=True), f, default_flow_style=False, sort_keys=False)

    # Create a production-ready config
    production_config = best_config_dict.copy()
    # Remove optuna-specific settings for production
    if 'optuna' in production_config:
        del production_config['optuna']
    if 'search_space' in production_config:
        del production_config['search_space']

    production_config_path = output_dir / "production_config.yaml"
    with open(production_config_path, 'w') as f:
        yaml.dump(production_config, f, default_flow_style=False, sort_keys=False)

    # Save optimization results
    optimization_results = {
        'best_trial': {
            'number': best_trial.number,
            'value': best_trial.value,
            'params': best_trial.params,
            'user_attrs': best_trial.user_attrs,
        },
        'study_summary': {
            'n_trials': len(study.trials),
            'completed_trials': len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
            'pruned_trials': len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]),
            'failed_trials': len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]),
            'direction': study.direction.name,
        }
    }

    results_path = output_dir / "optimization_results.json"
    with open(results_path, 'w') as f:
        json.dump(optimization_results, f, indent=2)

    print(f"💾 Best configuration saved to: {best_config_path}")
    print(f"🚀 Production config saved to: {production_config_path}")
    print(f"📊 Optimization results saved to: {results_path}")


def copy_best_trial_artifacts(study: optuna.Study, output_dir: Path) -> None:
    """Copy artifacts from the best trial to the optimization output directory."""
    if len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]) == 0:
        print("⚠️  No completed trials found. Cannot copy artifacts.")
        return

    best_trial = study.best_trial
    best_output_dir = best_trial.user_attrs.get('output_dir')

    if not best_output_dir or not Path(best_output_dir).exists():
        print(f"⚠️  Best trial output directory not found: {best_output_dir}")
        return

    best_artifacts_dir = output_dir / "best_trial_artifacts"
    best_artifacts_dir.mkdir(exist_ok=True)

    source_dir = Path(best_output_dir)

    # Copy important files
    files_to_copy = [
        "best_model.pt",
        "history.json",
        "test_metrics.json",
        "config.yaml"
    ]

    for filename in files_to_copy:
        source_file = source_dir / filename
        if source_file.exists():
            target_file = best_artifacts_dir / filename
            shutil.copy2(source_file, target_file)
            print(f"📁 Copied {filename} to artifacts directory")

    print(f"✅ Best trial artifacts copied to: {best_artifacts_dir}")


def export_trials_dataframe(study: optuna.Study, output_dir: Path) -> None:
    """Persist the full trials DataFrame for downstream analysis."""
    try:
        df = study.trials_dataframe(attrs=("number", "value", "state", "params", "user_attrs"))
    except Exception as exc:
        print(f"⚠️  Could not export trials dataframe: {exc}")
        return

    if df.empty:
        print("⚠️  Trials dataframe is empty; skipping export.")
        return

    csv_path = output_dir / "all_trials.csv"
    df.to_csv(csv_path, index=False)
    print(f"📈 Trials dataframe saved to: {csv_path}")


def cleanup_trial_output(path_like: Optional[str], root: Optional[Path], reason: str, *, force: bool = False) -> None:
    """Remove a trial output directory if it exists and lies within the allowed root."""
    if path_like is None:
        return

    candidate = Path(path_like)
    if not candidate.exists():
        return

    if root is not None:
        try:
            candidate.resolve().relative_to(root.resolve())
        except (ValueError, FileNotFoundError):
            print(f"⚠️  Skipping cleanup outside artifact root: {candidate}")
            return

    try:
        shutil.rmtree(candidate)
        icon = "🧹" if not force else "🧽"
        print(f"{icon} Removed trial artifacts at {candidate} ({reason})")
    except Exception as exc:
        print(f"⚠️  Could not remove trial artifacts at {candidate}: {exc}")


def suggest_hyperparameters(trial: optuna.Trial, search_space: DictConfig) -> Dict[str, Any]:
    """Suggest hyperparameters based on the search space configuration."""
    params: Dict[str, Any] = {}

    for param_name, param_config in search_space.items():
        method = param_config.method

        if method == "categorical":
            params[param_name] = trial.suggest_categorical(param_name, param_config.choices)
        elif method == "uniform":
            params[param_name] = trial.suggest_float(param_name, float(param_config.low), float(param_config.high))
        elif method == "loguniform":
            params[param_name] = trial.suggest_float(
                param_name,
                float(param_config.low),
                float(param_config.high),
                log=True,
            )
        elif method == "int":
            params[param_name] = trial.suggest_int(param_name, int(param_config.low), int(param_config.high))
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
    optimizer_type = params.get("optimizer_type")
    beta1 = params.get("beta1")
    beta2 = params.get("beta2")
    eps = params.get("eps")

    if optimizer_type:
        if optimizer_type == "adamw":
            trial_cfg.optimizer._target_ = "torch.optim.AdamW"
        elif optimizer_type == "adam":
            trial_cfg.optimizer._target_ = "torch.optim.Adam"
        elif optimizer_type == "rmsprop":
            trial_cfg.optimizer._target_ = "torch.optim.RMSprop"

    if optimizer_type in {"adamw", "adam"}:
        if beta1 is not None and beta2 is not None:
            trial_cfg.optimizer.betas = (float(beta1), float(beta2))
        elif "betas" in trial_cfg.optimizer:
            del trial_cfg.optimizer["betas"]
    elif "betas" in trial_cfg.optimizer:
        del trial_cfg.optimizer["betas"]

    if eps is not None:
        trial_cfg.optimizer.eps = float(eps)
    elif "eps" in trial_cfg.optimizer:
        del trial_cfg.optimizer["eps"]

    # Handle scheduler configuration
    if "scheduler_type" in params:
        scheduler_type = params["scheduler_type"]
        base_scheduler_cfg = trial_cfg.scheduler if trial_cfg.scheduler is not None else OmegaConf.create({})

        if scheduler_type == "plateau":
            patience = int(params.get("scheduler_patience", base_scheduler_cfg.get("patience", 5)))
            factor = float(params.get("scheduler_factor", base_scheduler_cfg.get("factor", 0.5)))
            min_lr = float(base_scheduler_cfg.get("min_lr", 1e-7))
            trial_cfg.scheduler = OmegaConf.create({
                "_target_": "torch.optim.lr_scheduler.ReduceLROnPlateau",
                "mode": "max",
                "factor": factor,
                "patience": patience,
                "min_lr": min_lr,
            })
        elif scheduler_type == "cosine":
            eta_min = float(base_scheduler_cfg.get("min_lr", 0.0))
            trial_cfg.scheduler = OmegaConf.create({
                "_target_": "torch.optim.lr_scheduler.CosineAnnealingLR",
                "T_max": max(1, int(trial_cfg.training.num_epochs)),
                "eta_min": eta_min,
            })
        elif scheduler_type == "linear":
            warmup_steps = max(1, int(params.get("warmup_steps", 1)))
            trial_cfg.scheduler = OmegaConf.create({
                "_target_": "torch.optim.lr_scheduler.LinearLR",
                "start_factor": 0.1,
                "end_factor": 1.0,
                "total_iters": warmup_steps,
            })
        elif scheduler_type == "exponential":
            gamma = float(params.get("scheduler_factor", base_scheduler_cfg.get("gamma", 0.9)))
            trial_cfg.scheduler = OmegaConf.create({
                "_target_": "torch.optim.lr_scheduler.ExponentialLR",
                "gamma": gamma,
            })

    # Handle loss function configuration
    if "loss_function" in params:
        loss_type = params["loss_function"]
        trial_cfg.loss._target_ = "model.DynamicLossFactory.create_loss"
        trial_cfg.loss.loss_type = loss_type

        # Apply loss-specific parameters based on loss type
        if loss_type in ["focal", "adaptive_focal", "hybrid_bce_focal", "hybrid_bce_adaptive_focal"]:
            if "alpha" in params:
                trial_cfg.loss.alpha = float(params["alpha"])
            if "gamma" in params:
                trial_cfg.loss.gamma = float(params["gamma"])

        if loss_type == "adaptive_focal" and "delta" in params:
            trial_cfg.loss.delta = float(params["delta"])

        if loss_type in ["hybrid_bce_focal", "hybrid_bce_adaptive_focal"] and "bce_weight" in params:
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

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    requested_study_name = cfg.optuna.study_name

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

    storage = cfg.optuna.get('storage')
    load_if_exists = cfg.optuna.get('load_if_exists', True)

    try:
        study = optuna.create_study(
            study_name=requested_study_name,
            direction=cfg.optuna.direction,
            storage=storage,
            load_if_exists=load_if_exists,
            pruner=pruner,
        )
    except optuna.exceptions.DuplicatedStudyError:
        if load_if_exists and storage:
            print(f"♻️  Study '{requested_study_name}' exists; loading existing study.")
            study = optuna.load_study(study_name=requested_study_name, storage=storage)
        else:
            fallback_name = f"{requested_study_name}_{timestamp}"
            print(f"⚠️  Study '{requested_study_name}' already exists. Using new study name '{fallback_name}'.")
            study = optuna.create_study(
                study_name=fallback_name,
                direction=cfg.optuna.direction,
                storage=storage,
                load_if_exists=False,
                pruner=pruner,
            )

    study_name = study.study_name
    optimization_dir = Path(f"outputs/optimization/{timestamp}_{study_name}")
    optimization_dir.mkdir(parents=True, exist_ok=True)

    cleanup_trial_dirs = cfg.optuna.get('cleanup_trial_dirs', False)
    keep_best_trial_dir = cfg.optuna.get('keep_best_trial_dir', True)
    remove_best_after_export = cfg.optuna.get('remove_best_trial_dir_after_export', False)

    artifact_root_value = cfg.optuna.get('artifact_root')
    default_artifact_root = cfg.get('output_dir', 'outputs')
    artifact_root: Optional[Path] = None
    base_candidate = artifact_root_value or default_artifact_root
    if base_candidate:
        root_path = Path(base_candidate)
        if not root_path.is_absolute():
            root_path = Path.cwd() / root_path
        artifact_root = root_path.resolve()

    best_trial_value = float('-inf')
    best_trial_dir: Optional[Path] = None
    best_trial_number: Optional[int] = None

    # Set up objective function
    def objective(trial: optuna.Trial) -> float:
        nonlocal best_trial_value, best_trial_dir, best_trial_number

        try:
            value = run_training_with_trial(cfg, trial)
        except optuna.exceptions.TrialPruned:
            if cleanup_trial_dirs:
                cleanup_trial_output(trial.user_attrs.get('output_dir'), artifact_root, f"pruned trial {trial.number}")
            raise
        except Exception as e:
            print(f"❌ Trial {trial.number} failed with error: {e}")
            import traceback
            traceback.print_exc()
            if cleanup_trial_dirs:
                cleanup_trial_output(trial.user_attrs.get('output_dir'), artifact_root, f"failed trial {trial.number}")
            # Return a very low score instead of pruning to avoid corrupting the study
            return -1.0
        else:
            output_dir = trial.user_attrs.get('output_dir')

            if value > best_trial_value:
                previous_best_dir = best_trial_dir
                best_trial_value = value
                best_trial_dir = Path(output_dir) if output_dir else None
                best_trial_number = trial.number

                if cleanup_trial_dirs and previous_best_dir and best_trial_dir and previous_best_dir != best_trial_dir and not keep_best_trial_dir:
                    cleanup_trial_output(str(previous_best_dir), artifact_root, f"superseded by trial {trial.number}")
            elif cleanup_trial_dirs:
                cleanup_trial_output(output_dir, artifact_root, f"non-best trial {trial.number}")

            return value

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
    try:
        save_best_configuration(study, cfg, optimization_dir)
        copy_best_trial_artifacts(study, optimization_dir)
        export_trials_dataframe(study, optimization_dir)
        if cleanup_trial_dirs and remove_best_after_export and best_trial_dir is not None:
            cleanup_trial_output(str(best_trial_dir), artifact_root, f"post-export cleanup for trial {best_trial_number}", force=True)
    except Exception as e:
        print(f"⚠️  Warning: Could not save optimization results: {e}")
        # Save basic study information as fallback
        study_info = {
            'study_name': study.study_name,
            'direction': study.direction.name,
            'n_trials': len(study.trials),
            'completed_trials': len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
            'pruned_trials': len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]),
            'failed_trials': len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]),
        }
        if len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]) > 0:
            study_info['best_value'] = study.best_value
            study_info['best_params'] = study.best_params

        with open(optimization_dir / "study_summary.json", "w") as f:
            json.dump(study_info, f, indent=2)

    print(f"\n✅ Maxed out optimization completed!")
    print(f"📊 Total trials: {len(study.trials)}")

    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if len(completed_trials) > 0:
        print(f"🏆 Best performance: {study.best_value:.4f}")
        print(f"🎯 Best trial: {study.best_trial.number}")
    else:
        print("⚠️  No trials completed successfully.")

    print(f"✅ Completed trials: {len(completed_trials)}")
    print(f"✂️  Pruned trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])}")
    print(f"❌ Failed trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL])}")


if __name__ == '__main__':
    main()
