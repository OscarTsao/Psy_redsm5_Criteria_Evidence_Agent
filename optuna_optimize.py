#!/usr/bin/env python3
"""
Optuna hyperparameter optimization script with best configuration saving.
Integrates with the existing Hydra-based training system.
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


def create_study_and_optimize(cfg: DictConfig) -> tuple[optuna.Study, float]:
    """Create Optuna study and run optimization."""

    optuna_cfg = cfg.optuna
    study_name = optuna_cfg.get('study_name', f'hpo_{datetime.now().strftime("%Y%m%d_%H%M%S")}')

    # Create study
    study = optuna.create_study(
        study_name=study_name,
        direction=optuna_cfg.direction,
        storage=optuna_cfg.get('storage'),
        load_if_exists=optuna_cfg.get('load_if_exists', True),
    )

    # Set up objective function
    def objective(trial: optuna.Trial) -> float:
        # Create a copy of the config for this trial
        trial_cfg = OmegaConf.copy(cfg)
        trial_cfg.trial = trial

        try:
            return run_training(trial_cfg)
        except Exception as e:
            print(f"Trial {trial.number} failed: {e}")
            raise optuna.exceptions.TrialPruned()

    # Run optimization
    study.optimize(
        objective,
        n_trials=optuna_cfg.n_trials,
        timeout=optuna_cfg.get('timeout'),
        n_jobs=1,  # Single GPU training
    )

    return study, study.best_value


def save_best_configuration(study: optuna.Study, base_cfg: DictConfig, output_dir: Path) -> None:
    """Save the best trial configuration and results."""

    best_trial = study.best_trial
    best_params = best_trial.params

    # Create optimized config by applying best parameters
    optimized_cfg = OmegaConf.copy(base_cfg)
    OmegaConf.set_struct(optimized_cfg, False)

    # Apply best parameters
    for param_name, value in best_params.items():
        if param_name == "train_batch_size":
            optimized_cfg.train_loader.batch_size = int(value)
        elif param_name == "eval_batch_size":
            optimized_cfg.eval_batch_size = int(value)
            optimized_cfg.val_loader.batch_size = int(value)
            optimized_cfg.test_loader.batch_size = int(value)
        elif param_name == "learning_rate":
            optimized_cfg.optimizer.lr = float(value)
        elif param_name == "weight_decay":
            optimized_cfg.optimizer.weight_decay = float(value)
        elif param_name == "alpha":
            optimized_cfg.loss.alpha = float(value)
        elif param_name == "gamma":
            optimized_cfg.loss.gamma = float(value)
        elif param_name == "delta":
            optimized_cfg.loss.delta = float(value)
        elif param_name == "dropout":
            optimized_cfg.model.dropout = float(value)
        elif param_name == "clip_grad_norm":
            optimized_cfg.training.clip_grad_norm = float(value)
        elif param_name == "threshold":
            optimized_cfg.training.threshold = float(value)
        elif param_name == "gradient_accumulation_steps":
            optimized_cfg.training.gradient_accumulation_steps = int(value)

    # Remove optuna-specific configs
    if 'optuna' in optimized_cfg:
        del optimized_cfg.optuna
    if 'search_space' in optimized_cfg:
        del optimized_cfg.search_space
    if 'trial' in optimized_cfg:
        del optimized_cfg.trial

    OmegaConf.set_struct(optimized_cfg, True)

    # Save optimized configuration
    optimized_config_path = output_dir / "best_config.yaml"
    OmegaConf.save(optimized_cfg, optimized_config_path)

    # Save optimization results
    results = {
        "study_name": study.study_name,
        "best_trial_number": best_trial.number,
        "best_value": study.best_value,
        "best_params": best_params,
        "optimization_direction": study.direction.name,
        "n_trials": len(study.trials),
        "datetime": datetime.now().isoformat(),
        "optimization_history": [
            {
                "trial_number": trial.number,
                "value": trial.value,
                "params": trial.params,
                "state": trial.state.name,
            }
            for trial in study.trials
        ],
    }

    results_path = output_dir / "optimization_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    # Create a ready-to-use config file for production
    production_config_path = output_dir / "production_config.yaml"
    production_cfg = OmegaConf.copy(optimized_cfg)

    # Adjust for production settings
    production_cfg.training.num_epochs = 50  # Reasonable default
    production_cfg.training.early_stopping_patience = 7

    OmegaConf.save(production_cfg, production_config_path)

    print(f"\n🎯 Optimization Results:")
    print(f"📊 Best {study.direction.name} value: {study.best_value:.4f}")
    print(f"📝 Best parameters: {best_params}")
    print(f"💾 Configurations saved to:")
    print(f"   - Best config: {optimized_config_path}")
    print(f"   - Production config: {production_config_path}")
    print(f"   - Full results: {results_path}")


def copy_best_trial_artifacts(study: optuna.Study, optimization_dir: Path) -> Optional[Path]:
    """Copy artifacts from the best trial to the optimization output directory."""

    best_trial = study.best_trial
    if best_trial.user_attrs.get('output_dir'):
        best_trial_dir = Path(best_trial.user_attrs['output_dir'])
        if best_trial_dir.exists():
            best_artifacts_dir = optimization_dir / "best_trial_artifacts"
            shutil.copytree(best_trial_dir, best_artifacts_dir, dirs_exist_ok=True)
            print(f"📦 Best trial artifacts copied to: {best_artifacts_dir}")
            return best_artifacts_dir

    return None


@hydra.main(version_base=None, config_path='configs', config_name='config')
def main(cfg: DictConfig) -> None:
    """Main optimization entry point."""

    # Extract training config
    if 'training' in cfg:
        training_cfg = cfg.training
    else:
        training_cfg = cfg

    # Ensure optuna is enabled
    if not training_cfg.optuna.get('enabled', False):
        print("❌ Optuna optimization is not enabled in the config.")
        print("Set 'optuna.enabled: true' in your config file.")
        return

    # Create optimization output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    study_name = training_cfg.optuna.get('study_name', 'optimization')
    optimization_dir = Path(f"outputs/optimization/{timestamp}_{study_name}")
    optimization_dir.mkdir(parents=True, exist_ok=True)

    # Save base configuration
    base_config_path = optimization_dir / "base_config.yaml"
    OmegaConf.save(training_cfg, base_config_path)

    print(f"🚀 Starting Optuna optimization...")
    print(f"📁 Output directory: {optimization_dir}")
    print(f"🎯 Direction: {training_cfg.optuna.direction}")
    print(f"🔄 Number of trials: {training_cfg.optuna.n_trials}")

    # Run optimization
    study, best_value = create_study_and_optimize(training_cfg)

    # Save results
    save_best_configuration(study, training_cfg, optimization_dir)

    # Copy best trial artifacts if available
    copy_best_trial_artifacts(study, optimization_dir)

    print(f"\n✅ Optimization completed!")
    print(f"🏆 Best performance: {best_value:.4f}")


if __name__ == '__main__':
    main()