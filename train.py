from __future__ import annotations

import json
import os
import glob
import multiprocessing as mp
from datetime import datetime
from pathlib import Path
from typing import Dict

import hydra
import numpy as np
import optuna
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.amp import GradScaler
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, accuracy_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import make_pairwise_datasets
from model import DynamicLossFactory, optimize_hardware_settings



def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except Exception:
        auc = None
    acc = accuracy_score(y_true, y_pred)
    return {
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'auc': None if auc is None else float(auc),
        'accuracy': float(acc),
    }


def evaluate(model, loader, device, threshold: float = 0.5, use_amp: bool = True):
    model.eval()
    all_y, all_pred, all_prob, all_cidx = [], [], [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc='Eval'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            cidx = batch['criterion_idx'].cpu().numpy()

            if use_amp:
                with autocast_context(use_amp, torch.float16):
                    logits = model(input_ids, attention_mask)
            else:
                logits = model(input_ids, attention_mask)

            probs = torch.sigmoid(logits)
            preds = (probs > threshold).float()

            all_y.append(labels.cpu().numpy())
            all_pred.append(preds.cpu().numpy())
            all_prob.append(probs.cpu().numpy())
            all_cidx.append(cidx)

    y = np.concatenate(all_y)
    y_hat = np.concatenate(all_pred)
    y_prob = np.concatenate(all_prob)
    cidx = np.concatenate(all_cidx)

    overall = compute_metrics(y, y_hat, y_prob)

    # Per-criterion metrics
    per_criteria = []
    for i in range(9):
        m = compute_metrics(y[cidx == i], y_hat[cidx == i], y_prob[cidx == i])
        m['criteria_idx'] = i
        per_criteria.append(m)

    return overall, per_criteria


def seed_everything(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_dataloaders(train_ds, val_ds, test_ds, cfg):
    def make_loader(ds, params, fallback_shuffle=False):
        batch_size = params.batch_size
        num_workers = params.num_workers
        if num_workers is None:
            num_workers = max(4, os.cpu_count() or 4)

        shuffle = params.get("shuffle", fallback_shuffle)
        drop_last = params.get("drop_last", False)
        prefetch_factor = params.get("prefetch_factor", None)
        persistent_workers = params.get("persistent_workers", False)

        if num_workers > 0:
            try:
                test_lock = mp.get_context().Lock()
                del test_lock
            except (RuntimeError, OSError, PermissionError):
                print("⚠️  System does not permit multiprocessing locks; falling back to num_workers=0.")
                num_workers = 0

        if num_workers == 0:
            prefetch_factor = None
            persistent_workers = False

        loader_kwargs = {
            "batch_size": batch_size,
            "shuffle": shuffle,
            "num_workers": num_workers,
            "pin_memory": params.get("pin_memory", True),
            "persistent_workers": persistent_workers and num_workers > 0,
            "drop_last": drop_last,
        }
        if prefetch_factor is not None:
            loader_kwargs["prefetch_factor"] = prefetch_factor

        try:
            return DataLoader(ds, **loader_kwargs)
        except (RuntimeError, OSError, PermissionError, ValueError) as exc:
            if num_workers > 0:
                print(f"⚠️  DataLoader worker startup failed ({exc}); retrying with num_workers=0.")
                loader_kwargs["num_workers"] = 0
                loader_kwargs["persistent_workers"] = False
                loader_kwargs.pop("prefetch_factor", None)
                return DataLoader(ds, **loader_kwargs)
            raise

    train_loader = make_loader(train_ds, cfg.train_loader, fallback_shuffle=True)
    val_loader = make_loader(val_ds, cfg.val_loader)
    test_loader = make_loader(test_ds, cfg.test_loader)
    return train_loader, val_loader, test_loader


def autocast_context(use_amp: bool, dtype: torch.dtype):
    if not use_amp:
        return torch.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu", dtype=dtype, enabled=False)
    return torch.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu", dtype=dtype)


def forward_loss(model, batch, criterion, device, use_amp=False, amp_dtype=torch.float16):
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    labels = batch['labels'].to(device)

    with autocast_context(use_amp, amp_dtype):
        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)
    return loss, logits


def train_one_epoch(
    model,
    train_loader,
    optimizer,
    criterion,
    device,
    gradient_accumulation_steps,
    clip_grad_norm,
    use_amp,
    scaler,
    amp_dtype,
    max_steps_per_epoch: int | None = None,
):
    model.train()
    total_loss = 0.0
    accumulated_loss = 0.0
    optimizer_steps = 0

    optimizer.zero_grad()
    for step, batch in enumerate(tqdm(train_loader, desc='Train', leave=False)):
        loss, _ = forward_loss(model, batch, criterion, device, use_amp, amp_dtype)
        loss = loss / gradient_accumulation_steps

        if scaler and use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        accumulated_loss += loss.item()

        if (step + 1) % gradient_accumulation_steps == 0:
            if scaler and use_amp:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
            if scaler and use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()

            total_loss += accumulated_loss
            accumulated_loss = 0.0
            optimizer_steps += 1

            if max_steps_per_epoch is not None and optimizer_steps >= max_steps_per_epoch:
                break

    if accumulated_loss > 0:
        if scaler and use_amp:
            scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
        if scaler and use_amp:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad()
        total_loss += accumulated_loss
        optimizer_steps += 1

    if optimizer_steps == 0:
        return 0.0

    return total_loss / optimizer_steps


def evaluate_model(model, loader, device, threshold: float = 0.5, use_amp: bool = True, amp_dtype=torch.float16):
    model.eval()
    all_y, all_pred, all_prob, all_cidx = [], [], [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc='Eval', leave=False):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            cidx = batch['criterion_idx'].cpu().numpy()

            with autocast_context(use_amp, amp_dtype):
                logits = model(input_ids, attention_mask)

            probs = torch.sigmoid(logits)
            preds = (probs > threshold).float()

            all_y.append(labels.cpu().numpy())
            all_pred.append(preds.cpu().numpy())
            all_prob.append(probs.cpu().numpy())
            all_cidx.append(cidx)

    y = np.concatenate(all_y)
    y_hat = np.concatenate(all_pred)
    y_prob = np.concatenate(all_prob)
    cidx = np.concatenate(all_cidx)

    overall = compute_metrics(y, y_hat, y_prob)

    per_criteria = []
    for i in range(9):
        subset = cidx == i
        if subset.any():
            m = compute_metrics(y[subset], y_hat[subset], y_prob[subset])
        else:
            m = {k: float('nan') for k in ['precision', 'recall', 'f1', 'auc', 'accuracy']}
        m['criteria_idx'] = i
        per_criteria.append(m)

    return overall, per_criteria


def cleanup_old_checkpoints(output_dir: Path, max_checkpoints: int = 5):
    """Keep only the most recent checkpoints, excluding best_model.pt"""
    checkpoint_pattern = str(output_dir / 'checkpoint_epoch_*.pt')
    checkpoints = glob.glob(checkpoint_pattern)

    if len(checkpoints) > max_checkpoints:
        # Sort by modification time (oldest first)
        checkpoints.sort(key=lambda x: os.path.getmtime(x))
        # Remove oldest checkpoints
        for old_checkpoint in checkpoints[:-max_checkpoints]:
            try:
                os.remove(old_checkpoint)
                print(f"Removed old checkpoint: {os.path.basename(old_checkpoint)}")
            except OSError as e:
                print(f"Warning: Could not remove {old_checkpoint}: {e}")


def save_history_and_checkpoint(
    output_dir: Path,
    model,
    optimizer,
    cfg,
    history,
    metrics,
    epoch,
    best: bool = False,
    max_checkpoints: int = 5,
    full_cfg=None,
):
    training_cfg = cfg.get('training', cfg)
    save_checkpoints = training_cfg.get('save_checkpoints', True)
    save_best_only = training_cfg.get('save_best_only', False)
    save_optimizer_state = training_cfg.get('save_optimizer_state', True)
    save_history_file = training_cfg.get('save_history', True)
    include_history_in_checkpoint = training_cfg.get('include_history_in_checkpoint', True)
    save_config_in_checkpoint = training_cfg.get('save_config_in_checkpoint', True)

    should_save_ckpt = save_checkpoints and (best or not save_best_only)

    if should_save_ckpt:
        ckpt = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'metrics': metrics,
        }

        if save_optimizer_state:
            ckpt['optimizer_state_dict'] = optimizer.state_dict()

        if save_config_in_checkpoint:
            # Save the full config if provided, otherwise fallback to cfg
            config_to_save = full_cfg if full_cfg is not None else cfg
            ckpt['config'] = OmegaConf.to_container(config_to_save, resolve=True)

        if include_history_in_checkpoint:
            ckpt['history'] = history

        ckpt_name = output_dir / ('best_model.pt' if best else f'checkpoint_epoch_{epoch}.pt')
        torch.save(ckpt, ckpt_name)

        if not best and max_checkpoints and max_checkpoints > 0:
            cleanup_old_checkpoints(output_dir, max_checkpoints)

    if save_history_file:
        history_path = output_dir / 'history.json'
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)


def extract_config(cfg: DictConfig) -> DictConfig:
    if 'posts_path' in cfg:
        return cfg
    if 'training' in cfg and isinstance(cfg.training, DictConfig):
        return cfg.training
    raise ValueError("Configuration missing required keys (posts_path, training, etc.)")


def run_training(cfg: DictConfig) -> float:
    # Apply hardware optimizations first
    from model import optimize_hardware_settings
    optimize_hardware_settings()

    # Store full config for checkpoint saving
    full_cfg = cfg

    # If cfg has 'training' key, we're receiving the full hydra config
    # Extract the training section which contains everything we need
    if 'training' in cfg:
        cfg = cfg.training

    trial = cfg.get("trial")
    if trial is not None:
        OmegaConf.set_struct(cfg, False)
        apply_trial_suggestions(cfg, trial)
        OmegaConf.set_struct(cfg, True)

    seed = cfg.get('seed', 42)
    seed_everything(seed)

    output_dir = ensure_output_dir(cfg)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Store output directory in trial for artifact copying
    if trial is not None:
        trial.set_user_attr('output_dir', str(output_dir))

    train_ds, val_ds, test_ds, _ = make_pairwise_datasets(
        cfg.posts_path,
        cfg.annotations_path,
        cfg.criteria_path,
        tokenizer_name=cfg.model.model_name,
        seed=seed,
    )

    model, device = instantiate(cfg.model)
    if cfg.training.get("use_grad_checkpointing", False) and hasattr(model.encoder, 'gradient_checkpointing_enable'):
        model.encoder.gradient_checkpointing_enable()

    if cfg.training.use_compile and hasattr(torch, 'compile'):
        model = torch.compile(model)

    # Enhanced loss function instantiation with dynamic factory support
    if hasattr(cfg.loss, 'loss_type'):
        from model import DynamicLossFactory
        loss_params = {k: v for k, v in cfg.loss.items() if k not in ['_target_', 'loss_type']}
        criterion = DynamicLossFactory.create_loss(cfg.loss.loss_type, **loss_params)
    else:
        criterion = instantiate(cfg.loss)

    optimizer = instantiate(cfg.optimizer, params=model.parameters())
    scheduler = instantiate(cfg.scheduler, optimizer=optimizer) if cfg.scheduler else None

    train_loader, val_loader, test_loader = build_dataloaders(train_ds, val_ds, test_ds, cfg)

    amp_device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
    use_amp = bool(cfg.training.use_amp and amp_device_type == 'cuda')

    scaler = GradScaler(amp_device_type, enabled=use_amp)

    best_metric = -float('inf')
    history = []
    patience_counter = 0
    early_stopping_patience = cfg.training.get('early_stopping_patience', 10)

    amp_dtype = getattr(torch, cfg.training.amp_dtype) if isinstance(cfg.training.amp_dtype, str) else cfg.training.amp_dtype
    if amp_device_type != 'cuda':
        amp_dtype = torch.float32

    for epoch in range(1, cfg.training.num_epochs + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            cfg.training.gradient_accumulation_steps,
            cfg.training.clip_grad_norm,
            use_amp,
            scaler,
            amp_dtype,
            cfg.training.get('max_steps_per_epoch'),
        )

        val_metrics, val_per = evaluate_model(
            model,
            val_loader,
            device,
            threshold=cfg.training.threshold,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
        )

        if scheduler:
            if hasattr(scheduler, 'step') and 'ReduceLROnPlateau' in str(type(scheduler)):
                scheduler.step(val_metrics.get(cfg.monitor_metric, 0.0))
            else:
                scheduler.step()

        history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_overall': val_metrics,
        })

        metric_value = val_metrics.get(cfg.monitor_metric, 0.0)
        is_best = metric_value > best_metric
        if is_best:
            best_metric = metric_value
            patience_counter = 0
        else:
            patience_counter += 1

        metrics_payload = {
            'train_loss': train_loss,
            'val_overall': val_metrics,
                'val_per_criteria': val_per,
            }
        max_checkpoints = cfg.training.get('max_checkpoints', 5)
        save_history_and_checkpoint(output_dir, model, optimizer, cfg, history, metrics_payload, epoch, best=is_best, max_checkpoints=max_checkpoints, full_cfg=full_cfg)

        status = f"Epoch {epoch}: train_loss={train_loss:.4f} val_{cfg.monitor_metric}={metric_value:.4f}"
        if is_best:
            status += " (new best)"
        else:
            status += f" (patience: {patience_counter}/{early_stopping_patience})"
        print(status)

        # Optuna pruning check
        trial = cfg.get('trial')
        if trial is not None:
            trial.report(metric_value, epoch)
            if trial.should_prune():
                print(f"Trial pruned at epoch {epoch}")
                raise optuna.exceptions.TrialPruned()

        # Early stopping check
        if patience_counter >= early_stopping_patience:
            print(f"Early stopping triggered after {epoch} epochs (patience: {early_stopping_patience})")
            break

    test_metrics, test_per = evaluate_model(
        model,
        test_loader,
        device,
        threshold=cfg.training.threshold,
        use_amp=use_amp,
        amp_dtype=amp_dtype,
    )
    with open(output_dir / 'test_metrics.json', 'w') as f:
        json.dump({'overall': test_metrics, 'per_criteria': test_per}, f, indent=2)

    return best_metric


def apply_trial_suggestions(cfg: DictConfig, trial: optuna.Trial) -> None:
    search_space = cfg.get("search_space")
    if not search_space:
        return

    for param_name, space_cfg in search_space.items():
        suggestion = sampler_dispatch(trial, param_name, space_cfg)

        if param_name == "train_batch_size":
            cfg.train_loader.batch_size = int(suggestion)
        elif param_name == "eval_batch_size":
            cfg.eval_batch_size = int(suggestion)
            cfg.val_loader.batch_size = int(suggestion)
            cfg.test_loader.batch_size = int(suggestion)
        elif param_name == "learning_rate":
            cfg.optimizer.lr = float(suggestion)
        elif param_name == "weight_decay":
            cfg.optimizer.weight_decay = float(suggestion)
        elif param_name == "dropout":
            cfg.model.dropout = float(suggestion)
        elif param_name == "clip_grad_norm":
            cfg.training.clip_grad_norm = float(suggestion)
        elif param_name == "threshold":
            cfg.training.threshold = float(suggestion)
        elif param_name == "gradient_accumulation_steps":
            cfg.training.gradient_accumulation_steps = int(suggestion)
        # Loss function type selection
        elif param_name == "loss_function":
            cfg.loss._target_ = f"model.DynamicLossFactory.create_loss"
            cfg.loss.loss_type = suggestion
        # Loss function parameters
        elif param_name in ["alpha", "gamma", "delta", "bce_weight", "pos_weight"]:
            if not hasattr(cfg.loss, param_name):
                setattr(cfg.loss, param_name, suggestion)
            else:
                cfg.loss[param_name] = suggestion
        else:
            # For unexpected parameters, try to set directly if they exist
            if hasattr(cfg, param_name):
                setattr(cfg, param_name, suggestion)

def sampler_dispatch(trial: optuna.Trial, name: str, space_cfg: DictConfig):
    method = space_cfg.method
    if method == 'uniform':
        return trial.suggest_float(name, space_cfg.low, space_cfg.high)
    if method == 'loguniform':
        return trial.suggest_float(name, space_cfg.low, space_cfg.high, log=True)
    if method == 'int':
        return trial.suggest_int(name, space_cfg.low, space_cfg.high)
    if method == 'categorical':
        return trial.suggest_categorical(name, space_cfg.choices)
    raise ValueError(f"Unsupported sampling method: {method}")


def ensure_output_dir(cfg: DictConfig) -> Path:
    if 'hydra' in cfg and 'run' in cfg.hydra and 'dir' in cfg.hydra.run:
        return Path(cfg.hydra.run.dir)
    base_dir = cfg.get('output_dir', 'outputs')
    run_dir = Path(base_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    return run_dir


@hydra.main(version_base=None, config_path='configs', config_name='config')
def main(cfg: DictConfig) -> None:
    print("Configuration:\n" + OmegaConf.to_yaml(cfg))
    # allow setting seed on structured config
    OmegaConf.set_struct(cfg.training, False)
    cfg.training.seed = cfg.get('seed', 42)
    OmegaConf.set_struct(cfg.training, True)
    # Pass full config to run_training so it can save complete config in checkpoint
    run_training(cfg)


if __name__ == '__main__':
    main()
