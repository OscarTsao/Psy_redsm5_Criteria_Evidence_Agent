from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import make_pairwise_datasets


def autocast_context(use_amp: bool, dtype: torch.dtype):
    if not use_amp:
        return torch.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu", dtype=dtype, enabled=False)
    return torch.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu", dtype=dtype)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except Exception:
        auc = None
    acc = accuracy_score(y_true, y_pred)
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auc": None if auc is None else float(auc),
        "accuracy": float(acc),
    }


def evaluate_model(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    threshold: float,
    use_amp: bool,
    amp_dtype: torch.dtype,
    capture_raw: bool = False,
) -> Tuple[Dict[str, float], list[Dict[str, float]], Dict[str, np.ndarray] | None]:
    model.eval()
    all_y, all_pred, all_prob, all_cidx, all_post_ids = [], [], [], [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating", leave=False):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with autocast_context(use_amp, amp_dtype):
                logits = model(input_ids, attention_mask)

            probs = torch.sigmoid(logits)
            preds = (probs > threshold).float()

            # ensure CPU float32 tensors before converting to numpy
            probs_np = probs.detach().float().cpu().numpy()
            preds_np = preds.detach().float().cpu().numpy()
            labels_np = labels.detach().float().cpu().numpy()
            cidx_np = batch["criterion_idx"].detach().cpu().numpy()

            all_y.append(labels_np)
            all_pred.append(preds_np)
            all_prob.append(probs_np)
            all_cidx.append(cidx_np)
            if capture_raw:
                all_post_ids.extend(batch["post_id"])

    y = np.concatenate(all_y) if all_y else np.array([])
    y_hat = np.concatenate(all_pred) if all_pred else np.array([])
    y_prob = np.concatenate(all_prob) if all_prob else np.array([])
    cidx = np.concatenate(all_cidx) if all_cidx else np.array([])

    overall = compute_metrics(y, y_hat, y_prob)

    per_criteria = []
    num_criteria = int(cidx.max() + 1) if cidx.size else 0
    for idx in range(num_criteria):
        subset = cidx == idx
        if subset.any():
            metrics = compute_metrics(y[subset], y_hat[subset], y_prob[subset])
        else:
            metrics = {k: float("nan") for k in ["precision", "recall", "f1", "auc", "accuracy"]}
        metrics["criteria_idx"] = idx
        per_criteria.append(metrics)

    raw_payload = None
    if capture_raw:
        raw_payload = {
            "probabilities": y_prob,
            "predictions": y_hat,
            "labels": y,
            "criterion_idx": cidx,
            "post_ids": all_post_ids,
        }

    return overall, per_criteria, raw_payload


def build_loader(dataset, loader_cfg: DictConfig) -> DataLoader:
    num_workers = loader_cfg.get("num_workers", None)
    if num_workers is None:
        num_workers = os.cpu_count() or 0
    shuffle = loader_cfg.get("shuffle", False)
    drop_last = loader_cfg.get("drop_last", False)
    pin_memory = loader_cfg.get("pin_memory", True)
    persistent_workers = loader_cfg.get("persistent_workers", False) and num_workers > 0
    prefetch_factor = loader_cfg.get("prefetch_factor", None)

    loader_kwargs = {
        "batch_size": loader_cfg.get("batch_size", 64),
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": persistent_workers,
        "drop_last": drop_last,
    }

    if prefetch_factor is not None and num_workers > 0:
        loader_kwargs["prefetch_factor"] = prefetch_factor

    return DataLoader(dataset, **loader_kwargs)


def apply_hardware_settings(hardware_cfg: DictConfig | Dict[str, bool]) -> None:
    if not torch.cuda.is_available():
        return

    enable_tf32 = hardware_cfg.get("enable_tf32", False)
    cudnn_benchmark = hardware_cfg.get("cudnn_benchmark", False)

    try:
        torch.backends.cuda.matmul.allow_tf32 = enable_tf32
        torch.backends.cudnn.allow_tf32 = enable_tf32
    except AttributeError:
        pass

    torch.backends.cudnn.benchmark = cudnn_benchmark


def load_training_config(run_dir: Path, checkpoint: Dict, checkpoint_path: Path) -> Tuple[DictConfig, str]:
    hydra_config_path = run_dir / ".hydra" / "config.yaml"
    direct_config_path = run_dir / "config.yaml"

    if hydra_config_path.exists():
        cfg = OmegaConf.load(hydra_config_path)
        if "training" in cfg:
            training_cfg = cfg.training
        else:
            training_cfg = cfg
        source = str(hydra_config_path)
        seed_value = cfg.get("seed", training_cfg.get("seed", 42))
    elif direct_config_path.exists():
        cfg = OmegaConf.load(direct_config_path)
        training_cfg = cfg  # The config file contains all sections
        source = str(direct_config_path)
        seed_value = cfg.get("seed", 42)
    else:
        config_data = checkpoint.get("config")
        if config_data is None:
            raise ValueError("Checkpoint does not contain a saved configuration under key 'config'.")
        training_cfg = OmegaConf.create(config_data)
        source = f"{checkpoint_path}::config"
        seed_value = training_cfg.get("seed", 42)

    OmegaConf.set_struct(training_cfg, False)
    training_cfg["seed"] = seed_value
    OmegaConf.set_struct(training_cfg, True)
    return training_cfg, source


def resolve_model(training_cfg: DictConfig) -> Tuple[torch.nn.Module, torch.device]:
    instantiated = instantiate(training_cfg.model)
    if isinstance(instantiated, tuple) and len(instantiated) == 2:
        model, device = instantiated
    else:
        model = instantiated
        target_device = training_cfg.model.get("device")
        if target_device is None:
            target_device = "cuda" if torch.cuda.is_available() else "cpu"
        device = torch.device(target_device)
        model = model.to(device)
    return model, device


def build_results_dataframe(raw_payload: Dict[str, np.ndarray], criteria_map: Dict[str, str]) -> pd.DataFrame:
    criteria_ids = sorted(criteria_map.keys(), key=lambda x: int(x.split(".")[1]))
    index_to_id = {idx: cid for idx, cid in enumerate(criteria_ids)}
    criterion_labels = [index_to_id.get(int(idx), f"unknown_{idx}") for idx in raw_payload["criterion_idx"]]

    return pd.DataFrame({
        "post_id": raw_payload["post_ids"],
        "probability": raw_payload["probabilities"].astype(float),
        "prediction": raw_payload["predictions"].astype(int),
        "true_label": raw_payload["labels"].astype(int),
        "criterion_idx": raw_payload["criterion_idx"].astype(int),
        "criterion_id": criterion_labels,
    })


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a stored run on the selected split using saved config and checkpoint.")
    parser.add_argument("--run", type=str, required=True, help="Run directory name or path under the training outputs root.")
    parser.add_argument("--training_root", type=str, default="outputs/training", help="Root directory containing training run folders.")
    parser.add_argument("--checkpoint", type=str, default="best_model.pt", help="Checkpoint filename or path to load (defaults to best_model.pt inside the run folder).")
    parser.add_argument("--evaluation_root", type=str, default="outputs/predictions", help="Directory where evaluation artefacts will be stored.")
    parser.add_argument("--name", type=str, default=None, help="Optional name suffix for the evaluation output folder.")
    parser.add_argument("--split", choices=["test", "val", "train"], default="test", help="Dataset split to evaluate.")
    parser.add_argument("--output_json", type=str, default=None, help="Optional override path for metrics JSON (defaults inside evaluation folder).")
    parser.add_argument("--raw_pairs_path", type=str, default=None, help="Optional override path for raw predictions CSV.")
    parser.add_argument("--no-save-raw-pairs", dest="save_raw_pairs", action="store_false", help="Disable saving raw pairwise predictions.")
    parser.set_defaults(save_raw_pairs=True)
    args = parser.parse_args()

    run_input = Path(args.run)
    training_root = Path(args.training_root)
    candidate_paths = []
    if run_input.exists():
        run_dir = run_input
    else:
        candidate_paths.extend([
            run_input,
            training_root / run_input.name,
            training_root / args.run,
        ])
        run_dir = None
        for candidate in candidate_paths:
            if candidate.exists():
                run_dir = candidate
                break
    if run_dir is None or not run_dir.exists():
        checked = ", ".join(str(p.resolve()) for p in candidate_paths if p)
        raise FileNotFoundError(f"Could not locate training run directory using '{args.run}'. Checked: {checked}")

    if args.checkpoint:
        checkpoint_candidate = Path(args.checkpoint)
        if checkpoint_candidate.is_file():
            checkpoint_path = checkpoint_candidate
        else:
            checkpoint_path = run_dir / args.checkpoint
    else:
        checkpoint_path = run_dir / "best_model.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    evaluation_root = Path(args.evaluation_root)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    eval_name = args.name or f"{run_dir.name}_{args.split}"
    evaluation_dir = evaluation_root / f"{timestamp}_{eval_name}"
    evaluation_dir.mkdir(parents=True, exist_ok=False)

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    training_cfg, config_source = load_training_config(run_dir, checkpoint, checkpoint_path)

    seed = training_cfg.get("seed", 42)
    posts_path = training_cfg.posts_path
    annotations_path = training_cfg.annotations_path
    criteria_path = training_cfg.criteria_path

    tokenizer_name = training_cfg.model.model_name

    # Rebuild splits with the same dataset paths the run was trained on
    split_datasets = make_pairwise_datasets(
        groundtruth_path=training_cfg.groundtruth_path,
        criteria_path=criteria_path,
        tokenizer_name=tokenizer_name,
        seed=seed,
    )
    train_ds, val_ds, test_ds, criteria_map = split_datasets

    split_to_dataset = {
        "train": (train_ds, training_cfg.train_loader),
        "val": (val_ds, training_cfg.val_loader),
        "test": (test_ds, training_cfg.test_loader),
    }

    dataset, loader_cfg = split_to_dataset[args.split]

    model, device = resolve_model(training_cfg)
    
    # Handle torch.compile state dict with _orig_mod prefixes
    state_dict = checkpoint["model_state_dict"]
    if any(key.startswith("_orig_mod.") for key in state_dict.keys()):
        # Remove _orig_mod prefixes from state dict keys
        cleaned_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith("_orig_mod."):
                cleaned_key = key[10:]  # Remove "_orig_mod." prefix
                cleaned_state_dict[cleaned_key] = value
            else:
                cleaned_state_dict[key] = value
        state_dict = cleaned_state_dict
    
    model.load_state_dict(state_dict)

    hardware_cfg = training_cfg.get("hardware", {})
    apply_hardware_settings(hardware_cfg)

    loader = build_loader(dataset, loader_cfg)

    training_settings = training_cfg.training
    amp_dtype_cfg = training_settings.amp_dtype
    if isinstance(amp_dtype_cfg, str):
        amp_dtype = getattr(torch, amp_dtype_cfg)
    else:
        amp_dtype = amp_dtype_cfg

    if hardware_cfg.get("use_bfloat16_if_available", False) and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        amp_dtype = torch.bfloat16

    capture_raw = args.save_raw_pairs
    overall_metrics, per_criteria_metrics, raw_payload = evaluate_model(
        model,
        loader,
        device,
        threshold=training_settings.threshold,
        use_amp=training_settings.use_amp,
        amp_dtype=amp_dtype,
        capture_raw=capture_raw,
    )

    results = {
        "config_source": config_source,
        "training_run": str(run_dir.resolve()),
        "checkpoint": str(checkpoint_path),
        "split": args.split,
        "overall": overall_metrics,
        "per_criteria": per_criteria_metrics,
    }

    output_json_path = Path(args.output_json) if args.output_json else evaluation_dir / "metrics.json"
    with open(output_json_path, "w") as f:
        json.dump(results, f, indent=2)

    if args.output_json is None:
        print(f"Evaluation artefacts stored in {evaluation_dir}")
    print(f"Loaded configuration from {config_source}")
    print(f"Evaluated checkpoint: {checkpoint_path}")
    print(f"Split: {args.split}")
    print("Overall metrics:")
    for key, value in overall_metrics.items():
        print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")
    print(f"Metrics written to {output_json_path}")

    if capture_raw and raw_payload is not None:
        raw_pairs_path = Path(args.raw_pairs_path) if args.raw_pairs_path else evaluation_dir / "raw_pairs.csv"
        raw_df = build_results_dataframe(raw_payload, criteria_map)
        raw_df.to_csv(raw_pairs_path, index=False)
        print(f"Raw pairwise predictions saved to {raw_pairs_path}")

    config_copy_path = evaluation_dir / "training_config.yaml"
    OmegaConf.save(training_cfg, config_copy_path)


if __name__ == "__main__":
    main()
