#!/usr/bin/env python3
"""
Script to run a series of training experiments with different loss functions
using the same base configuration from a previous training run.
"""

import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import yaml

DEFAULT_GROUNDTRUTH_PATH = "Data/groundtruth/redsm5_ground_truth.json"
DEFAULT_CRITERIA_PATH = "Data/DSM-5/DSM_Criteria_Array_Fixed_Major_Depressive.json"

def load_test_metrics(metrics_path: Path) -> Optional[dict]:
    """Load the test metrics JSON if it exists."""
    if not metrics_path.exists():
        return None
    try:
        with open(metrics_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as exc:
        print(f"⚠️  Could not parse test metrics at {metrics_path}: {exc}")
    return None


def run_training_with_loss(base_config_path: str, loss_type: str, loss_params: Optional[dict] = None) -> dict:
    """Run training with specific loss function override."""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"outputs/training/loss_sweep_{loss_type}_{timestamp}")

    cmd = ["python", "train.py", f"--config-path={base_config_path}"]

    # Hydra uses the file stem as the config name (e.g., config.yaml -> config)
    config_file = Path(base_config_path) / "config.yaml"
    config_name = config_file.stem
    cmd.append(f"--config-name={config_name}")

    # Ensure the config is compatible with the current data pipeline
    with open(config_file, "r", encoding="utf-8") as f:
        loaded_cfg = yaml.safe_load(f)

    if "groundtruth_path" not in loaded_cfg:
        cmd.append(f"+groundtruth_path={DEFAULT_GROUNDTRUTH_PATH}")

    if "criteria_path" not in loaded_cfg:
        cmd.append(f"+criteria_path={DEFAULT_CRITERIA_PATH}")

    loss_cfg = loaded_cfg.get("loss", {}) or {}
    has_loss_type = isinstance(loss_cfg, dict) and "loss_type" in loss_cfg

    if not has_loss_type:
        cmd.append("loss._target_=model.DynamicLossFactory.create_loss")
        cmd.append(f"+loss.loss_type={loss_type}")
    else:
        cmd.append(f"loss.loss_type={loss_type}")

    # Add loss-specific parameters
    if loss_params:
        existing_loss_keys = set(loss_cfg.keys()) if isinstance(loss_cfg, dict) else set()
        for param, value in loss_params.items():
            prefix = "+loss" if param not in existing_loss_keys else "loss"
            cmd.append(f"{prefix}.{param}={value}")

    cmd.append(f"hydra.run.dir={output_dir}")
    cmd.append("hydra.job.chdir=false")  # Prevent Hydra from changing working directory

    print(f"\n{'='*60}")
    print(f"Starting training with loss function: {loss_type}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}\n")

    start_time = time.time()

    try:
        subprocess.run(cmd, check=True, capture_output=False)

        end_time = time.time()
        duration = end_time - start_time

        print(f"\n{'='*60}")
        print(f"✅ COMPLETED: {loss_type} training")
        print(f"Duration: {duration/60:.1f} minutes")
        print(f"{'='*60}\n")

        metrics = load_test_metrics(output_dir / "test_metrics.json")
        if metrics is None:
            print(f"⚠️  test_metrics.json not found in {output_dir}")

        return {
            "success": True,
            "output_dir": output_dir,
            "metrics": metrics,
        }

    except subprocess.CalledProcessError as e:
        print(f"\n{'='*60}")
        print(f"❌ FAILED: {loss_type} training")
        print(f"Error code: {e.returncode}")
        print(f"{'='*60}\n")

        return {
            "success": False,
            "output_dir": output_dir,
            "metrics": None,
        }

def main():
    """Main function to run loss function sweep."""

    # Base configuration path
    base_config_path = "outputs/training/20250930_222338"

    # Verify base config exists
    config_file = Path(base_config_path) / "config.yaml"
    if not config_file.exists():
        print(f"❌ Base config not found: {config_file}")
        sys.exit(1)

    # Define loss functions to test (excluding hybrid_bce_adaptive_focal which is already done)
    loss_functions = [
        {
            "name": "bce",
            "params": {}
        },
        {
            "name": "weighted_bce",
            "params": {
                "pos_weight": 4.380814513612265  # Same as original config
            }
        },
        {
            "name": "focal",
            "params": {
                "alpha": 0.278422037479522,  # Same as original config
                "gamma": 4.374808397481278   # Same as original config
            }
        },
        {
            "name": "adaptive_focal",
            "params": {
                "alpha": 0.278422037479522,  # Same as original config
                "gamma": 4.374808397481278,  # Same as original config
                "delta": 2.2488567857875084  # Same as original config
            }
        },
        {
            "name": "hybrid_bce_focal",
            "params": {
                "alpha": 0.278422037479522,      # Same as original config
                "gamma": 4.374808397481278,      # Same as original config
                "bce_weight": 0.20103709477170445, # Same as original config
                "pos_weight": 4.380814513612265   # Same as original config
            }
        }
    ]

    print("🚀 Starting loss function sweep")
    print(f"Base config: {base_config_path}")
    print(f"Testing {len(loss_functions)} loss functions")
    print(f"Loss functions: {', '.join([lf['name'] for lf in loss_functions])}")

    # Track results
    results = {}
    total_start_time = time.time()

    # Run training for each loss function
    for i, loss_config in enumerate(loss_functions, 1):
        loss_name = loss_config["name"]
        loss_params = loss_config["params"]

        print(f"\n🔄 Progress: {i}/{len(loss_functions)} - Testing {loss_name}")

        outcome = run_training_with_loss(
            base_config_path=base_config_path,
            loss_type=loss_name,
            loss_params=loss_params
        )

        results[loss_name] = outcome

        # Optional: Add delay between runs to avoid resource conflicts
        if i < len(loss_functions):
            print("⏳ Waiting 30 seconds before next run...")
            time.sleep(30)

    # Final summary
    total_end_time = time.time()
    total_duration = total_end_time - total_start_time

    print(f"\n{'='*80}")
    print("🎯 LOSS FUNCTION SWEEP COMPLETE")
    print(f"{'='*80}")
    print(f"Total duration: {total_duration/3600:.1f} hours")
    print(f"Base config: {base_config_path}")
    print("\nResults:")
    successes = 0
    for loss_name, info in results.items():
        status_icon = "✅" if info["success"] else "❌"
        summary = "N/A"
        metrics = info.get("metrics") or {}
        overall = metrics.get("overall") if isinstance(metrics, dict) else None
        if overall:
            summary = (
                f"f1={overall.get('f1', float('nan')):.4f} "
                f"precision={overall.get('precision', float('nan')):.4f} "
                f"recall={overall.get('recall', float('nan')):.4f}"
            )
        print(f"  {loss_name:20} : {status_icon} {info['output_dir']} {summary}")
        if info["success"]:
            successes += 1

    print(f"\nSummary: {successes}/{len(loss_functions)} training runs completed successfully")

    if successes == len(loss_functions):
        print("🎉 All training runs completed successfully!")
    else:
        print("⚠️  Some training runs failed. Check the logs above.")

    print(f"{'='*80}")

if __name__ == "__main__":
    main()
