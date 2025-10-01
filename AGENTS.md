# Repository Guidelines

## Project Structure & Module Organization
Core training logic lives in `train.py`, with supporting modules in `model.py` (architecture and hybrid loss), `data.py` (pairwise dataset prep), `predict.py`, and `calculate_metrics.py`. Hydra configs sit under `configs/`, anchored by `config.yaml` and training overrides in `configs/training/`. Expected inputs reside in `Data/redsm5/` and `Data/DSM-5/`, while `outputs/<timestamp>/` captures run artefacts, checkpoints, and Hydra snapshots. Utility scripts live in `scripts/`, and lightweight validation scripts are `test_setup.py` and `test_training.py`.

## Build, Test, and Development Commands
- `pip install -r requirements.txt` — install Python dependencies.
- `python train.py` — launch default Hydra training; override with flags such as `python train.py training=hpo optuna.n_trials=50`.
- `python predict.py --checkpoint_path outputs/<run>/best_model.pt --output_dir outputs/<run>/predictions` — export inference results.
- `python calculate_metrics.py` — recompute aggregate metrics for a finished run.

## Coding Style & Naming Conventions
Follow standard PEP 8: four-space indentation, descriptive snake_case identifiers, and module-level constants in CAPS. Maintain type hints (see `train.py`) and prefer single-quoted strings for consistency. Keep Hydra config names lowercase with underscores (`training/default.yaml`), and align experiment branches with the `scripts/git_workflow.sh` pattern `experiment/<focus-area>`.

## Testing Guidelines
Use targeted smoke tests before proposing changes. `python test_setup.py` checks data availability and a minimal forward pass (requires the CSV/JSON files in `Data/`). `python test_training.py` validates config loading, hardware toggles, and checkpoint cleanup. Investigate any ✗ output and keep temporary artefacts out of version control. Expand coverage with focused PyTest cases when modifying training or loss logic.

## Commit & Pull Request Guidelines
Existing history uses concise, imperative summaries (e.g., "Fix HPO issues and enable fresh study restart"). Mirror that format, keep the first line under ~72 characters, and add context in the body if needed. The helper `scripts/git_workflow.sh commit "Message"` embeds timestamp metadata; update totals before invoking it. Pull requests should link the relevant issue, list Hydra overrides or Optuna settings, summarise validation metrics, note data prerequisites, and point reviewers to the run directory under `outputs/`.

## Experiment Tracking & Data Notes
Do not commit raw datasets; store them under `Data/` locally to satisfy path assumptions in configs. After each experiment, capture the resolved config from `outputs/<timestamp>/hydra/` and surface per-criterion metrics when reporting results. Rotate old artefacts with `scripts/git_workflow.sh clean-outputs` to keep only the five newest runs.
