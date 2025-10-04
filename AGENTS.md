# Repository Guidelines

## Project Structure & Module Organization
Training flows through `train.py`, which wires together the architecture and hybrid loss in `model.py`, data prep helpers in `data.py`, and Hydra configuration under `configs/`. Default settings live in `configs/config.yaml`, while overrides sit in `configs/training/`. Input artefacts must be staged in `Data/redsm5/` and `Data/DSM-5/`. Outputs land in `outputs/<timestamp>/`, including checkpoints (`best_model.pt`), Hydra snapshots, and metrics. CLI utilities reside in `scripts/`, and quick validation helpers are `test_setup.py` and `test_training.py`.

## Build, Test, and Development Commands
- `pip install -r requirements.txt` — install pinned training dependencies.
- `python train.py` — launch the default Hydra job; append overrides such as `training=hpo optuna.n_trials=50` for sweeps.
- `python predict.py --checkpoint_path outputs/<run>/best_model.pt --output_dir outputs/<run>/predictions` — export predictions from a saved checkpoint.
- `python calculate_metrics.py` — recompute aggregate scores for a completed run.
- `scripts/git_workflow.sh clean-outputs` — prune older run directories, keeping the five newest.

## Coding Style & Naming Conventions
Follow PEP 8 with four-space indentation, snake_case identifiers, and module constants in CAPS. Prefer single-quoted strings for consistency. Add type hints when touching training or data paths. Hydra config filenames stay lowercase with underscores (e.g., `training/default.yaml`). Name experiment branches `experiment/<focus-area>` to match `scripts/git_workflow.sh`.

## Testing Guidelines
Run `python test_setup.py` before committing to confirm data access and a minimal forward pass (requires the CSV/JSON inputs). Use `python test_training.py` to verify config loading, hardware flags, and checkpoint cleanup. Investigate any ✗ output and avoid committing artefacts under `outputs/`. Add focused PyTest coverage when introducing new loss logic or data transforms.

## Commit & Pull Request Guidelines
Use concise, imperative commit messages under ~72 characters (e.g., "Enable fresh HPO restart"). When using the helper script, call `scripts/git_workflow.sh commit "Message"` after updating totals. Pull requests should link the issue, list Hydra overrides or Optuna settings, summarize key metrics, note data prerequisites, and point reviewers to the relevant `outputs/<timestamp>/` directory.

## Experiment Tracking Notes
Never version raw data; keep it only under `Data/`. Record resolved configs from `outputs/<timestamp>/hydra/` and capture per-criterion metrics when sharing results. Archive old runs instead of deleting them if you need reproducibility data.
