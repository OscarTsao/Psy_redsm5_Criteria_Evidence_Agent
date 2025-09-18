# SpanBERT DSM-5 Criteria Classification

This project implements a SpanBERT-based multi-label classification system for matching DSM-5 Major Depressive Disorder criteria to Reddit posts from the RedSM5 dataset.

## Project Structure

- `data_preprocessing.py`: Data loading and preprocessing utilities
- `model.py`: SpanBERT model architecture for multi-label classification
- `train.py`: Training script with fine-tuning capabilities
- `predict.py`: Prediction and evaluation script
- `requirements.txt`: Required Python packages

## Criteria Mapping

The system classifies text into 9 DSM-5 criteria (A.1 to A.9):

1. **A.1**: Depressed mood → `DEPRESSED_MOOD`
2. **A.2**: Anhedonia → `ANHEDONIA`
3. **A.3**: Appetite change → `APPETITE_CHANGE`
4. **A.4**: Sleep issues → `SLEEP_ISSUES`
5. **A.5**: Psychomotor changes → `PSYCHOMOTOR`
6. **A.6**: Fatigue → `FATIGUE`
7. **A.7**: Worthlessness → `WORTHLESSNESS`
8. **A.8**: Cognitive issues → `COGNITIVE_ISSUES`
9. **A.9**: Suicidal thoughts → `SUICIDAL_THOUGHTS`

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Training

```bash
python train.py \
    --posts_path Data/redsm5/redsm5_posts.csv \
    --annotations_path Data/redsm5/redsm5_annotations.csv \
    --criteria_path Data/DSM-5/DSM_Criteria_Array_Fixed_Major_Depressive.json \
    --num_epochs 20 \
    --batch_size 16 \
    --learning_rate 2e-5 \
    --use_focal_loss \
    --output_dir outputs
```

### Prediction and Evaluation

```bash
python predict.py \
    --checkpoint_path outputs/best_model.pt \
    --posts_path Data/redsm5/redsm5_posts.csv \
    --annotations_path Data/redsm5/redsm5_annotations.csv \
    --criteria_path Data/DSM-5/DSM_Criteria_Array_Fixed_Major_Depressive.json \
    --threshold 0.5 \
    --output_dir outputs
```

## Output Files

The system generates:
- `predictions.csv`: Contains post IDs, predicted criteria labels, ground truth labels, and probabilities
- `evaluation_metrics.json`: Detailed metrics including precision, recall, F1, and AUC
- `evaluation_summary.txt`: Human-readable metrics summary
- `training_history.json`: Training and validation metrics per epoch

## Model Architecture

- Base Model: SpanBERT (spanbert-base-cased)
- Classification Head: Two-layer MLP with dropout
- Loss Function: Binary Cross-Entropy or Focal Loss (for imbalanced data)
- Output: Multi-label predictions for 9 criteria

## Evaluation Metrics

- **Macro/Micro Precision, Recall, F1**: Overall performance
- **Hamming Loss**: Fraction of wrong labels
- **Exact Match Ratio**: Percentage of samples with all labels correctly predicted
- **AUC-ROC**: Area under the ROC curve
- **Per-criteria metrics**: Individual performance for each DSM-5 criterion