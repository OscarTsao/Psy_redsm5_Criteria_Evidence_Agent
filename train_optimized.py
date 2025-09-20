#!/usr/bin/env python3
"""
Optimized training script for BERT Large DSM-5 classification with RTX 3090 optimizations.
This script includes all performance optimizations for maximum training speed.
"""

import argparse
import os
import time

def main():
    parser = argparse.ArgumentParser(description='Train optimized BERT for DSM-5 classification')
    parser.add_argument('--posts_path', type=str, default='Data/redsm5/redsm5_posts.csv')
    parser.add_argument('--annotations_path', type=str, default='Data/redsm5/redsm5_annotations.csv')
    parser.add_argument('--criteria_path', type=str, default='Data/DSM-5/DSM_Criteria_Array_Fixed_Major_Depressive.json')
    parser.add_argument('--model_name', type=str, default='google-bert/bert-large-uncased-whole-word-masking-finetuned-squad')

    # Optimized settings for RTX 3090 (24GB VRAM)
    parser.add_argument('--batch_size', type=int, default=32, help='Increased batch size for RTX 3090')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=2, help='Effective batch size = batch_size * accumulation_steps')
    parser.add_argument('--num_workers', type=int, default=8, help='Optimized for multi-core CPU')

    # Training parameters
    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--learning_rate', type=float, default=2e-5)
    parser.add_argument('--warmup_ratio', type=float, default=0.1)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--patience', type=int, default=5)

    # Performance optimizations
    parser.add_argument('--use_amp', action='store_true', default=True, help='Use Automatic Mixed Precision (FP16)')
    parser.add_argument('--use_compile', action='store_true', default=True, help='Use torch.compile optimization')
    parser.add_argument('--use_focal_loss', action='store_true', help='Use Focal Loss for class imbalance')

    # Output
    parser.add_argument('--output_dir', type=str, default='outputs_optimized')
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    print("=== Optimized BERT Large Training for DSM-5 Classification ===")
    print(f"Effective batch size: {args.batch_size * args.gradient_accumulation_steps}")
    print(f"Mixed precision (FP16): {args.use_amp}")
    print(f"Torch compile: {args.use_compile}")
    print(f"Data workers: {args.num_workers}")
    print(f"Output directory: {args.output_dir}")
    print()

    # Build the training command
    train_cmd = [
        'python', 'train.py',
        '--posts_path', args.posts_path,
        '--annotations_path', args.annotations_path,
        '--criteria_path', args.criteria_path,
        '--model_name', args.model_name,
        '--batch_size', str(args.batch_size),
        '--gradient_accumulation_steps', str(args.gradient_accumulation_steps),
        '--num_workers', str(args.num_workers),
        '--num_epochs', str(args.num_epochs),
        '--learning_rate', str(args.learning_rate),
        '--warmup_ratio', str(args.warmup_ratio),
        '--max_grad_norm', str(args.max_grad_norm),
        '--dropout', str(args.dropout),
        '--threshold', str(args.threshold),
        '--patience', str(args.patience),
        '--output_dir', args.output_dir,
        '--seed', str(args.seed)
    ]

    if args.use_amp:
        train_cmd.append('--use_amp')
    if args.use_compile:
        train_cmd.append('--use_compile')
    if args.use_focal_loss:
        train_cmd.append('--use_focal_loss')

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Run training
    print("Starting optimized training...")
    start_time = time.time()

    import subprocess
    result = subprocess.run(train_cmd, capture_output=False)

    end_time = time.time()
    training_time = end_time - start_time

    print(f"\nTraining completed in {training_time:.2f} seconds ({training_time/60:.2f} minutes)")

    if result.returncode == 0:
        print("✅ Training successful!")
    else:
        print("❌ Training failed with return code:", result.returncode)

    return result.returncode

if __name__ == "__main__":
    exit(main())