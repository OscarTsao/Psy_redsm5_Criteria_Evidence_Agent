#!/usr/bin/env python3
"""
Performance benchmarking script to compare training speeds with different optimization levels.
"""

import torch
import time
import argparse
from torch.cuda.amp import autocast
from model import get_model

def benchmark_forward_pass(model, batch_size=16, seq_length=512, num_iterations=100, use_amp=False, use_compile=False):
    """Benchmark forward pass speed."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if use_compile and hasattr(torch, 'compile'):
        model = torch.compile(model, mode='max-autotune')

    # Generate dummy data
    input_ids = torch.randint(0, 30522, (batch_size, seq_length)).to(device)
    attention_mask = torch.ones((batch_size, seq_length)).to(device)

    model.eval()

    # Warmup
    for _ in range(10):
        if use_amp:
            with autocast():
                _ = model(input_ids, attention_mask)
        else:
            _ = model(input_ids, attention_mask)

    torch.cuda.synchronize()

    # Benchmark
    start_time = time.time()
    for _ in range(num_iterations):
        if use_amp:
            with autocast():
                _ = model(input_ids, attention_mask)
        else:
            _ = model(input_ids, attention_mask)

    torch.cuda.synchronize()
    end_time = time.time()

    avg_time = (end_time - start_time) / num_iterations
    throughput = batch_size / avg_time

    return avg_time, throughput

def main():
    parser = argparse.ArgumentParser(description='Benchmark BERT Large performance')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for benchmarking')
    parser.add_argument('--seq_length', type=int, default=512, help='Sequence length')
    parser.add_argument('--num_iterations', type=int, default=100, help='Number of iterations')

    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available. Exiting.")
        return

    print("=== BERT Large Performance Benchmark ===")
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"Batch size: {args.batch_size}")
    print(f"Sequence length: {args.seq_length}")
    print(f"Iterations: {args.num_iterations}")
    print()

    # Initialize model
    model, device = get_model('google-bert/bert-large-uncased-whole-word-masking-finetuned-squad', num_criteria=9)

    configurations = [
        ("Baseline (FP32)", False, False),
        ("Mixed Precision (FP16)", True, False),
        ("Compiled (FP32)", False, True),
        ("Compiled + Mixed Precision", True, True),
    ]

    results = []

    for config_name, use_amp, use_compile in configurations:
        print(f"Testing {config_name}...")

        try:
            avg_time, throughput = benchmark_forward_pass(
                model, args.batch_size, args.seq_length, args.num_iterations, use_amp, use_compile
            )
            results.append((config_name, avg_time, throughput))
            print(f"  Average time: {avg_time:.4f}s")
            print(f"  Throughput: {throughput:.2f} samples/sec")

            # Memory usage
            memory_used = torch.cuda.max_memory_allocated() / 1024**3  # GB
            print(f"  Peak memory: {memory_used:.2f} GB")
            print()

        except Exception as e:
            print(f"  Failed: {e}")
            print()

        # Reset memory stats
        torch.cuda.reset_peak_memory_stats()

    # Summary
    print("=== Performance Summary ===")
    baseline_throughput = results[0][2] if results else 1

    for config_name, avg_time, throughput in results:
        speedup = throughput / baseline_throughput
        print(f"{config_name:25}: {throughput:6.2f} samples/sec ({speedup:.2f}x speedup)")

    print()
    print("Recommended settings for RTX 3090:")
    print("- Batch size: 32-64 (with gradient accumulation)")
    print("- Mixed precision: Enabled")
    print("- torch.compile: Enabled")
    print("- Gradient checkpointing: Enabled")
    print("- Data workers: 4-8")

if __name__ == "__main__":
    main()