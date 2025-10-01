#!/usr/bin/env python3
"""
Test script to compare CSV vs JSON loading performance for groundtruth data.
"""
import time
import pandas as pd
import json
import numpy as np

def test_csv_loading():
    """Test CSV loading performance."""
    start_time = time.time()
    df = pd.read_csv('Data/groundtruth/redsm5_ground_truth.csv')
    load_time = time.time() - start_time

    # Test basic operations
    start_time = time.time()
    sample_data = df.head(1000).copy()
    process_time = time.time() - start_time

    return {
        'format': 'CSV',
        'load_time': load_time,
        'process_time': process_time,
        'total_time': load_time + process_time,
        'memory_usage': df.memory_usage(deep=True).sum() / (1024**2),  # MB
        'rows': len(df),
        'columns': len(df.columns)
    }

def test_json_loading():
    """Test JSON loading performance."""
    start_time = time.time()
    with open('Data/groundtruth/redsm5_ground_truth.json', 'r') as f:
        data = [json.loads(line) for line in f]
    df = pd.DataFrame(data)
    load_time = time.time() - start_time

    # Test basic operations
    start_time = time.time()
    sample_data = df.head(1000).copy()
    process_time = time.time() - start_time

    return {
        'format': 'JSON',
        'load_time': load_time,
        'process_time': process_time,
        'total_time': load_time + process_time,
        'memory_usage': df.memory_usage(deep=True).sum() / (1024**2),  # MB
        'rows': len(df),
        'columns': len(df.columns)
    }

def test_json_streaming():
    """Test JSON line-by-line streaming for memory efficiency."""
    start_time = time.time()
    rows = []
    with open('Data/groundtruth/redsm5_ground_truth.json', 'r') as f:
        for line in f:
            rows.append(json.loads(line))
    df = pd.DataFrame(rows)
    load_time = time.time() - start_time

    return {
        'format': 'JSON_STREAM',
        'load_time': load_time,
        'memory_usage': df.memory_usage(deep=True).sum() / (1024**2),  # MB
        'rows': len(df),
        'columns': len(df.columns)
    }

if __name__ == "__main__":
    print("Testing data loading performance...\n")

    # Test CSV
    print("Testing CSV loading...")
    csv_results = test_csv_loading()

    # Test JSON
    print("Testing JSON loading...")
    json_results = test_json_loading()

    # Test JSON streaming
    print("Testing JSON streaming...")
    stream_results = test_json_streaming()

    # Compare results
    print("\n" + "="*60)
    print("PERFORMANCE COMPARISON")
    print("="*60)

    for results in [csv_results, json_results, stream_results]:
        print(f"\nFormat: {results['format']}")
        print(f"Load time: {results['load_time']:.4f}s")
        if 'process_time' in results:
            print(f"Process time: {results['process_time']:.4f}s")
            print(f"Total time: {results['total_time']:.4f}s")
        print(f"Memory usage: {results['memory_usage']:.2f} MB")
        print(f"Rows: {results['rows']:,}")
        print(f"Columns: {results['columns']}")

    # Recommendation
    print("\n" + "="*60)
    print("RECOMMENDATION")
    print("="*60)

    if csv_results['total_time'] < json_results['total_time']:
        print("CSV is faster for loading and processing.")
        recommended = "CSV"
    else:
        print("JSON is faster for loading and processing.")
        recommended = "JSON"

    if csv_results['memory_usage'] < json_results['memory_usage']:
        print("CSV uses less memory.")
    else:
        print("JSON uses less memory.")

    print(f"\nRecommended format: {recommended}")