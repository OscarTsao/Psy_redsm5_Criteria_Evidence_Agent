import torch
from model import SpanBERTForDSM5Classification, get_model
from data_preprocessing import prepare_data, split_data, create_datasets

print("Testing SpanBERT setup...")

# Test data loading
print("\n1. Testing data preprocessing...")
df = prepare_data(
    "Data/redsm5/redsm5_posts.csv",
    "Data/redsm5/redsm5_annotations.csv",
    "Data/DSM-5/DSM_Criteria_Array_Fixed_Major_Depressive.json"
)
print(f"   ✓ Loaded {len(df)} posts")

# Test data splitting
train_df, val_df, test_df = split_data(df)
print(f"   ✓ Split data: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

# Test model initialization
print("\n2. Testing model initialization...")
try:
    model, device = get_model(num_criteria=9, device='cpu')
    print(f"   ✓ Model initialized on {device}")

    # Test forward pass with dummy input
    dummy_input_ids = torch.randint(0, 1000, (2, 128)).to(device)
    dummy_attention_mask = torch.ones(2, 128).to(device)

    with torch.no_grad():
        output = model(dummy_input_ids, dummy_attention_mask)

    print(f"   ✓ Forward pass successful, output shape: {output.shape}")
    assert output.shape == (2, 9), "Output shape mismatch"

except Exception as e:
    print(f"   ✗ Error: {e}")

print("\n3. Testing dataset creation...")
try:
    train_dataset, val_dataset, test_dataset, tokenizer = create_datasets(
        train_df[:10], val_df[:5], test_df[:5]  # Small subset for testing
    )
    print(f"   ✓ Datasets created successfully")

    # Test single sample
    sample = train_dataset[0]
    print(f"   ✓ Sample keys: {sample.keys()}")
    print(f"   ✓ Input shape: {sample['input_ids'].shape}")
    print(f"   ✓ Labels shape: {sample['labels'].shape}")

except Exception as e:
    print(f"   ✗ Error: {e}")

print("\n✅ All tests passed! Ready for training.")