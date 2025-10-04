import torch
import json
from pathlib import Path

# Load the checkpoint
checkpoint_path = "outputs/training/20250930_170357/best_model.pt"
checkpoint = torch.load(checkpoint_path, map_location='cpu')

# Display the keys in the checkpoint
print("Keys in best_model.pt:")
for key in checkpoint.keys():
    print(f"  {key}")

print("\n" + "="*50)

# Show detailed info for each key (except large tensors)
for key, value in checkpoint.items():
    print(f"\n{key}:")
    if key == 'model_state_dict':
        print(f"  Type: {type(value)}")
        print(f"  Number of parameters: {len(value)}")
        print("  Sample parameter names:")
        for i, param_name in enumerate(list(value.keys())[:5]):
            print(f"    {param_name}: {value[param_name].shape}")
        if len(value) > 5:
            print(f"    ... and {len(value) - 5} more parameters")
    elif key == 'optimizer_state_dict':
        print(f"  Type: {type(value)}")
        print(f"  Keys: {list(value.keys())}")
    elif key == 'config':
        print(f"  Type: {type(value)}")
        print("  Configuration structure:")
        config_str = json.dumps(value, indent=2)
        print(config_str[:1000] + "..." if len(config_str) > 1000 else config_str)
    elif key == 'history':
        print(f"  Type: {type(value)}")
        print(f"  Length: {len(value)} epochs")
        if value:
            print("  Sample history entry:")
            print(json.dumps(value[0], indent=2))
    else:
        print(f"  Type: {type(value)}")
        print(f"  Value: {value}")