# inspect_pt.py
import torch
import sys
import pprint # For cleaner dictionary printing

if len(sys.argv) != 2:
    print("Usage: python inspect_pt.py <path_to_pt_file>")
    sys.exit(1)

file_path = sys.argv[1]

try:
    print(f"Loading checkpoint: {file_path}")
    # Load onto CPU to avoid GPU issues
    checkpoint = torch.load(file_path, map_location='cpu')
    print(f"\nCheckpoint type: {type(checkpoint)}")

    if isinstance(checkpoint, dict):
        print("\n--- Top-level keys ---")
        # Use pprint for potentially nested dicts like config
        keys_list = list(checkpoint.keys())
        print(keys_list)

        print("\n--- Checking for 'model_state_dict' ---")
        if 'model_state_dict' in checkpoint:
            print("  FOUND 'model_state_dict' key.")
            state_dict = checkpoint['model_state_dict']
            print(f"  Type of value: {type(state_dict)}")
            if isinstance(state_dict, dict):
                 num_tensors = len(state_dict)
                 print(f"  Number of tensors inside: {num_tensors}")
                 print("  Sample keys inside 'model_state_dict':")
                 for i, k in enumerate(state_dict.keys()):
                      if i >= 10: break
                      print(f"    - {k}")
            else:
                 print("  WARNING: Value under 'model_state_dict' is NOT a dictionary!")
        else:
            print("  ERROR: 'model_state_dict' key NOT FOUND!")

        # Check other keys
        print("\n--- Checking other keys ---")
        if 'config' in checkpoint: print("- Found 'config'")
        if 'hyper_parameters' in checkpoint: print("- Found 'hyper_parameters'")
        if 'pruning_info' in checkpoint: print("- Found 'pruning_info'")

    else:
        print("\nCheckpoint is not a dictionary.")

except Exception as e:
    print(f"\nAn error occurred during inspection: {e}")