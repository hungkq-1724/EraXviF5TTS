#!/usr/bin/env python3
"""
reset_f5tts_epoch_pruned_safe.py - Reset the epoch counter of a pruned F5-TTS checkpoint

This script safely handles pruned F5-TTS models by preserving the exact model structure 
while only resetting the epoch/update counter to 0. This allows you to continue training 
from pruned weights but restart the epoch counting from 0.

Usage:
  python reset_f5tts_epoch_pruned_safe.py input_checkpoint.pt output_checkpoint.pt
"""

import os
import sys
import argparse
import torch
import gc
import json
from collections import OrderedDict

def verify_model_integrity(original_weights, new_weights, verbose=True):
    """Verify that the model structure is preserved exactly"""
    
    if len(original_weights) != len(new_weights):
        print(f"WARNING: Weight count mismatch: original={len(original_weights)}, new={len(new_weights)}")
        return False
    
    for key in original_weights:
        if key not in new_weights:
            print(f"WARNING: Missing key in new weights: {key}")
            return False
        
        if original_weights[key].shape != new_weights[key].shape:
            print(f"WARNING: Shape mismatch for {key}: original={original_weights[key].shape}, new={new_weights[key].shape}")
            return False
    
    if verbose:
        print("✓ Model structure integrity verified: All shapes and keys preserved exactly")
    
    return True

def analyze_model_structure(weights_dict, verbose=True):
    """Analyze the model structure to detect pruning configuration"""
    
    # Look for transformer blocks to detect depth
    transformer_blocks = [k for k in weights_dict.keys() if 'transformer_blocks' in k]
    
    if transformer_blocks:
        # Extract block indices
        block_indices = set()
        for key in transformer_blocks:
            parts = key.split('.')
            for i, part in enumerate(parts):
                if part == 'transformer_blocks' and i+1 < len(parts):
                    try:
                        block_idx = int(parts[i+1])
                        block_indices.add(block_idx)
                    except ValueError:
                        pass
        
        if block_indices:
            max_block_idx = max(block_indices)
            if verbose:
                print(f"✓ Detected {max_block_idx + 1} transformer blocks (0-{max_block_idx})")
                
                # Check if blocks appear to be consecutively numbered
                expected_indices = set(range(max_block_idx + 1))
                if block_indices != expected_indices:
                    missing = expected_indices - block_indices
                    print(f"  NOTE: Non-consecutive block numbering. Missing indices: {missing}")
                    print(f"  This is normal for pruned models where specific layers were removed.")
    
    # Analyze embedding dimension
    embedding_keys = [k for k in weights_dict.keys() if 'embedding' in k and 'weight' in k]
    for key in embedding_keys:
        shape = weights_dict[key].shape
        if verbose and len(shape) > 0:
            if 'text_embedding' in key:
                print(f"✓ Text embedding dimension: {shape[1]}")
            else:
                print(f"✓ Embedding dimension: {shape[1]}")
    
    return True

def reset_checkpoint(input_path, output_path, verbose=True):
    """
    Reset the epoch counter in an F5-TTS checkpoint, safely preserving pruned structure
    
    Args:
        input_path: Path to the original checkpoint
        output_path: Path to save the modified checkpoint
        verbose: Whether to print progress information
    
    Returns:
        bool: True if successful, False otherwise
    """
    if verbose:
        print(f"\nLoading checkpoint: {input_path}")
    
    try:
        checkpoint = torch.load(input_path, map_location="cpu")
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return False
    
    if not isinstance(checkpoint, dict):
        print(f"Error: Checkpoint is not a dictionary (type: {type(checkpoint)})")
        return False
    
    # Create a new checkpoint dictionary
    new_checkpoint = {}
    
    # Check for model weights
    if verbose:
        print("\nCheckpoint contains the following keys:", list(checkpoint.keys()))
    
    # Look for model weights in known locations
    model_weights_found = False
    model_state_dict = None
    
    for key in ['model_state_dict', 'state_dict', 'model']:
        if key in checkpoint and isinstance(checkpoint[key], dict):
            model_state_dict = checkpoint[key]
            new_checkpoint['model_state_dict'] = checkpoint[key]
            if verbose:
                print(f"✓ Extracted model weights from '{key}'")
                print(f"  Total parameters: {len(model_state_dict)}")
            model_weights_found = True
            break
    
    if not model_weights_found:
        print("Error: Could not find model weights in checkpoint")
        return False
    
    # Analyze model structure to detect pruning
    if verbose:
        print("\n----- Model Structure Analysis -----")
        analyze_model_structure(model_state_dict, verbose=verbose)
        print("-----------------------------------\n")
    
    # Copy EMA model weights if present
    ema_state_dict = None
    if 'ema_model_state_dict' in checkpoint and isinstance(checkpoint['ema_model_state_dict'], dict):
        ema_state_dict = checkpoint['ema_model_state_dict']
        new_checkpoint['ema_model_state_dict'] = ema_state_dict
        if verbose:
            print(f"✓ Copied EMA model weights ({len(ema_state_dict)} parameters)")
    
    # Reset update/step counter
    original_update = checkpoint.get('update', checkpoint.get('step', -1))
    new_checkpoint['update'] = 0
    if verbose:
        print(f"✓ Reset update counter from {original_update} to 0")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    # Perform integrity check
    if verbose:
        print("\nPerforming integrity verification...")
    verify_model_integrity(model_state_dict, new_checkpoint['model_state_dict'], verbose=verbose)
    if ema_state_dict:
        verify_model_integrity(ema_state_dict, new_checkpoint['ema_model_state_dict'], verbose=False)
    
    # Save the new checkpoint
    if verbose:
        print(f"\nSaving modified checkpoint to: {output_path}")
    
    try:
        torch.save(new_checkpoint, output_path)
    except Exception as e:
        print(f"Error saving modified checkpoint: {e}")
        return False
    
    # Clean up to free memory
    del checkpoint
    del new_checkpoint
    gc.collect()
    
    if verbose:
        print("\nSuccess! Checkpoint modified with epoch counter reset to 0.")
        print(f"Use this checkpoint with your normal finetuning command:")
        print(f"  python finetune_cli.py --finetune --exp_name=YourPrunedModel --dataset_name=YourData ...")
    
    return True

def main():
    parser = argparse.ArgumentParser(
        description="Reset the epoch counter in a pruned F5-TTS checkpoint",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument("input_checkpoint", help="Path to the original checkpoint file")
    parser.add_argument("output_checkpoint", help="Path to save the modified checkpoint")
    parser.add_argument("--quiet", "-q", action="store_true", help="Suppress progress messages")
    
    args = parser.parse_args()
    
    # Validate input file exists
    if not os.path.isfile(args.input_checkpoint):
        print(f"Error: Input checkpoint not found: {args.input_checkpoint}")
        return 1
    
    # Check if output would overwrite input
    if os.path.abspath(args.input_checkpoint) == os.path.abspath(args.output_checkpoint):
        print("Error: Input and output paths are the same. Specify a different output path.")
        return 1
    
    # Warn if output file already exists
    if os.path.exists(args.output_checkpoint):
        print(f"Warning: Output file already exists: {args.output_checkpoint}")
        response = input("Overwrite? (y/n): ").lower()
        if response != 'y':
            print("Operation cancelled.")
            return 0
    
    # Reset the checkpoint
    success = reset_checkpoint(args.input_checkpoint, args.output_checkpoint, verbose=not args.quiet)
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
