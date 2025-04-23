# --- START OF MODIFIED FILE definitive-f5tts-pruner.py ---

import torch
import numpy as np
import os
import argparse
import json
import logging
import re
import datetime
from collections import defaultdict
from safetensors.torch import load_file as load_safetensors_file
# We are only saving as .pt, so save_safetensors_file is not needed

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_checkpoint_data(model_path):
    """
    Loads checkpoint data from either .pt or .safetensors file.
    Detects and handles EMA prefixes ('ema_model.').
    Strips EMA prefix AND ignores common metadata keys ('initted', 'step').

    Returns:
        tuple: (full_checkpoint_data, state_dict, config, state_dict_key, is_ema)
    """
    state_dict = None
    config = None
    state_dict_key = None
    is_ema = False
    full_checkpoint_data = None # Populated for .pt or companion .pt
    ignore_metadata_keys = {"initted", "step"} # Define keys to ignore explicitly

    if model_path.endswith(".safetensors"):
        logger.info(f"Loading state_dict from safetensors file: {model_path}")
        raw_state_dict = load_safetensors_file(model_path, device='cpu')
        logger.info(f"Loaded {len(raw_state_dict)} tensors from safetensors.")

        ema_prefix = "ema_model."
        num_ema_keys = sum(1 for k in raw_state_dict if k.startswith(ema_prefix))
        if num_ema_keys > 0 and num_ema_keys >= 0.8 * len(raw_state_dict):
            logger.info("Detected 'ema_model.' prefix in safetensors keys. Assuming EMA model.")
            is_ema = True
            state_dict_key = 'ema_model_state_dict'
            state_dict = {}
            for k, v in raw_state_dict.items():
                # Skip ignore keys regardless of prefix
                if k in ignore_metadata_keys or k.replace(ema_prefix,'') in ignore_metadata_keys:
                    logger.debug(f"Ignoring metadata key: {k}")
                    continue
                # Strip prefix if present
                if k.startswith(ema_prefix):
                    state_dict[k[len(ema_prefix):]] = v
                else:
                    # Keep non-prefix, non-ignored keys (log warning)
                    logger.warning(f"Keeping key '{k}' without expected EMA prefix and not in ignore list.")
                    state_dict[k] = v
            logger.info(f"Stripped '{ema_prefix}' prefix and ignored metadata keys from state_dict.")
        else:
            logger.info("Did not detect significant 'ema_model.' prefix. Assuming non-EMA model.")
            is_ema = False
            state_dict = {} # Build clean dict even for non-EMA
            for k,v in raw_state_dict.items():
                 if k in ignore_metadata_keys:
                      logger.debug(f"Ignoring metadata key: {k}")
                      continue
                 state_dict[k] = v
            # Try to infer state_dict_key later if needed

        # [...] Companion .pt loading logic remains the same...
        pt_path_guess = model_path.replace(".safetensors", ".pt")
        if os.path.exists(pt_path_guess):
            logger.info(f"Attempting to load metadata from companion .pt file: {pt_path_guess}")
            try:
                companion_ckpt = torch.load(pt_path_guess, map_location='cpu')
                full_checkpoint_data = companion_ckpt
                if isinstance(companion_ckpt, dict):
                    if 'config' in companion_ckpt: config = companion_ckpt['config']; logger.info("Loaded 'config' from companion.")
                    elif 'hyper_parameters' in companion_ckpt: config = companion_ckpt['hyper_parameters']; logger.info("Loaded 'hyper_parameters' from companion.")
                    if 'ema_model_state_dict' in companion_ckpt:
                        if not is_ema: logger.warning("Companion .pt has EMA key, overriding safetensors EMA detection."); is_ema = True
                        state_dict_key = 'ema_model_state_dict'
                    elif any(k in companion_ckpt for k in ['model', 'model_state_dict', 'state_dict']):
                         if is_ema: logger.warning(f"Safetensors detected EMA, but companion .pt has non-EMA key. Sticking with is_ema=True.")
                         elif state_dict_key is None:
                             if 'model' in companion_ckpt: state_dict_key = 'model'
                             elif 'model_state_dict' in companion_ckpt: state_dict_key = 'model_state_dict'
                             elif 'state_dict' in companion_ckpt: state_dict_key = 'state_dict'
            except Exception as e: logger.warning(f"Could not load or parse companion .pt file '{pt_path_guess}': {e}")
        else: logger.warning(f"Companion .pt file '{pt_path_guess}' not found.")
        if state_dict_key is None:
             if not is_ema: state_dict_key = 'state_dict'; logger.warning("Assuming state_dict_key='state_dict'.")


    elif model_path.endswith(".pt"):
        logger.info(f"Loading checkpoint from .pt file: {model_path}")
        checkpoint = torch.load(model_path, map_location='cpu')
        full_checkpoint_data = checkpoint
        if not isinstance(checkpoint, dict): raise ValueError(".pt file not a dict")

        # Find config (remains same)
        if 'config' in checkpoint: config = checkpoint['config']; logger.info("Found 'config' key.")
        elif 'hyper_parameters' in checkpoint: config = checkpoint['hyper_parameters']; logger.info("Found 'hyper_parameters' key.")
        else: logger.warning("No 'config' or 'hyper_parameters' key.")

        raw_state_dict_source = None
        # Find raw state dict source and type
        if 'ema_model_state_dict' in checkpoint:
            raw_state_dict_source = checkpoint['ema_model_state_dict']
            state_dict_key = 'ema_model_state_dict'; is_ema = True
            logger.info("Found EMA state dict under 'ema_model_state_dict'")
        elif 'model' in checkpoint: raw_state_dict_source = checkpoint['model']; state_dict_key = 'model'; logger.info("Found state dict under 'model'")
        elif 'model_state_dict' in checkpoint: raw_state_dict_source = checkpoint['model_state_dict']; state_dict_key = 'model_state_dict'; logger.info("Found state dict under 'model_state_dict'")
        elif 'state_dict' in checkpoint: raw_state_dict_source = checkpoint['state_dict']; state_dict_key = 'state_dict'; logger.info("Found state dict under 'state_dict'")
        else: # Guessing logic (remains same)
            for key in checkpoint.keys():
                 if key not in ['config', 'hyper_parameters', 'optimizer_states', 'lr_schedulers'] and isinstance(checkpoint[key], dict) and len(checkpoint[key]) > 10:
                     if any('weight' in k or 'bias' in k for k in checkpoint[key].keys()):
                          raw_state_dict_source = checkpoint[key]; state_dict_key = key; logger.info(f"Found potential state dict under '{key}'"); break
            else:
                 if all(isinstance(v, torch.Tensor) for k,v in checkpoint.items() if k not in ['config', 'hyper_parameters']) and len(checkpoint)>10 :
                     raw_state_dict_source = checkpoint; state_dict_key = None; logger.warning("Using entire checkpoint as state_dict.")
                 else: raise ValueError("Could not identify state_dict in .pt file.")

        # Process the found raw_state_dict_source
        state_dict = {}
        ema_prefix = "ema_model."
        model_prefix = "model." # Common prefix for non-EMA PL models
        prefix_to_strip = None
        if is_ema:
             prefix_to_strip = ema_prefix # Assume internal keys might also have it
        elif not is_ema: # Check for 'model.' prefix in non-EMA
             first_key = next(iter(raw_state_dict_source.keys()), None)
             if first_key and all(k.startswith(model_prefix) for k in raw_state_dict_source.keys()):
                   prefix_to_strip = model_prefix

        logger.info(f"Processing state dict. EMA={is_ema}. Prefix to strip='{prefix_to_strip}'")
        for k, v in raw_state_dict_source.items():
             if k in ignore_metadata_keys:
                  logger.debug(f"Ignoring metadata key: {k}")
                  continue
             # Check prefix after ignoring metadata
             current_key = k
             if prefix_to_strip and k.startswith(prefix_to_strip):
                  current_key = k[len(prefix_to_strip):]
             state_dict[current_key] = v
        if prefix_to_strip: logger.info(f"Stripped prefix '{prefix_to_strip}'")


    else: raise ValueError(f"Unsupported file extension: {model_path}")

    if state_dict is None: raise ValueError(f"Failed to load or identify state_dict from {model_path}")

    # Final check for empty state dict
    if not state_dict:
        logger.error("State dict is empty after processing! Check source file and ignore keys.")
        raise ValueError("Processed state_dict is empty.")

    return full_checkpoint_data, state_dict, config, state_dict_key, is_ema
    
class DefinitiveF5TTSPruner:
    """
    Definitive F5-TTS model pruner that correctly handles both transformer_blocks
    and text_blocks, supporting .pt and .safetensors inputs and saving to .pt.
    """

    def __init__(
            self,
            model_path: str,
            output_dir: str,
            output_name: str,
            target_layers: int = 8,
            device: str = "cuda" # Note: Device is not used much as loading is cpu
        ):
        """
        Initialize the pruner.
        """
        self.model_path = model_path
        self.output_dir = output_dir
        # Ensure output name ends with .pt
        if not output_name.lower().endswith(".pt"):
             logger.warning(f"Output name '{output_name}' does not end with .pt. Appending '.pt'.")
             self.output_name = output_name + ".pt"
        else:
             self.output_name = output_name
        self.target_layers = target_layers
        self.device = device # Keep it, though primarily CPU is used for loading/analysis

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Load and analyze checkpoint data
        logger.info(f"Loading checkpoint data from {model_path}")
        try:
             self.full_checkpoint_data, self.state_dict, self.config, self.state_dict_key, self.is_ema = load_checkpoint_data(model_path)
             logger.info("Checkpoint data loaded successfully.")
             # Add self.checkpoint as alias for compatibility if needed, prefer full_checkpoint_data
             self.checkpoint = self.full_checkpoint_data if self.full_checkpoint_data is not None else {} # Use empty dict if safetensors input
        except Exception as e:
             logger.error(f"Failed to load checkpoint data: {e}", exc_info=True)
             raise

        logger.info("Analyzing checkpoint structure")
        self.analyze_checkpoint_structure() # Renamed from analyze_checkpoint

    def analyze_checkpoint_structure(self):
        """Analyze the state dict and config to identify structure."""
        # State dict and config are already loaded in __init__
        if self.state_dict is None:
            raise RuntimeError("State dictionary was not loaded correctly.")

        logger.info(f"State dict key used/inferred: '{self.state_dict_key}'")
        logger.info(f"Is EMA model: {self.is_ema}")
        if self.config:
             logger.info("Config/Hyperparameters found.")
             # Optionally print part of the config: logger.info(f"Config sample: {dict(list(self.config.items())[:5])}")
        else:
             logger.warning("Config/Hyperparameters not found. Depth determination will rely solely on state_dict keys.")

        # Identify transformer_blocks and text_blocks from the state_dict
        self.analyze_block_structure()

    def analyze_block_structure(self):
        """Analyze transformer_blocks and text_blocks in the state dict."""
        # Note: Assumes self.state_dict has EMA prefix already stripped by load_checkpoint_data
        self.transformer_blocks = set()
        self.text_blocks = set()
        self.other_potential_blocks = defaultdict(set) # For debugging

        if not self.state_dict:
             logger.error("State dict is empty or None, cannot analyze block structure.")
             return

        # --- Define Patterns Based on F5/DiT Structure (after EMA strip) ---
        # --- THIS IS THE KEY PATTERN FROM THE LOGS ---
        transformer_pattern_f5_actual = r"transformer\.transformer_blocks\.(\d+)\."
        # ----------------------------------------------
        text_pattern_f5 = r"transformer\.text_embed\.text_blocks\.(\d+)\."

        # Include older/generic patterns as fallbacks (lower priority)
        transformer_pattern_generic_blocks = r"blocks\.(\d+)\."
        transformer_pattern_generic_layers = r"layers\.(\d+)\."
        transformer_pattern_original = r"transformer_blocks\.(\d+)\."
        text_pattern_original = r"text_blocks\.(\d+)\."

        # Prioritize the likely F5 pattern identified from logs
        possible_transformer_patterns = [
            transformer_pattern_f5_actual, # <<< Most likely pattern first
            transformer_pattern_generic_blocks,
            transformer_pattern_generic_layers,
            transformer_pattern_original
        ]
        possible_text_patterns = [
            text_pattern_f5, # Most likely F5 text pattern first
            text_pattern_original
        ]

        found_transformer_pattern = None
        found_text_pattern = None

        logger.info("Attempting to identify block key patterns in the (potentially cleaned) state_dict...")

        # Try to find transformer blocks
        for pattern in possible_transformer_patterns:
            temp_blocks = set()
            regex = re.compile(pattern)
            for key in self.state_dict.keys():
                match = regex.match(key)
                if match:
                    temp_blocks.add(int(match.group(1)))
            if temp_blocks:
                self.transformer_blocks = temp_blocks
                found_transformer_pattern = pattern
                logger.info(f"Identified transformer blocks using pattern: '{pattern}'")
                break # Success! Stop searching for transformer patterns

        # Try to find text blocks
        for pattern in possible_text_patterns:
             temp_blocks = set()
             regex = re.compile(pattern)
             for key in self.state_dict.keys():
                  match = regex.match(key)
                  if match:
                       temp_blocks.add(int(match.group(1)))
             if temp_blocks:
                  self.text_blocks = temp_blocks
                  found_text_pattern = pattern
                  logger.info(f"Identified text blocks using pattern: '{pattern}'")
                  break # Success! Stop searching for text patterns

        # --- Debugging if still not found (should not happen now for transformers) ---
        if not self.transformer_blocks:
            logger.error("Could not identify transformer blocks using known patterns even after EMA stripping and using pattern from logs.")
            # [...] (rest of the debug logic remains the same, but hopefully won't be needed)
            logger.warning("Analyzing key structure for potential block patterns...")
            for key in self.state_dict.keys():
                 match = re.match(r"([a-zA-Z0-9_\.]+)\.(\d+)\.", key)
                 if match:
                      prefix = match.group(1)
                      if '.' in prefix or prefix in ['blocks', 'layers', 'transformer_blocks']:
                           index = int(match.group(2))
                           self.other_potential_blocks[prefix].add(index)
            if self.other_potential_blocks:
                logger.warning("Found potential block structures:")
                for prefix, indices in sorted(self.other_potential_blocks.items()):
                     logger.warning(f"  Prefix: '{prefix}', Indices found: {sorted(list(indices))[:5]}... (Total: {len(indices)})")
            else: logger.error("Could not find any keys matching potential block patterns like 'name.number.'")
            sample_keys = list(self.state_dict.keys())[:20]
            logger.error(f"Sample CLEANED state_dict keys: {sample_keys}")
            raise ValueError("Failed to identify any transformer block structure in the state_dict.")
        # ---

        # --- Sort and proceed ---
        self.transformer_blocks = sorted(list(self.transformer_blocks))
        self.text_blocks = sorted(list(self.text_blocks))

        logger.info(f"Found transformer_blocks indices in state_dict: {self.transformer_blocks}")
        logger.info(f"Found text_blocks indices in state_dict: {self.text_blocks}")

        # --- Config/Depth checking logic remains the same ---
        self.num_layers = None
        if self.config and isinstance(self.config, dict):
            arch_config = self.config
            if 'model' in arch_config and isinstance(arch_config['model'], dict): arch_config = arch_config['model']
            if 'arch' in arch_config and isinstance(arch_config['arch'], dict): arch_config = arch_config['arch']
            depth_keys_to_check = ['depth', 'n_layer', 'num_hidden_layers']
            found_depth_key = None
            for key in depth_keys_to_check:
                if key in arch_config:
                    self.num_layers = arch_config[key]; found_depth_key = key
                    logger.info(f"Depth found in config ({found_depth_key}): {self.num_layers}")
                    break
            if self.num_layers is not None and self.transformer_blocks and len(self.transformer_blocks) != self.num_layers:
                logger.warning(f"Config depth ({self.num_layers}) doesn't match state_dict transformer blocks ({len(self.transformer_blocks)})! Using state_dict count.")
                self.num_layers = len(self.transformer_blocks)
            elif self.num_layers is None: logger.warning("Could not find depth key in config.")

        if self.num_layers is None:
            if not self.transformer_blocks:
                 logger.error("Cannot determine model depth: No config and no transformer_blocks identified.")
                 raise ValueError("Could not determine model depth.")
            self.num_layers = len(self.transformer_blocks)
            logger.info(f"Using number of transformer_blocks found in state_dict as depth: {self.num_layers}")
            
    def analyze_layer_importance(self):
        """Analyze layer importance based on transformer_block weights using robust SNR metrics."""
        logger.info("Analyzing transformer_block importance using SNR approach")
    
        # Prepare data structures for layer metrics
        weight_diversity = defaultdict(list)  # Signal component (entropy/diversity)
        weight_magnitudes = defaultdict(list)  # Signal strength (magnitude)
        noise_estimates = defaultdict(list)   # Noise component (variance)
        param_counts = defaultdict(int)       # Parameter counts per block
    
        # First, examine the state_dict keys to understand the structure
        all_keys = list(self.state_dict.keys())
        logger.info(f"Total keys in state_dict: {len(all_keys)}")
        sample_keys = all_keys[:10] if len(all_keys) > 10 else all_keys
        logger.info(f"Sample keys from state_dict: {sample_keys}")
        
        # Print details about transformer block indices
        logger.info(f"Transformer blocks to analyze: {self.transformer_blocks}")
        
        # Initialize variables to track key detection
        found_weights = False
        working_pattern_template = None
        matched_keys_count = 0
        
        # Try more patterns with more flexibility
        possible_patterns = [
            "transformer.transformer_blocks.{}.",
            "transformer_blocks.{}.",
            "blocks.{}.",
            "layers.{}."
        ]
        
        # Try each pattern with each block index until we find one that works
        for pattern in possible_patterns:
            for block_idx in self.transformer_blocks[:3]:  # Try first few blocks
                prefix = pattern.format(block_idx)
                matching_keys = [k for k in all_keys if k.startswith(prefix) and ('weight' in k or 'bias' in k)]
                
                if matching_keys:
                    working_pattern_template = pattern
                    matched_keys_count = len(matching_keys)
                    logger.info(f"Found working pattern: '{pattern}' with {len(matching_keys)} weight/bias keys")
                    logger.info(f"Example matching keys: {matching_keys[:3]}")
                    found_weights = True
                    break
            
            if found_weights:
                break
        
        # If we fail to find a pattern, fall back to position-based importance
        if not found_weights:
            logger.error("Failed to find any working pattern for transformer blocks.")
            
            # Create default scores based on block position
            num_blocks = len(self.transformer_blocks)
            position_scores = []
            
            for i, block_idx in enumerate(self.transformer_blocks):
                # Calculate a score that favors first and last few layers
                if i < 2:  # First two layers
                    score = 1.0 - (i * 0.05)
                elif i >= num_blocks - 2:  # Last two layers
                    score = 0.9 - ((num_blocks - i - 1) * 0.05)
                else:  # Middle layers get scores between 0.5-0.8
                    relative_pos = i / (num_blocks - 1)  # 0 to 1
                    # U-shaped curve for middle layers
                    middle_score = 0.5 + 0.3 * (1 - 4 * (relative_pos - 0.5)**2)
                    score = middle_score
                
                position_scores.append((block_idx, float(score)))
                logger.info("Block {}: FALLBACK position-based score = {:.4f}".format(block_idx, score))
            
            # Sort by score (highest first)
            self.block_scores = sorted(position_scores, key=lambda x: x[1], reverse=True)
            
            # Save these fallback scores
            try:
                with open(os.path.join(self.output_dir, "fallback_position_scores.json"), "w") as f:
                    json.dump({
                        "block_scores": self.block_scores,
                        "num_blocks": num_blocks,
                        "note": "Fallback position-based scores used because weight analysis failed"
                    }, f, indent=2)
            except Exception as e:
                logger.error(f"Failed to save fallback scores: {e}")
            
            # Also save in the standard block_scores format
            try:
                with open(os.path.join(self.output_dir, "block_scores.json"), "w") as f:
                    json.dump({
                        "block_scores": [(int(idx), float(score)) for idx, score in self.block_scores],
                        "num_blocks": num_blocks,
                        "is_fallback": True
                    }, f, indent=2)
            except Exception as e:
                logger.error(f"Failed to save block scores: {e}")
            
            return self.block_scores
        
        # If we get here, we found a working pattern, so proceed with weight analysis
        logger.info(f"Using pattern template '{working_pattern_template}' for weight analysis")
        
        # Now analyze each transformer block using the detected pattern
        for block_idx in self.transformer_blocks:
            pattern_to_use = working_pattern_template.format(block_idx)
            block_weight_keys = [k for k in all_keys if k.startswith(pattern_to_use) and ('weight' in k or 'bias' in k)]
            
            if not block_weight_keys:
                logger.warning("No weight/bias tensors found for block {}".format(block_idx))
                continue
                
            logger.debug("Found {} weight/bias keys for block {}".format(len(block_weight_keys), block_idx))
            
            for key in block_weight_keys:
                tensor = self.state_dict[key]
                
                # Skip empty or very small tensors
                if tensor.numel() < 10:
                    continue
                    
                # Flatten tensor for analysis
                flat_weights = tensor.view(-1).detach().cpu().numpy()
                
                # Skip if all zeros or nearly constant values
                if np.allclose(flat_weights, flat_weights[0], rtol=1e-5, atol=1e-8):
                    continue
                
                # 1. Calculate weight diversity (entropy) - Signal Component
                try:
                    hist, _ = np.histogram(flat_weights, bins=50, density=True)
                    # Ensure hist sums to 1, handle empty bins
                    hist_sum = np.sum(hist)
                    if hist_sum > 1e-6:  # Only calculate if histogram has content
                        hist = hist / hist_sum
                        non_zero_hist = hist[hist > 0]
                        if len(non_zero_hist) > 0:
                            entropy = -np.sum(non_zero_hist * np.log2(non_zero_hist + 1e-10))
                            weight_diversity[block_idx].append(entropy)
                except Exception as e:
                    logger.warning("Error calculating entropy for block {}, key {}: {}".format(block_idx, key, e))
                
                # 2. Calculate weight magnitude - Signal Strength
                try:
                    magnitude = np.mean(np.abs(flat_weights))
                    if magnitude > 0:
                        weight_magnitudes[block_idx].append(magnitude)
                except Exception as e:
                    logger.warning("Error calculating magnitude for block {}, key {}: {}".format(block_idx, key, e))
                
                # 3. Calculate noise estimate (standard deviation)
                try:
                    noise = np.std(flat_weights)
                    if noise > 0:
                        noise_estimates[block_idx].append(noise)
                except Exception as e:
                    logger.warning("Error calculating noise for block {}, key {}: {}".format(block_idx, key, e))
                
                # Count parameters
                param_counts[block_idx] += tensor.numel()
        
        # Check if we have any valid metrics
        if not any(weight_diversity) or not any(weight_magnitudes) or not any(noise_estimates):
            logger.error("No valid metrics calculated. Using position-based fallback.")
            # Use same position-based fallback as above
            num_blocks = len(self.transformer_blocks)
            position_scores = []
            
            for i, block_idx in enumerate(self.transformer_blocks):
                if i < 2:  # First two layers
                    score = 1.0 - (i * 0.05)
                elif i >= num_blocks - 2:  # Last two layers
                    score = 0.9 - ((num_blocks - i - 1) * 0.05)
                else:  # Middle layers
                    relative_pos = i / (num_blocks - 1)
                    middle_score = 0.5 + 0.3 * (1 - 4 * (relative_pos - 0.5)**2)
                    score = middle_score
                
                position_scores.append((block_idx, float(score)))
            
            self.block_scores = sorted(position_scores, key=lambda x: x[1], reverse=True)
            return self.block_scores
        
        # Calculate max values safely
        diversity_values = [np.mean(d) for d in weight_diversity.values() if d]
        magnitude_values = [np.mean(m) for m in weight_magnitudes.values() if m]
        noise_values = [np.mean(n) for n in noise_estimates.values() if n]
        
        max_diversity = max(diversity_values) if diversity_values else 1.0
        max_magnitude = max(magnitude_values) if magnitude_values else 1.0
        max_noise = max(noise_values) if noise_values else 1.0
        
        # Calculate SNR-based metrics for each block
        block_scores = []
        
        for block_idx in self.transformer_blocks:
            # Default score components
            avg_diversity = 0.0
            avg_magnitude = 0.0
            avg_noise = 0.0
            norm_diversity = 0.0
            norm_magnitude = 0.0
            norm_noise = 0.0
            true_snr = 0.0
            
            # Calculate metrics if we have data
            if block_idx in weight_diversity and weight_diversity[block_idx]:
                avg_diversity = np.mean(weight_diversity[block_idx])
                norm_diversity = avg_diversity / max_diversity
            
            if block_idx in weight_magnitudes and weight_magnitudes[block_idx]:
                avg_magnitude = np.mean(weight_magnitudes[block_idx])
                norm_magnitude = avg_magnitude / max_magnitude
            
            if block_idx in noise_estimates and noise_estimates[block_idx]:
                avg_noise = np.mean(noise_estimates[block_idx])
                norm_noise = avg_noise / max_noise
                # Calculate SNR if we have both signal components and noise
                if norm_diversity > 0 and norm_magnitude > 0:
                    signal = norm_diversity * norm_magnitude
                    true_snr = signal / (norm_noise + 1e-10)  # Avoid div by zero
            
            # Calculate importance score
            # If we have all components, use the full SNR formula
            if norm_diversity > 0 and norm_magnitude > 0 and norm_noise > 0:
                importance = 0.4 * norm_diversity + 0.3 * norm_magnitude + 0.3 * true_snr
            # Otherwise fall back to just using the components we have
            elif norm_diversity > 0 or norm_magnitude > 0:
                importance = 0.6 * norm_diversity + 0.4 * norm_magnitude
            else:
                # If no valid metrics for this block, give it a baseline score based on position
                pos = self.transformer_blocks.index(block_idx)
                num_blocks = len(self.transformer_blocks)
                if pos < 2 or pos >= num_blocks - 2:  # First or last two blocks
                    importance = 0.5  # Important enough to keep
                else:
                    importance = 0.3  # Less important middle block
            
            # Add the score
            block_scores.append((block_idx, float(importance)))
            
            # Log metrics - FIX: use string format method instead of f-strings
            metrics_str = "Block {}: diversity={:.4f}, magnitude={:.6f}, noise={:.6f}, SNR={:.4f}, score={:.4f}, params={:,}".format(
                block_idx, 
                avg_diversity,
                avg_magnitude, 
                avg_noise,
                true_snr,
                importance,
                param_counts.get(block_idx, 0)
            )
            
            logger.info(metrics_str)
        
        # Sort by importance score (highest first)
        self.block_scores = sorted(block_scores, key=lambda x: x[1], reverse=True)
        
        # Prepare detailed metrics for saving
        detailed_metrics = {}
        for block_idx in self.transformer_blocks:
            metrics = {
                "diversity": float(np.mean(weight_diversity[block_idx])) if block_idx in weight_diversity and weight_diversity[block_idx] else 0.0,
                "magnitude": float(np.mean(weight_magnitudes[block_idx])) if block_idx in weight_magnitudes and weight_magnitudes[block_idx] else 0.0,
                "noise": float(np.mean(noise_estimates[block_idx])) if block_idx in noise_estimates and noise_estimates[block_idx] else 0.0,
                "param_count": int(param_counts[block_idx]) if block_idx in param_counts else 0,
                "importance_score": float(dict(self.block_scores)[block_idx]) if block_idx in dict(self.block_scores) else 0.0
            }
            detailed_metrics[str(block_idx)] = metrics
        
        # Save detailed metrics
        metrics_data = {
            "block_scores": self.block_scores,
            "num_blocks": len(self.transformer_blocks),
            "pattern_used": working_pattern_template,
            "detailed_metrics": detailed_metrics
        }
        
        try:
            with open(os.path.join(self.output_dir, "snr_layer_metrics.json"), "w") as f:
                json.dump(metrics_data, f, indent=2)
            logger.info("Detailed SNR metrics saved")
        except Exception as e:
            logger.error("Failed to save detailed metrics: {}".format(e))
        
        # Also save the standard block_scores format
        try:
            with open(os.path.join(self.output_dir, "block_scores.json"), "w") as f:
                json.dump({
                    "block_scores": [(int(idx), float(score)) for idx, score in self.block_scores],
                    "num_blocks": len(self.transformer_blocks)
                }, f, indent=2)
            logger.info("Block scores saved")
        except Exception as e:
            logger.error("Failed to save block scores: {}".format(e))
        
        return self.block_scores
    
    def select_blocks_to_keep(self):
        """Select which transformer_blocks to keep based on importance scores."""
        if not hasattr(self, 'block_scores') or not self.block_scores:
            logger.info("Importance scores not calculated or empty, attempting calculation now.")
            self.analyze_layer_importance()
            if not hasattr(self, 'block_scores') or not self.block_scores:
                 logger.error("Cannot select blocks: Importance scores are missing.")
                 # Fallback: keep first N blocks if importance failed
                 if self.target_layers > len(self.transformer_blocks):
                       logger.warning(f"Target layers ({self.target_layers}) > available blocks ({len(self.transformer_blocks)}). Keeping all available blocks.")
                       self.blocks_to_keep = self.transformer_blocks
                 else:
                       logger.warning("Falling back to keeping the first {} transformer blocks.".format(self.target_layers))
                       self.blocks_to_keep = sorted(self.transformer_blocks[:self.target_layers])
                 return self.blocks_to_keep


        if self.target_layers >= len(self.transformer_blocks):
             logger.warning(f"Target layers ({self.target_layers}) >= number of available blocks ({len(self.transformer_blocks)}). Keeping all blocks.")
             self.blocks_to_keep = sorted(self.transformer_blocks)
        elif self.target_layers <= 0:
             logger.warning(f"Target layers ({self.target_layers}) <= 0. Keeping no transformer blocks.")
             self.blocks_to_keep = []
        elif self.target_layers <= 4:
             logger.warning(f"Target layers ({self.target_layers}) <= 4. Keeping first {self.target_layers} blocks for stability.")
             # Keep first N based on original index
             self.blocks_to_keep = sorted(self.transformer_blocks[:self.target_layers])
        else:
             # Strategy: Always keep first 2 and last 2 blocks, select best from middle
             if len(self.transformer_blocks) < 4:
                   logger.warning(f"Model has less than 4 layers ({len(self.transformer_blocks)}), cannot enforce keeping first/last 2.")
                   # Keep the top N scored blocks among the available ones
                   top_scored_indices = [idx for idx, score in self.block_scores[:self.target_layers]]
                   self.blocks_to_keep = sorted(top_scored_indices)
             else:
                must_keep = set()
                must_keep.add(self.transformer_blocks[0])
                must_keep.add(self.transformer_blocks[1])
                must_keep.add(self.transformer_blocks[-2])
                must_keep.add(self.transformer_blocks[-1])
                must_keep = sorted(list(must_keep))

                # Filter out must_keep blocks from scores
                middle_blocks_scored = [(idx, score) for idx, score in self.block_scores
                                        if idx not in must_keep]

                # Calculate how many middle blocks to keep
                middle_to_keep_count = self.target_layers - len(must_keep)
                if middle_to_keep_count < 0:
                    logger.warning(f"Target layers ({self.target_layers}) less than minimum required ({len(must_keep)}) based on first/last 2 rule. Selecting from must_keep list.")
                    # Select the highest scored ones from the must_keep list itself? Or just first N? Let's take first N for simplicity.
                    self.blocks_to_keep = must_keep[:self.target_layers]
                else:
                    # Select top-scoring middle blocks
                    selected_middle = [idx for idx, score in middle_blocks_scored[:middle_to_keep_count]]
                    self.blocks_to_keep = sorted(must_keep + selected_middle)

        logger.info(f"Selected transformer_blocks indices to keep: {self.blocks_to_keep}")

        # Save selected blocks
        keep_path = os.path.join(self.output_dir, "blocks_to_keep.json")
        try:
            with open(keep_path, "w") as f:
                 json.dump({
                    "blocks_to_keep": self.blocks_to_keep,
                    "original_num_transformer_blocks": len(self.transformer_blocks),
                     "original_transformer_block_indices": self.transformer_blocks,
                    "target_layers": self.target_layers
                 }, f, indent=2)
            logger.info(f"Selected blocks information saved to {keep_path}")
        except Exception as e:
             logger.error(f"Failed to save selected blocks info to {keep_path}: {e}")

        return self.blocks_to_keep


    def prune_checkpoint(self):
        """Create a pruned version of the checkpoint with proper sequential indexing."""
        if not hasattr(self, 'blocks_to_keep'):
             logger.info("Blocks to keep not selected yet. Running selection process.")
             self.select_blocks_to_keep()

        logger.info(f"Pruning checkpoint to keep transformer_blocks: {self.blocks_to_keep}")
        logger.info(f"Keeping all {len(self.text_blocks)} text_blocks: {self.text_blocks}")

        # Create block mapping from old indices to sequential new indices for transformers
        block_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(self.blocks_to_keep)}
        logger.info(f"Transformer block mapping (old_index -> new_index): {block_mapping}")

        # Create a completely new state dict with sequential indices
        new_state_dict = {}
        param_count_before = sum(p.numel() for p in self.state_dict.values())
        params_kept = 0
        params_discarded = 0

# --- Steps 1, 2, 3 Combined: Construct Pruned State Dict ---
        # Iterate through the *cleaned* original state_dict (no prefixes, no metadata)
        # and build the new one containing only necessary keys with remapped indices.

        new_state_dict = {}
        # Use the confirmed pattern for transformer blocks
        transformer_regex = re.compile(r"transformer\.transformer_blocks\.(\d+)\.")
        # Use the confirmed pattern for text blocks (assuming they are kept)
        text_regex = re.compile(r"transformer\.text_embed\.text_blocks\.(\d+)\.")

        param_count_before = sum(p.numel() for p in self.state_dict.values())
        params_kept_count = 0
        params_discarded_count = 0
        kept_keys_set = set()
        discarded_keys_set = set()

        logger.info("Constructing pruned state dict by filtering and remapping...")
        for key, tensor in self.state_dict.items():
            match_transformer = transformer_regex.match(key)
            match_text = text_regex.match(key) # Check for text blocks too

            if match_transformer: # It's a transformer block key
                 block_idx = int(match_transformer.group(1))
                 if block_idx in block_mapping: # It's a block we want to KEEP
                      new_idx = block_mapping[block_idx]
                      # Create the new key with the sequential index
                      new_key = key.replace(f"transformer.transformer_blocks.{block_idx}.", f"transformer.transformer_blocks.{new_idx}.")
                      new_state_dict[new_key] = tensor
                      params_kept_count += tensor.numel()
                      kept_keys_set.add(new_key)
                 else: # It's a transformer block we want to DISCARD
                      params_discarded_count += tensor.numel()
                      discarded_keys_set.add(key)
                      continue # Skip this key explicitly

            elif match_text: # It's a text block key (assume keep all for now)
                 # Keep text blocks with original index, no remapping needed typically
                 new_state_dict[key] = tensor
                 params_kept_count += tensor.numel()
                 kept_keys_set.add(key)

            else: # It's NOT a transformer or text block key (e.g., embed, norm_out)
                 new_state_dict[key] = tensor # Keep it as is
                 params_kept_count += tensor.numel()
                 kept_keys_set.add(key)

        logger.info(f"Finished constructing pruned state dict.")
        logger.info(f"Original state_dict parameters (approx): {param_count_before:,}")
        logger.info(f"Kept parameters (calculated): {params_kept_count:,} ({len(kept_keys_set)} keys)")
        logger.info(f"Discarded parameters (calculated): {params_discarded_count:,} ({len(discarded_keys_set)} keys)")
        final_pruned_params = sum(p.numel() for p in new_state_dict.values())
        logger.info(f"Final pruned state_dict parameters: {final_pruned_params:,} ({len(new_state_dict)} keys)")
        if final_pruned_params != params_kept_count:
            logger.warning("Mismatch between calculated kept parameters and final sum!")

        # 4. Create the pruned checkpoint dictionary structure
        # We need to reconstruct the structure expected by torch.save
        pruned_checkpoint = {}

        # Copy metadata from original checkpoint if it was loaded (.pt input)
        if self.full_checkpoint_data:
             for key, value in self.full_checkpoint_data.items():
                  # Don't copy the original state dict itself
                  if key != self.state_dict_key:
                     pruned_checkpoint[key] = value
        # If input was safetensors, full_checkpoint_data might be None or only contain companion metadata.
        # We need to ensure essential keys like 'config' are present if they were found.
        elif self.config is not None and 'config' not in pruned_checkpoint:
              pruned_checkpoint['config'] = self.config
        elif self.config is not None and 'hyper_parameters' not in pruned_checkpoint:
              # If companion .pt used 'hyper_parameters', preserve that structure
              companion_haskey = False
              if self.full_checkpoint_data and 'hyper_parameters' in self.full_checkpoint_data:
                   companion_haskey = True
              if companion_haskey:
                   pruned_checkpoint['hyper_parameters'] = self.config
              else: # Default to 'config' if unsure
                   pruned_checkpoint['config'] = self.config


        # 5. Put the pruned state_dict back into the checkpoint structure
        #    FORCE the key to 'model_state_dict' for compatibility.
        #    Use the state_dict WITH EMA PREFIX STRIPPED.

        # new_state_dict already contains the pruned weights with cleaned keys (no EMA prefix)
        final_state_dict_to_save = new_state_dict
        target_key_for_saving = 'model_state_dict' # The key required by the inference script

        logger.info(f"Preparing final checkpoint structure. Target key: '{target_key_for_saving}'")
        if self.is_ema:
            logger.info("Original model was EMA, pruned weights are cleaned (no prefix).")
        else:
            logger.info("Original model was not EMA.")


        # Start building the pruned_checkpoint dictionary
        final_pruned_checkpoint = {}
        config_updated = False # Initialize here before potentially updating

        # Add other metadata first (config, pruning_info, etc.)
        # Try copying from original full_checkpoint_data if it exists (.pt input or companion)
        if self.full_checkpoint_data:
             for key, value in self.full_checkpoint_data.items():
                  # Don't copy the original state dict key (e.g., 'ema_model_state_dict', 'state_dict')
                  # Also avoid copying keys we will explicitly set (like config or the target state dict key)
                  if key not in [self.state_dict_key, target_key_for_saving, 'config', 'hyper_parameters']:
                     final_pruned_checkpoint[key] = value
             logger.info(f"Copied {len(final_pruned_checkpoint)} non-state/config keys from original/companion checkpoint.")

        # Add config if it was loaded, ensuring correct key ('config' or 'hyper_parameters')
        if self.config is not None:
            config_key_to_use = 'config' # Default
            # Check if original used hyper_parameters instead
            if self.full_checkpoint_data and 'hyper_parameters' in self.full_checkpoint_data and 'config' not in self.full_checkpoint_data:
                 config_key_to_use = 'hyper_parameters'
            final_pruned_checkpoint[config_key_to_use] = self.config # Add the potentially updated config
            logger.info(f"Added config under key '{config_key_to_use}'.")
            # Update depth within the added config (moved logic here for clarity)
            # config_updated = False # <<< MOVED Initialization Before This Block
            if config_key_to_use in final_pruned_checkpoint and isinstance(final_pruned_checkpoint[config_key_to_use], dict):
                logger.info(f"Attempting to update depth in config under key '{config_key_to_use}'")
                current_level = final_pruned_checkpoint[config_key_to_use]
                possible_paths = [ # Copied from Step 6 before
                      ['model', 'arch', 'depth'], ['arch', 'depth'], ['model', 'depth'], ['depth'],
                      ['model', 'arch', 'n_layer'], ['arch', 'n_layer'], ['n_layer'],
                      ['model','arch','num_hidden_layers'], ['num_hidden_layers'] ]
                updated = False
                for path in possible_paths:
                     temp_level = current_level; valid_path = True
                     for i, key in enumerate(path):
                          if isinstance(temp_level, dict) and key in temp_level:
                               if i == len(path) - 1:
                                    logger.info(f"Found depth key at path: {config_key_to_use}.{' -> '.join(path)}. Updating value.")
                                    temp_level[key] = len(self.blocks_to_keep)
                                    config_updated = True; updated = True; break # Set config_updated to True here
                               else: temp_level = temp_level[key]
                          else: valid_path = False; break
                     if updated: break
                if not updated: logger.warning(f"Could not find a known depth key path within '{config_key_to_use}' to update.")
            else: logger.warning("Config dictionary not found or not a dict in the final checkpoint. Cannot update depth.")
        else:
            logger.info("Config was not loaded (e.g., from safetensors without companion .pt). Cannot update depth.")


        # --- FORCE THE STATE DICT KEY ---
        # Add the pruned state dictionary under the specific target key
        final_pruned_checkpoint[target_key_for_saving] = final_state_dict_to_save
        logger.info(f"Added pruned state_dict (EMA prefix removed if applicable) under FORCED key: '{target_key_for_saving}'")


        # Add pruning info (must happen after the main structure is built)
        # Now config_updated will always have a value (False if not updated, True if updated)
        final_pruned_checkpoint['pruning_info'] = {
             'original_model_path': self.model_path,
             'pruning_script_version': 'definitive-f5tts-pruner_v6_config_init', # Version bump
             'original_transformer_indices': self.transformer_blocks,
             'transformer_blocks_kept_indices': self.blocks_to_keep,
             'transformer_block_mapping_old_to_new': block_mapping,
             'text_blocks_kept_indices': self.text_blocks,
             'target_transformer_layers': self.target_layers,
             'final_transformer_layers': len(self.blocks_to_keep),
             'config_depth_updated': config_updated, # Use the initialized/updated variable
             'pruning_time': datetime.datetime.now().isoformat(),
             'original_state_dict_key': self.state_dict_key,
             'was_ema': self.is_ema
        }
        logger.info("Added pruning metadata to the checkpoint.")

        # Assign the fully constructed dict back to the variable used for saving
        pruned_checkpoint = final_pruned_checkpoint

        # 6. Update config if it exists (Logic moved into step 5 above)
        # 7. Add pruning metadata (Logic moved into step 5 above)

        # 8. Save the pruned checkpoint as a .pt file
        output_path = os.path.join(self.output_dir, self.output_name)
        try:
            # --- Add Debug log right before saving ---
            if isinstance(pruned_checkpoint, dict):
                logger.info(f"DEBUG: Keys in checkpoint right before saving: {list(pruned_checkpoint.keys())}")
            else:
                 logger.error("DEBUG: pruned_checkpoint is NOT a dict before saving!")
            # ---
            torch.save(pruned_checkpoint, output_path)
            logger.info(f"Pruned model saved successfully to: {output_path}")
        except Exception as e:
             logger.error(f"Failed to save pruned checkpoint to {output_path}: {e}", exc_info=True)
             raise

        # 9. Calculate size reduction (remains the same)
        # [...]
        # 6. Update config if it exists
        config_updated = False
        if isinstance(pruned_checkpoint, dict): # Ensure pruned_checkpoint is a dict before accessing keys
            config_key_found = None
            if 'config' in pruned_checkpoint and isinstance(pruned_checkpoint['config'], dict):
                 config_key_found = 'config'
            elif 'hyper_parameters' in pruned_checkpoint and isinstance(pruned_checkpoint['hyper_parameters'], dict):
                 config_key_found = 'hyper_parameters'

            if config_key_found:
                 logger.info(f"Attempting to update depth in config under key '{config_key_found}'")
                 # Navigate potentially nested structure
                 current_level = pruned_checkpoint[config_key_found]
                 path_to_depth = []
                 updated = False

                 # Common paths to check based on F5TTS YAML and general practices
                 possible_paths = [
                      ['model', 'arch', 'depth'],
                      ['arch', 'depth'],
                      ['model', 'depth'],
                      ['depth'],
                      ['model', 'arch', 'n_layer'],
                      ['arch', 'n_layer'],
                      ['n_layer'],
                      ['model','arch','num_hidden_layers'],
                      ['num_hidden_layers']
                 ]

                 for path in possible_paths:
                      temp_level = current_level
                      valid_path = True
                      for i, key in enumerate(path):
                           if isinstance(temp_level, dict) and key in temp_level:
                                if i == len(path) - 1: # Reached the depth key
                                     logger.info(f"Found depth key at path: {config_key_found}.{' -> '.join(path)}. Updating value.")
                                     temp_level[key] = len(self.blocks_to_keep)
                                     config_updated = True
                                     updated = True
                                     break
                                else: # Navigate deeper
                                     temp_level = temp_level[key]
                           else:
                                valid_path = False
                                break # Invalid path
                      if updated: break # Stop searching if updated

                 if not updated:
                       logger.warning(f"Could not find a known depth key path within '{config_key_found}' to update.")
            else:
                logger.warning("Config dictionary not found in the final checkpoint structure. Cannot update depth.")
        else:
             logger.warning("Pruned checkpoint is not a dictionary. Cannot update config or add pruning info.")


        # 7. Add pruning metadata (only if checkpoint is a dict)
        if isinstance(pruned_checkpoint, dict):
             pruned_checkpoint['pruning_info'] = {
                'original_model_path': self.model_path,
                'pruning_script_version': 'definitive-f5tts-pruner_v2_safetensors', # Identify script version
                'original_transformer_indices': self.transformer_blocks,
                'transformer_blocks_kept_indices': self.blocks_to_keep,
                'transformer_block_mapping_old_to_new': block_mapping,
                'text_blocks_kept_indices': self.text_blocks, # Assuming text blocks are always kept fully
                'target_transformer_layers': self.target_layers,
                'final_transformer_layers': len(self.blocks_to_keep),
                 'config_depth_updated': config_updated,
                'pruning_time': datetime.datetime.now().isoformat()
             }
             logger.info("Added pruning metadata to the checkpoint.")

        # 8. Save the pruned checkpoint as a .pt file
        output_path = os.path.join(self.output_dir, self.output_name)
        try:
            torch.save(pruned_checkpoint, output_path)
            logger.info(f"Pruned model saved successfully to: {output_path}")
        except Exception as e:
             logger.error(f"Failed to save pruned checkpoint to {output_path}: {e}", exc_info=True)
             raise

        # 9. Calculate size reduction
        try:
            original_size_bytes = os.path.getsize(self.model_path)
            pruned_size_bytes = os.path.getsize(output_path)
            original_size_mb = original_size_bytes / (1024 * 1024)
            pruned_size_mb = pruned_size_bytes / (1024 * 1024)
            if original_size_bytes > 0:
                 reduction_percent = (1 - pruned_size_bytes / original_size_bytes) * 100
                 logger.info(f"Original size: {original_size_mb:.2f} MB")
                 logger.info(f"Pruned size:   {pruned_size_mb:.2f} MB")
                 logger.info(f"Size reduction: {reduction_percent:.2f}%")
            else:
                 logger.warning("Original model size is 0 bytes. Cannot calculate reduction.")

        except FileNotFoundError:
             logger.error("Could not find original or pruned file to calculate size reduction.")
        except Exception as e:
             logger.error(f"Error calculating file sizes: {e}")

        return output_path

# --- Helper functions (set_blocks_and_prune, test_pruned_model) ---
# These functions seem okay and don't need major changes for safetensors input,
# as they operate on the Pruner object or the output file.
# Small adjustment in test_pruned_model to use the shared loading logic might be cleaner.

def set_blocks_and_prune(pruner, blocks_to_keep):
    """
    Set specific blocks to keep and run pruning.

    Args:
        pruner: An instance of the DefinitiveF5TTSPruner class
        blocks_to_keep: List of transformer_block indices to keep
    """
    logger.info(f"Manually setting transformer blocks to keep: {blocks_to_keep}")
    # Ensure they are sorted and unique integers from the available blocks
    available_blocks = set(pruner.transformer_blocks)
    valid_blocks_to_keep = sorted(list(set(int(b) for b in blocks_to_keep if int(b) in available_blocks)))
    if len(valid_blocks_to_keep) != len(blocks_to_keep):
         logger.warning(f"Some manually specified blocks were invalid or duplicates. Using valid blocks: {valid_blocks_to_keep}")

    if not valid_blocks_to_keep and blocks_to_keep:
         raise ValueError("No valid blocks specified in manual list.")

    pruner.blocks_to_keep = valid_blocks_to_keep
    pruner.target_layers = len(valid_blocks_to_keep) # Update target to reflect manual choice

    # Prune the checkpoint
    return pruner.prune_checkpoint()


def test_pruned_model(model_path, device="cpu"):
    """
    Test if a pruned model (always .pt format output) has the correct structure.

    Args:
        model_path: Path to the pruned model (.pt file)
        device: Device to use (primarily for loading checks)
    """
    print(f"\n--- Testing Pruned Model Structure: {model_path} ---")

    if not model_path.endswith(".pt"):
        print(f"ERROR: Test function expects a .pt file, got {model_path}")
        return False

    if not os.path.exists(model_path):
        print(f"ERROR: Pruned model file not found at {model_path}")
        return False

    try:
        # Use the same loading logic for consistency in finding state_dict and config
        full_checkpoint_data, state_dict, config, state_dict_key, is_ema = load_checkpoint_data(model_path)
    except Exception as e:
        print(f"ERROR loading pruned model for testing: {e}")
        return False

    if state_dict is None:
        print("ERROR: Could not locate state dict in the pruned model file.")
        return False

    # Count blocks in the loaded state_dict
    transformer_blocks_found = set()
    text_blocks_found = set()

    # Use the state_dict directly (it might have prefixes removed already by load_checkpoint_data)
    current_state_dict = state_dict
    # Handle case where EMA prefix might still be there if loading failed slightly differently
    # (though load_checkpoint_data should handle it)
    if is_ema and all(k.startswith("ema_model.") for k in current_state_dict.keys() if '.' in k):
         logger.info("Testing: Removing ema_model prefix for block analysis")
         current_state_dict = {k.replace("ema_model.", ""): v for k, v in current_state_dict.items()}


    for key in current_state_dict.keys():
        match_transformer = re.match(r"transformer_blocks\.(\d+)\.", key)
        if match_transformer:
            transformer_blocks_found.add(int(match_transformer.group(1)))

        match_text = re.match(r"text_blocks\.(\d+)\.", key)
        if match_text:
            text_blocks_found.add(int(match_text.group(1)))

    transformer_blocks_sorted = sorted(list(transformer_blocks_found))
    text_blocks_sorted = sorted(list(text_blocks_found))

    print(f"Found transformer_blocks indices: {transformer_blocks_sorted}")
    print(f"Found text_blocks indices: {text_blocks_sorted}")

    # Check if transformer blocks are sequential from 0
    num_transformer_layers = len(transformer_blocks_sorted)
    expected_transformer_blocks = list(range(num_transformer_layers))
    is_sequential = (transformer_blocks_sorted == expected_transformer_blocks)

    if is_sequential:
        print(f"✓ OK: Transformer blocks are sequential from 0 to {num_transformer_layers - 1}.")
    else:
        print(f"✗ ERROR: Transformer blocks are not sequential! Expected {expected_transformer_blocks}, found {transformer_blocks_sorted}")

    # Check config depth consistency
    config_depth = None
    config_depth_matches = False
    if config and isinstance(config, dict):
         # Check common depth keys again
         arch_config = config
         if 'model' in arch_config and isinstance(arch_config['model'], dict): arch_config = arch_config['model']
         if 'arch' in arch_config and isinstance(arch_config['arch'], dict): arch_config = arch_config['arch']

         if 'depth' in arch_config: config_depth = arch_config['depth']
         elif 'n_layer' in arch_config: config_depth = arch_config['n_layer']
         elif 'num_hidden_layers' in arch_config: config_depth = arch_config['num_hidden_layers']

         if config_depth is not None:
             print(f"Config depth found: {config_depth}")
             if config_depth == num_transformer_layers:
                 print(f"✓ OK: Config depth ({config_depth}) matches the number of transformer blocks found ({num_transformer_layers}).")
                 config_depth_matches = True
             else:
                 print(f"✗ WARNING: Config depth ({config_depth}) does NOT match the number of transformer blocks found ({num_transformer_layers}).")
         else:
             print("INFO: Depth key not found in loaded config for verification.")
    else:
        print("INFO: Config not found in the checkpoint for verification.")

    # Check pruning info
    pruning_info_valid = False
    if full_checkpoint_data and 'pruning_info' in full_checkpoint_data:
         pruning_info = full_checkpoint_data['pruning_info']
         print(f"Pruning info found:")
         print(json.dumps(pruning_info, indent=2))
         # Verify consistency within pruning_info
         if 'final_transformer_layers' in pruning_info and pruning_info['final_transformer_layers'] == num_transformer_layers:
             print("✓ OK: Pruning info 'final_transformer_layers' matches found blocks.")
             pruning_info_valid = True
         else:
             print("✗ WARNING: Pruning info 'final_transformer_layers' mismatch or missing.")
         if 'config_depth_updated' in pruning_info and pruning_info['config_depth_updated'] and not config_depth_matches:
              print("✗ WARNING: Pruning info claims config depth was updated, but test shows mismatch or config missing.")

    else:
        print("INFO: Pruning info not found in the checkpoint.")

    print("--- Test Summary ---")
    final_ok = True
    if not is_sequential:
         print("✗ FAILED: Transformer blocks are not sequential.")
         final_ok = False
    else:
         print("✓ PASSED: Transformer blocks sequentiality.")

    if config_depth is not None and not config_depth_matches:
         print("✗ WARNING: Config depth does not match found blocks.")
         # Not marking as fail, as config might be optional/unreliable
    elif config_depth is not None and config_depth_matches:
         print("✓ PASSED: Config depth consistency.")

    if not pruning_info_valid and ('pruning_info' in full_checkpoint_data):
         print("✗ WARNING: Pruning info seems inconsistent.")
    elif pruning_info_valid:
         print("✓ PASSED: Pruning info consistency.")

    print("--------------------")
    return final_ok # Return True only if sequentiality passed


def main():
    """Main function to prune a F5-TTS model."""
    parser = argparse.ArgumentParser(
        description="Definitive F5-TTS model pruner. Handles .pt and .safetensors inputs, outputs .pt.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter # Show defaults
    )
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the input model checkpoint (.pt or .safetensors)")
    parser.add_argument("--output_dir", type=str, default="pruned_models",
                        help="Directory to save the pruned model")
    parser.add_argument("--output_name", type=str, default="pruned_model.pt",
                        help="Name of the output file (will ensure .pt extension)")
    parser.add_argument("--target_layers", type=int, default=14,
                        help="Number of transformer layers to keep in the pruned model")
    parser.add_argument("--manual_blocks", type=str, default=None,
                        help="Comma-separated list of original transformer block indices to keep (e.g., '0,1,5,10,20,21'). Overrides automatic importance-based selection.")
    parser.add_argument("--test_only", action="store_true",
                        help="Only load and test the structure of the model specified by --model_path (expects .pt output format)")
    parser.add_argument("--device", type=str, default="cpu", # Default to CPU as most ops are there
                        help="Device preference ('cuda' or 'cpu'). Mainly affects loading map_location if needed, but script prefers CPU.")

    args = parser.parse_args()

    if args.test_only:
        logger.info(f"--- Running in Test-Only Mode for: {args.model_path} ---")
        test_pruned_model(args.model_path, args.device)
        logger.info("--- Test-Only Mode Finished ---")
    else:
        logger.info(f"--- Running Pruning Process ---")
        logger.info(f"Input Model: {args.model_path}")
        logger.info(f"Output Dir: {args.output_dir}")
        logger.info(f"Output Name: {args.output_name}")
        logger.info(f"Target Transformer Layers: {args.target_layers}")

        if not os.path.exists(args.model_path):
             logger.error(f"Input model file not found: {args.model_path}")
             return

        try:
             # Create pruner instance (loads and analyzes)
             pruner = DefinitiveF5TTSPruner(
                 model_path=args.model_path,
                 output_dir=args.output_dir,
                 output_name=args.output_name,
                 target_layers=args.target_layers,
                 device=args.device
             )

             pruned_path = None
             if args.manual_blocks:
                 logger.info(f"Manual block selection requested: {args.manual_blocks}")
                 # Manual block selection
                 try:
                     blocks_to_keep = [int(b.strip()) for b in args.manual_blocks.split(",")]
                     pruned_path = set_blocks_and_prune(pruner, blocks_to_keep)
                 except ValueError as e:
                      logger.error(f"Invalid format for --manual_blocks: {e}. Must be comma-separated integers.")
                      return
                 except Exception as e:
                      logger.error(f"Error during manual pruning: {e}", exc_info=True)
                      return
             else:
                 # Automatic block selection based on importance
                 logger.info("Automatic block selection based on layer importance.")
                 try:
                     pruner.analyze_layer_importance() # Calculate scores
                     pruner.select_blocks_to_keep()    # Select based on scores and target
                     pruned_path = pruner.prune_checkpoint() # Perform pruning and saving
                 except Exception as e:
                      logger.error(f"Error during automatic pruning: {e}", exc_info=True)
                      return

             if pruned_path and os.path.exists(pruned_path):
                  logger.info(f"\nPruning seemingly complete!")
                  logger.info(f"Pruned model saved to: {pruned_path}")
                  # Test the pruned model automatically
                  logger.info("\n--- Automatically Testing Pruned Model Structure ---")
                  test_passed = test_pruned_model(pruned_path, args.device)
                  if test_passed:
                       logger.info("--- Pruned Model Test Passed ---")
                  else:
                       logger.error("--- Pruned Model Test Failed ---")
             else:
                  logger.error("Pruning process finished, but the output file path was not generated or file doesn't exist.")

        except Exception as e:
             logger.error(f"An error occurred during the pruning process: {e}", exc_info=True)

        logger.info(f"--- Pruning Process Finished ---")


if __name__ == "__main__":
    main()

# Example command lines:
# Prune using automatic selection:
# python definitive-f5tts-pruner.py --model_path model.safetensors --output_dir pruned --output_name auto_pruned_8L.pt --target_layers 8
# Prune using manual selection:
# python definitive-f5tts-pruner.py --model_path model.pt --output_dir pruned --output_name manual_pruned_5L.pt --manual_blocks "0,1,10,20,21"
# Test an already pruned file:
# python definitive-f5tts-pruner.py --test_only --model_path pruned/manual_pruned_5L.pt

# --- END OF MODIFIED FILE ---