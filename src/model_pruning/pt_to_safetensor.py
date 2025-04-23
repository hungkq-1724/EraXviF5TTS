import torch
import argparse
import os
import logging
from safetensors.torch import save_file
from collections import OrderedDict
import datetime

# --- Configuration ---
# Common keys where state_dict might be stored in .pt files (in order of preference)
STATE_DICT_KEYS = [
    'state_dict',           # PyTorch Lightning default
    'model_state_dict',     # Common convention
    'model',                # Sometimes used
    'ema_model_state_dict', # For EMA checkpoints (special handling below)
    'module',               # Sometimes used with DataParallel/DDP wrappers directly saved
]

# Common prefixes to strip from keys within the state_dict
PREFIXES_TO_STRIP = [
    "module.",              # From DataParallel/DDP wrappers
    "model.",               # Common PL prefix
    "_orig_mod.",           # From torch.compile with dynamic=True
    # EMA prefix handled separately if ema_key is found
]
EMA_KEY = 'ema_model_state_dict'
EMA_PREFIX = 'ema_model.'

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def find_and_clean_state_dict(checkpoint):
    """
    Attempts to find the state_dict within a loaded checkpoint and clean its keys.

    Args:
        checkpoint: The loaded data from torch.load().

    Returns:
        tuple: (cleaned_state_dict, info)
               - cleaned_state_dict: The state dict with prefixes stripped.
               - info: Dictionary containing metadata about the extraction process.
               Returns (None, info) if state_dict cannot be found.
    """
    if not isinstance(checkpoint, dict):
        # Assume the loaded object IS the state_dict itself
        logger.info("Input checkpoint is not a dictionary. Assuming it IS the state_dict.")
        state_dict = checkpoint
        original_key = "[root]"
        is_ema = False # Cannot determine EMA status easily from raw state dict
    else:
        state_dict = None
        original_key = None
        is_ema = False

        # Prioritize EMA key if present
        if EMA_KEY in checkpoint:
            state_dict = checkpoint[EMA_KEY]
            original_key = EMA_KEY
            is_ema = True
            logger.info(f"Found state_dict under EMA key: '{EMA_KEY}'.")
        else:
            # Search standard keys
            for key in STATE_DICT_KEYS:
                if key in checkpoint:
                    state_dict = checkpoint[key]
                    original_key = key
                    logger.info(f"Found state_dict under standard key: '{key}'.")
                    break

        if state_dict is None:
            # Fallback: Check if the root dictionary looks like a state_dict
            is_likely_state_dict = all(isinstance(v, torch.Tensor) for k, v in checkpoint.items() if not k.startswith('_'))
            if is_likely_state_dict and len(checkpoint) > 0:
                logger.warning("No standard key found, but root dictionary looks like a state_dict. Using root.")
                state_dict = checkpoint
                original_key = "[root]"
            else:
                logger.error(f"Could not find state_dict using keys {STATE_DICT_KEYS} or heuristics.")
                return None, {"original_key": None, "is_ema": False, "prefix_stripped": None}

    if not isinstance(state_dict, (dict, OrderedDict)):
        logger.error(f"Data found under key '{original_key}' is not a dictionary (type: {type(state_dict)}). Cannot process.")
        return None, {"original_key": original_key, "is_ema": is_ema, "prefix_stripped": None}

    cleaned_state_dict = {}
    prefix_stripped = None

    # --- EMA Prefix Stripping (if EMA key was found) ---
    if is_ema:
        num_ema_prefix = sum(1 for k in state_dict if k.startswith(EMA_PREFIX))
        # Strip if most keys have the prefix (allow for metadata like 'initted', 'step')
        if num_ema_prefix > 0 and num_ema_prefix >= 0.8 * len(state_dict):
            prefix_stripped = EMA_PREFIX
            logger.info(f"Stripping prefix '{EMA_PREFIX}' from EMA state_dict keys...")
            for k, v in state_dict.items():
                if k.startswith(EMA_PREFIX):
                    cleaned_state_dict[k[len(EMA_PREFIX):]] = v
                elif k not in ["initted", "step"]: # Keep special keys out only if they don't have prefix
                    logger.warning(f"Keeping EMA key '{k}' without expected prefix '{EMA_PREFIX}'.")
                    cleaned_state_dict[k] = v
                else:
                    logger.info(f"Ignoring EMA metadata key: '{k}'")
            state_dict = cleaned_state_dict # Use the cleaned version for further checks
            cleaned_state_dict = {} # Reset for next stripping phase
        else:
            logger.info(f"EMA key '{EMA_KEY}' found, but internal keys do not consistently start with '{EMA_PREFIX}'. Skipping EMA prefix strip.")
            # Need to filter out metadata manually if prefix wasn't stripped
            temp_dict = {}
            for k, v in state_dict.items():
                 if k not in ["initted", "step"]:
                      temp_dict[k] = v
                 else:
                      logger.info(f"Ignoring EMA metadata key: '{k}'")
            state_dict = temp_dict # Use filtered dict


    # --- Standard Prefix Stripping ---
    possible_prefix = None
    if not prefix_stripped and len(state_dict) > 0: # Only check if EMA prefix wasn't already stripped
        first_key = next(iter(state_dict.keys()))
        for prefix in PREFIXES_TO_STRIP:
            if first_key.startswith(prefix):
                # Check if ALL keys start with this prefix
                if all(k.startswith(prefix) for k in state_dict):
                    possible_prefix = prefix
                    break

    if possible_prefix:
        prefix_stripped = possible_prefix
        logger.info(f"Stripping prefix '{prefix_stripped}' from state_dict keys...")
        for k, v in state_dict.items():
            cleaned_state_dict[k[len(prefix_stripped):]] = v
    else:
        logger.info("No common standard prefix found or already stripped EMA prefix. Using keys as is.")
        cleaned_state_dict = state_dict # Use the (potentially EMA-filtered) dict

    # Final check for empty dict
    if not cleaned_state_dict:
         logger.error("Resulting state_dict is empty after processing. Check input file and logic.")
         return None, {"original_key": original_key, "is_ema": is_ema, "prefix_stripped": prefix_stripped}

    info = {
        "original_key": original_key,
        "is_ema": is_ema,
        "prefix_stripped": prefix_stripped,
        "num_tensors": len(cleaned_state_dict)
    }
    return cleaned_state_dict, info


def convert_pt_to_safetensors(pt_path, sf_path):
    """
    Loads a .pt file, extracts the state_dict, cleans keys, and saves as .safetensors.
    """
    if not os.path.exists(pt_path):
        logger.error(f"Input file not found: {pt_path}")
        return False

    logger.info(f"Loading PyTorch checkpoint: {pt_path}")
    try:
        # Load on CPU to avoid GPU memory issues during conversion
        # Explicitly set weights_only=False for now, add warning.
        logger.warning("Loading with `weights_only=False`. Be sure you trust the source of the .pt file.")
        checkpoint = torch.load(pt_path, map_location="cpu", weights_only=False) # Set to True if you are SURE it only contains weights
    except Exception as e:
        logger.error(f"Failed to load {pt_path}: {e}", exc_info=True)
        return False

    logger.info("Extracting and cleaning state_dict...")
    state_dict, extract_info = find_and_clean_state_dict(checkpoint)

    if state_dict is None:
        logger.error("Failed to extract state_dict from the checkpoint.")
        return False

    logger.info(f"Extracted {extract_info['num_tensors']} tensors.")
    if extract_info['original_key']: logger.info(f"  (Found under key: '{extract_info['original_key']}')")
    if extract_info['is_ema']: logger.info(f"  (Detected as EMA)")
    if extract_info['prefix_stripped']: logger.info(f"  (Stripped prefix: '{extract_info['prefix_stripped']}')")


    # Ensure output directory exists
    try:
        os.makedirs(os.path.dirname(sf_path), exist_ok=True)
    except OSError as e:
        logger.error(f"Failed to create output directory {os.path.dirname(sf_path)}: {e}")
        return False

    # Prepare metadata
    metadata = {
        "format": "pt", # Indicate original format
        "conversion_source": os.path.basename(pt_path),
        "conversion_timestamp": datetime.datetime.utcnow().isoformat() + "Z",
    }
    if extract_info['original_key']: metadata["original_key"] = str(extract_info['original_key'])
    if extract_info['is_ema']: metadata["original_was_ema"] = "true"
    if extract_info['prefix_stripped']: metadata["stripped_prefix"] = str(extract_info['prefix_stripped'])
    metadata["parameter_count"] = str(sum(p.numel() for p in state_dict.values()))


    logger.info(f"Saving state_dict to safetensors: {sf_path}")
    try:
        save_file(state_dict, sf_path, metadata=metadata)
        logger.info("Successfully saved safetensors file.")
    except Exception as e:
        logger.error(f"Failed to save {sf_path}: {e}", exc_info=True)
        return False

    # Compare file sizes
    try:
        pt_size = os.path.getsize(pt_path)
        sf_size = os.path.getsize(sf_path)
        logger.info(f"File size comparison:")
        logger.info(f"  Input  (.pt): {pt_size / (1024*1024):,.2f} MB")
        logger.info(f"  Output (.safetensors): {sf_size / (1024*1024):,.2f} MB")
        if pt_size > 0:
            reduction = (1 - sf_size / pt_size) * 100
            logger.info(f"  Size reduction: {reduction:.2f}%")
    except Exception as e:
        logger.warning(f"Could not compare file sizes: {e}")

    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert a PyTorch .pt checkpoint (containing model state_dict, possibly other data) "
                    "to a .safetensors file containing only the cleaned model weights."
    )
    parser.add_argument("--input_pt", help="Path to the input .pt file")
    parser.add_argument("--output_safetensors", help="Path to the output .safetensors file")

    args = parser.parse_args()

    # Ensure output path ends with .safetensors
    output_path = args.output_safetensors
    if not output_path.lower().endswith(".safetensors"):
        logger.warning(f"Output filename '{output_path}' does not end with .safetensors. Appending it.")
        output_path += ".safetensors"

    convert_pt_to_safetensors(args.input_pt, output_path)

'''
python pt_to_safetensor.py --input_pt "/mnt/data02/TTS/F5-TTS/ckpts/steve_combined_female/model_last.pt" --output_safetensors "/mnt/data02/TTS/F5-TTS/ckpts/steve_combined_female/model_last.safetensor"
'''
