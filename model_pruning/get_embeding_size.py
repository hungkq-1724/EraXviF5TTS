import torch
import argparse
import os
import logging
from safetensors.torch import load_file
from collections import OrderedDict

# --- Configuration ---
# Common keys where state_dict might be stored in .pt files (order matters)
STATE_DICT_KEYS_PT = [
    'model_state_dict',     # Common convention, often cleaned
    'state_dict',           # PyTorch Lightning default
    'ema_model_state_dict', # For EMA checkpoints (contains prefixed keys)
    'model',                # Sometimes used
    'module',               # DDP/DP wrappers
]

# Potential key names or endings for the main token embedding weight tensor
# Add more known patterns if needed for other model types
EMBEDDING_KEY_PATTERNS = [
    # Most specific first (based on F5)
    'transformer.text_embed.text_embed.weight',
    # Common patterns (check endings)
    'text_embed.text_embed.weight',
    'embed_tokens.weight',
    'wte.weight',
    'word_embeddings.weight',
    'token_embedding.weight',
    'input_embeddings.weight',
    'embeddings.word_embeddings.weight', # BERT style
]

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def find_state_dict_in_pt(checkpoint):
    """Finds the state dictionary within a loaded .pt checkpoint dictionary."""
    if not isinstance(checkpoint, dict):
        logger.warning("Input .pt checkpoint is not a dictionary. Assuming it *is* the state_dict.")
        return checkpoint, "[root]" # Indicate it was the root object

    for key in STATE_DICT_KEYS_PT:
        if key in checkpoint:
            potential_sd = checkpoint[key]
            if isinstance(potential_sd, dict) and potential_sd:
                logger.info(f"Found potential state_dict under key: '{key}'")
                return potential_sd, key
            else:
                logger.debug(f"Key '{key}' found but is not a valid state_dict. Skipping.")

    # Fallback: Check if root looks like state_dict
    is_likely_state_dict = all(isinstance(v, torch.Tensor) for k, v in checkpoint.items() if not k.startswith('_'))
    if is_likely_state_dict and len(checkpoint) > 0:
        logger.warning("No standard key found, but root dictionary looks like a state_dict. Using root.")
        return checkpoint, "[root]"

    logger.error(f"Could not find state_dict using keys {STATE_DICT_KEYS_PT} or root heuristic.")
    return None, None

def find_embedding_tensor(state_dict):
    """Finds the embedding tensor within a state dictionary using patterns."""
    if not isinstance(state_dict, dict) or not state_dict:
        logger.error("Invalid or empty state_dict provided.")
        return None, None

    # 1. Check for exact matches first
    for pattern in EMBEDDING_KEY_PATTERNS:
        if pattern in state_dict:
            tensor = state_dict[pattern]
            if isinstance(tensor, torch.Tensor) and tensor.ndim == 2:
                logger.info(f"Found embedding tensor via exact match: '{pattern}'")
                return tensor, pattern
            else:
                logger.warning(f"Key '{pattern}' found but is not a 2D Tensor. Skipping.")

    # 2. Check for keys ending with patterns (more general)
    for pattern_end in EMBEDDING_KEY_PATTERNS:
         # Avoid overly broad matches like just '.weight' unless specific
        if pattern_end.endswith('.weight') and len(pattern_end) > len('.weight'):
             for key, tensor in state_dict.items():
                 if key.endswith(pattern_end):
                      if isinstance(tensor, torch.Tensor) and tensor.ndim == 2:
                          logger.info(f"Found potential embedding tensor via suffix match: '{key}' (pattern ending: '{pattern_end}')")
                          return tensor, key
                      else:
                           logger.debug(f"Key '{key}' matches suffix but is not a 2D Tensor. Skipping.")

    logger.error(f"Could not find a likely embedding tensor using patterns: {EMBEDDING_KEY_PATTERNS}")
    return None, None


def get_embedding_info(model_path):
    """Loads a model file and reports embedding/vocab size."""
    if not os.path.exists(model_path):
        logger.error(f"File not found: {model_path}")
        return

    logger.info(f"Loading model file: {model_path}")
    state_dict = None
    source_info = ""

    try:
        if model_path.lower().endswith(".safetensors"):
            state_dict = load_file(model_path, device="cpu")
            source_info = "Loaded directly from .safetensors"
            logger.info(f"Successfully loaded .safetensors file. Found {len(state_dict)} tensors.")
            if not isinstance(state_dict, dict):
                 logger.error("Loaded safetensors data is not a dictionary.")
                 return

        elif model_path.lower().endswith(".pt"):
            logger.warning("Loading .pt file with weights_only=False. Ensure source is trusted.")
            checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
            logger.info("Successfully loaded .pt file.")
            state_dict, found_key = find_state_dict_in_pt(checkpoint)
            if state_dict:
                 source_info = f"Extracted from .pt using key: '{found_key}'"
            else:
                 logger.error("Could not extract state_dict from .pt file.")
                 return
            # Cleanup large checkpoint object if possible
            del checkpoint
            import gc
            gc.collect()

        else:
            logger.error(f"Unsupported file extension: {os.path.splitext(model_path)[1]}")
            return

        if state_dict is None or not state_dict:
            logger.error("Failed to get a valid state_dict.")
            return

        # --- Find Embedding Tensor ---
        embedding_tensor, found_key_name = find_embedding_tensor(state_dict)

        if embedding_tensor is None:
            logger.error("Could not identify the embedding weight tensor in the state_dict.")
            # Optionally print some keys to help user identify
            logger.info(f"Sample keys found: {list(state_dict.keys())[:20]}")
            return

        # --- Report Results ---
        shape = embedding_tensor.shape
        vocab_size = shape[0]
        embed_dim = shape[1]

        print("\n--- Embedding Info ---")
        print(f"File: {model_path}")
        print(f"Source: {source_info}")
        print(f"Identified Embedding Key: '{found_key_name}'")
        print(f"Embedding Tensor Shape: {list(shape)}")
        print(f"Detected Vocabulary Size: {vocab_size}")
        print(f"Detected Embedding Dimension: {embed_dim}")
        print("----------------------")

    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Inspect a .pt or .safetensors model file to find the vocabulary size "
                    "based on the token embedding layer."
    )
    parser.add_argument("model_path", help="Path to the input model file (.pt or .safetensors)")

    args = parser.parse_args()
    get_embedding_info(args.model_path)

'''
grep -cv '^\s*$' /home/steve/data02/TTS/F5-TTS/data/steve_combined_multi_char/vocab.txt
'''