from __future__ import annotations

import gc
import math
import os

import torch
import torchaudio
import wandb
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from ema_pytorch import EMA
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, SequentialLR
from torch.utils.data import DataLoader, Dataset, SequentialSampler
from tqdm import tqdm

from f5_tts.model import CFM
from f5_tts.model.dataset import DynamicBatchSampler, collate_fn
from f5_tts.model.utils import default, exists

# trainer

import re

class Trainer:
    def __init__(
        self,
        model: CFM,
        epochs,
        learning_rate,
        weight_decay=0.1,
        num_warmup_updates=20000,
        save_per_updates=1000,
        keep_last_n_checkpoints: int = -1,  # -1 to keep all, 0 to not save intermediate, > 0 to keep last N checkpoints
        checkpoint_path=None,
        batch_size_per_gpu=32,
        batch_size_type: str = "sample",
        max_samples=32,
        grad_accumulation_steps=1,
        max_grad_norm=1.0,
        noise_scheduler: str | None = None,
        duration_predictor: torch.nn.Module | None = None,
        logger: str | None = "wandb",  # "wandb" | "tensorboard" | None
        wandb_project="test_f5-tts",
        wandb_run_name="test_run",
        wandb_resume_id: str = None,
        log_samples: bool = False,
        last_per_updates=None,
        accelerate_kwargs: dict = dict(),
        ema_kwargs: dict = dict(),
        bnb_optimizer: bool = False,
        mel_spec_type: str = "vocos",  # "vocos" | "bigvgan"
        is_local_vocoder: bool = False,  # use local path vocoder
        local_vocoder_path: str = "",  # local vocoder path
        model_cfg_dict: dict = dict(),  # training config
        duration_loss_weight: float = 0.1,
        ref_texts=None,  # List of reference texts for consistent sample generation
        ref_audio_paths=None,  # List of paths to reference audio files
        ref_sample_text_prompts=None,  # Text prompts to use for the reference samples
    ):
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)

        if logger == "wandb" and not wandb.api.api_key:
            logger = None
        self.log_samples = log_samples

        self.accelerator = Accelerator(
            log_with=logger if logger == "wandb" else None,
            kwargs_handlers=[ddp_kwargs],
            gradient_accumulation_steps=grad_accumulation_steps,
            **accelerate_kwargs,
        )

        self.logger = logger
        if self.logger == "wandb":
            if exists(wandb_resume_id):
                init_kwargs = {"wandb": {"resume": "allow", "name": wandb_run_name, "id": wandb_resume_id}}
            else:
                init_kwargs = {"wandb": {"resume": "allow", "name": wandb_run_name}}

            if not model_cfg_dict:
                model_cfg_dict = {
                    "epochs": epochs,
                    "learning_rate": learning_rate,
                    "num_warmup_updates": num_warmup_updates,
                    "batch_size_per_gpu": batch_size_per_gpu,
                    "batch_size_type": batch_size_type,
                    "max_samples": max_samples,
                    "grad_accumulation_steps": grad_accumulation_steps,
                    "max_grad_norm": max_grad_norm,
                    "noise_scheduler": noise_scheduler,
                }
            model_cfg_dict["gpus"] = self.accelerator.num_processes
            self.accelerator.init_trackers(
                project_name=wandb_project,
                init_kwargs=init_kwargs,
                config=model_cfg_dict,
            )

        elif self.logger == "tensorboard":
            from torch.utils.tensorboard import SummaryWriter

            self.writer = SummaryWriter(log_dir=f"runs/{wandb_run_name}")

        self.model = model

        if self.is_main:
            self.ema_model = EMA(model, include_online_model=False, **ema_kwargs)
            self.ema_model.to(self.accelerator.device)

            print(f"Using logger: {logger}")
            if grad_accumulation_steps > 1:
                print(
                    "Gradient accumulation checkpointing with per_updates now, old logic per_steps used with before f992c4e"
                )
 
        self.epochs = epochs
        self.num_warmup_updates = num_warmup_updates
        self.save_per_updates = save_per_updates
        self.keep_last_n_checkpoints = keep_last_n_checkpoints
        self.last_per_updates = default(last_per_updates, save_per_updates)
        self.checkpoint_path = default(checkpoint_path, "ckpts/test_f5-tts")

        self.batch_size_per_gpu = batch_size_per_gpu
        self.batch_size_type = batch_size_type
        self.max_samples = max_samples
        self.grad_accumulation_steps = grad_accumulation_steps
        self.max_grad_norm = max_grad_norm

        # mel vocoder config
        self.vocoder_name = mel_spec_type
        self.is_local_vocoder = is_local_vocoder
        self.local_vocoder_path = local_vocoder_path

        self.noise_scheduler = noise_scheduler

        # Duration predictor setup
        self.duration_predictor = duration_predictor
        self.duration_loss_weight = duration_loss_weight
        
        # If duration predictor is provided, attach it to the model
        if self.duration_predictor is not None:
            setattr(self.model, 'duration_predictor', self.duration_predictor)
            total_params = sum(p.numel() for p in self.duration_predictor.parameters() if p.requires_grad)
            print(f"Total number of trainable parameters in Duration Predictor: {total_params}")

        if bnb_optimizer:
            import bitsandbytes as bnb

            self.optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=learning_rate, weight_decay=weight_decay, 
                                                 betas=(0.9, 0.98),
                                                 eps=1e-8)
        else:
            self.optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay, 
                                   betas=(0.9, 0.98),
                                   eps=1e-8)
        self.model, self.optimizer = self.accelerator.prepare(self.model, self.optimizer)

        # Store reference sample information
        self.ref_texts = ref_texts or []
        self.ref_audio_paths = ref_audio_paths or []
        self.ref_sample_text_prompts = ref_sample_text_prompts or []
        
        # Ensure lists are the same length or empty
        max_len = max(len(self.ref_texts), len(self.ref_audio_paths), len(self.ref_sample_text_prompts))
        if max_len > 0:
            if len(self.ref_texts) > 0 and len(self.ref_texts) != max_len:
                print(f"Warning: ref_texts length ({len(self.ref_texts)}) does not match other reference lists ({max_len}). Using available items only.")
            if len(self.ref_audio_paths) > 0 and len(self.ref_audio_paths) != max_len:
                print(f"Warning: ref_audio_paths length ({len(self.ref_audio_paths)}) does not match other reference lists ({max_len}). Using available items only.")
            if len(self.ref_sample_text_prompts) > 0 and len(self.ref_sample_text_prompts) != max_len:
                print(f"Warning: ref_sample_text_prompts length ({len(self.ref_sample_text_prompts)}) does not match other reference lists ({max_len}). Using available items only.")
        
        # Pre-load reference audios if paths are provided
        self.ref_mels = []
        if self.is_main and self.ref_audio_paths and log_samples:
            try:
                from f5_tts.model.modules import MelSpec
                
                # First, determine the target sample rate from the model or use a default
                target_sample_rate = 24000  # Default value
                if hasattr(self.model, 'mel_spec') and hasattr(self.model.mel_spec, 'target_sample_rate'):
                    target_sample_rate = self.model.mel_spec.target_sample_rate
                
                # Create a mel spectrogram converter if needed
                mel_spec_kwargs = {
                    'n_fft': 1024, 
                    'hop_length': 256, 
                    'win_length': 1024,
                    'n_mel_channels': n_mel_channels if 'n_mel_channels' in locals() else 100,
                    'target_sample_rate': target_sample_rate,
                    'mel_spec_type': mel_spec_type
                }
                
                mel_spec = self.model.mel_spec if hasattr(self.model, 'mel_spec') else MelSpec(**mel_spec_kwargs)
                
                print(f"Loading reference audios with target sample rate: {target_sample_rate}")
                for audio_path in self.ref_audio_paths:
                    if os.path.exists(audio_path):
                        # Load audio and convert to mel spectrogram
                        print(f"Loading reference audio: {audio_path}")
                        waveform, sr = torchaudio.load(audio_path)
                        print(f"  Original sample rate: {sr}")
                        
                        if sr != target_sample_rate:
                            print(f"  Resampling from {sr} to {target_sample_rate}")
                            waveform = torchaudio.functional.resample(waveform, sr, target_sample_rate)
                        
                        # Convert to mel spectrogram
                        mel = mel_spec(waveform.to(self.accelerator.device)).cpu()
                        self.ref_mels.append(mel)
                        print(f"  Successfully loaded reference audio. Mel shape: {mel.shape}")
                        
                        # Print the associated text and prompt if available
                        idx = len(self.ref_mels) - 1
                        if idx < len(self.ref_texts):
                            print(f"  Reference text: {self.ref_texts[idx]}")
                        if idx < len(self.ref_sample_text_prompts):
                            print(f"  Sample text prompt: {self.ref_sample_text_prompts[idx]}")
                    else:
                        print(f"Warning: Reference audio file not found: {audio_path}")
            except Exception as e:
                print(f"Error loading reference audio files: {e}")
                import traceback
                traceback.print_exc()  # Print the full traceback for better debugging
                
    # Remove or comment out the pinyin import if it was added
    # from f5_tts.model.utils import convert_char_to_pinyin

    # ... inside the Trainer class ...

    def generate_reference_samples(self, global_update, vocoder, nfe_step, cfg_strength, sway_sampling_coef):
        """
        Generate samples from fixed reference examples for consistent quality monitoring.
        Uses combined Ref+Prompt CHARACTER text and slices output mel.
        """
        # Ensure this runs only on the main process and if logging/references are enabled
        if not self.is_main or not self.log_samples or not self.ref_mels:
            return
        
        log_samples_path = f"{self.checkpoint_path}/ref_samples"
        os.makedirs(log_samples_path, exist_ok=True)
        
        # Select the model for sampling (EMA preferably)
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        model_for_sampling = self.ema_model.ema_model if hasattr(self, 'ema_model') else unwrapped_model
        target_sample_rate = unwrapped_model.mel_spec.target_sample_rate
        
        # Set model to evaluation mode for sampling
        model_for_sampling.eval() 
        
        with torch.inference_mode():
            # Iterate through each reference mel provided during initialization
            for idx, ref_mel in enumerate(self.ref_mels):
                try:
                    # --- Get Corresponding Reference Text and Prompt Text ---
                    ref_text = self.ref_texts[idx] if idx < len(self.ref_texts) else ""
                    prompt_text_for_generation = self.ref_sample_text_prompts[idx] if idx < len(self.ref_sample_text_prompts) else "Default test reference."
                    original_full_prompt = prompt_text_for_generation # For saving later
                    
                    # --- FIX 1: Combine text using CHARACTERS ---
                    # Ensure texts are strings. Add space if needed for separation.
                    combined_text = str(ref_text) + " " + str(prompt_text_for_generation)
                    # Wrap in a list as model.sample expects list[str] or tensor
                    final_text_list_for_model = [combined_text]
                    print(f"Using combined CHARACTER text for model: {combined_text[:100]}...")

                    # Prepare the reference mel spectrogram
                    ref_mel = ref_mel.to(self.accelerator.device) 
                    ref_mel_for_sample = ref_mel.permute(0, 2, 1) # Shape [1, mel_len, n_mel_channels]

                    # Decode and Save Original Reference Audio (remains the same)
                    print(f"Decoding reference mel (shape: {ref_mel.shape}) for saving source audio")
                    if self.vocoder_name == "vocos":
                        ref_audio = vocoder.decode(ref_mel).cpu()
                    elif self.vocoder_name == "bigvgan":
                        ref_audio = vocoder(ref_mel).squeeze(0).cpu()
                    ref_path = f"{log_samples_path}/update_{global_update}_ref{idx}_source.wav"
                    if ref_audio.ndim == 1: ref_audio = ref_audio.unsqueeze(0)
                    elif ref_audio.ndim == 3 and ref_audio.shape[1] == 1: ref_audio = ref_audio.squeeze(1)
                    torchaudio.save(ref_path, ref_audio, target_sample_rate)
                    print(f"Saved reference audio source: {ref_path}")
                    
                    # Determine Target Duration (keep * 2 for now)
                    batch_size, cond_seq_len = ref_mel_for_sample.shape[:2]
                    target_duration_frames = cond_seq_len * 2 
                    target_duration = torch.full((batch_size,), target_duration_frames, device=ref_mel_for_sample.device, dtype=torch.long)
                    print(f"Requesting target duration for sample: {target_duration.item()} frames")

                    # Generate Mel Spectrogram using the Model with CHARACTER text
                    print(f"Generating sample mel spectrogram...")
                    generated_mel, _ = model_for_sampling.sample(
                        cond=ref_mel_for_sample,
                        text=final_text_list_for_model, # Use combined CHARACTER text
                        duration=target_duration, 
                        steps=nfe_step,
                        cfg_strength=cfg_strength,
                        sway_sampling_coef=sway_sampling_coef,
                    )
                    # Output shape: [batch, target_duration_frames, dim]
                    
                    # Check for Invalid Values
                    if torch.isnan(generated_mel).any() or torch.isinf(generated_mel).any():
                        print(f"Error: NaNs or Infs detected in generated mel for reference sample {idx}. Skipping generation.")
                        continue 
                        
                    # Prepare Generated Mel for Vocoder
                    generated_mel = generated_mel.to(torch.float32)
                    ref_mel_frames = ref_mel.shape[-1] # Same as cond_seq_len
                    print(f"Original reference mel frames: {ref_mel_frames}")
                    print(f"Generated mel shape: {generated_mel.shape}")

                    # --- FIX 2: SLICE the output mel like the Wrapper ---
                    if generated_mel.shape[1] > ref_mel_frames:
                         mel_for_vocoder = generated_mel[:, ref_mel_frames:, :].permute(0, 2, 1).to(self.accelerator.device) # Slice and permute
                         print(f"Using SLICED generated mel (after ref) for vocoder. Shape: {mel_for_vocoder.shape}")
                    else:
                         print(f"Warning: Generated mel length ({generated_mel.shape[1]}) not longer than reference ({ref_mel_frames}). Output slice is empty.")
                         mel_for_vocoder = None 

                    # Decode Sliced Mel to Audio
                    if mel_for_vocoder is not None and mel_for_vocoder.shape[-1] > 0: 
                         print(f"Decoding sliced mel for saving generated audio")
                         if self.vocoder_name == "vocos":
                             gen_audio = vocoder.decode(mel_for_vocoder).cpu()
                         elif self.vocoder_name == "bigvgan":
                             gen_audio = vocoder(mel_for_vocoder).squeeze(0).cpu()

                         # Ensure audio tensor shape [1, audio_len] for torchaudio.save
                         if gen_audio.ndim == 1: gen_audio = gen_audio.unsqueeze(0)
                         elif gen_audio.ndim == 3 and gen_audio.shape[1] == 1: gen_audio = gen_audio.squeeze(1)

                         # Save Generated Audio
                         gen_path = f"{log_samples_path}/update_{global_update}_ref{idx}_gen.wav"
                         torchaudio.save(gen_path, gen_audio, target_sample_rate)
                         print(f"Saved generated sample (from sliced mel): {gen_path}")
                    else:
                         print("Skipping saving generated audio: sliced mel part is empty or generation too short.")

                    # Save Prompt Text (Save the original prompt for clarity)
                    txt_path = f"{log_samples_path}/update_{global_update}_ref{idx}_prompt.txt"
                    with open(txt_path, 'w') as f:
                        f.write(original_full_prompt)
                    
                    print(f"Completed processing for reference sample {idx} at update {global_update}")
                    
                except Exception as e:
                    print(f"Error processing reference sample {idx}: {e}")
                    import traceback
                    traceback.print_exc()
        
        # Restore model to training mode after processing all samples
        model_for_sampling.train()
        
    @property
    def is_main(self):
        return self.accelerator.is_main_process

    def save_checkpoint(self, update, last=False):
        self.accelerator.wait_for_everyone()
        if self.is_main:
            checkpoint = dict(
                model_state_dict=self.accelerator.unwrap_model(self.model).state_dict(),
                optimizer_state_dict=self.accelerator.unwrap_model(self.optimizer).state_dict(),
                ema_model_state_dict=self.ema_model.state_dict(),
                scheduler_state_dict=self.scheduler.state_dict(),
                update=update,
            )
            if not os.path.exists(self.checkpoint_path):
                os.makedirs(self.checkpoint_path)
                
            if last:
                self.accelerator.save(checkpoint, f"{self.checkpoint_path}/model_last.pt")
                print(f"Saved last checkpoint at update {update}")
            else:
                if self.keep_last_n_checkpoints == 0:
                    return
                self.accelerator.save(checkpoint, f"{self.checkpoint_path}/model_{update}.pt")
                if self.keep_last_n_checkpoints > 0:
                    try:
                        # Get all checkpoint files excluding pretrained models and last checkpoint
                        checkpoints = [
                            f
                            for f in os.listdir(self.checkpoint_path)
                            if f.startswith("model_")
                            and not f.startswith("pretrained_")  # Exclude pretrained models
                            and f.endswith(".pt")
                            and f != "model_last.pt"  # Exclude the last checkpoint
                        ]
                        
                        # Filter to only include numerically named checkpoints
                        numeric_checkpoints = []
                        for ckpt in checkpoints:
                            try:
                                # Extract the number part and validate it's an integer
                                num_part = ckpt.split("_")[1].split(".")[0]
                                int(num_part)  # This will raise ValueError if not a valid integer
                                numeric_checkpoints.append(ckpt)
                            except (ValueError, IndexError):
                                # Skip files that don't follow the expected naming pattern
                                print(f"Skipping non-numeric checkpoint during cleanup: {ckpt}")
                        
                        # Sort the numeric checkpoints by their number
                        numeric_checkpoints.sort(key=lambda x: int(x.split("_")[1].split(".")[0]))
                        
                        # Remove oldest checkpoints if we have more than the limit
                        while len(numeric_checkpoints) > self.keep_last_n_checkpoints:
                            oldest_checkpoint = numeric_checkpoints.pop(0)
                            os.remove(os.path.join(self.checkpoint_path, oldest_checkpoint))
                            print(f"Removed old checkpoint: {oldest_checkpoint}")
                            
                    except Exception as e:
                        print(f"Warning: Error during checkpoint cleanup: {e}")
                        # Don't delete any checkpoints if there's an error in sorting/handling
                        

    def load_checkpoint(self):
        """Loads the most recent checkpoint from the checkpoint_path."""
        # --- Determine Checkpoint Path ---
        latest_checkpoint_name = None
        checkpoint_path_to_load = None
        if exists(self.checkpoint_path) and os.path.isdir(self.checkpoint_path):
            # Prioritize model_last.pt
            last_ckpt_path = os.path.join(self.checkpoint_path, "model_last.pt")
            if os.path.exists(last_ckpt_path):
                latest_checkpoint_name = "model_last.pt"
                self.accelerator.print(f"Found last checkpoint: {latest_checkpoint_name}")
            else:
                # Find latest training or pretrained checkpoint
                all_checkpoints = [
                    f for f in os.listdir(self.checkpoint_path)
                    if (f.startswith("model_") or f.startswith("pretrained_")) and f.endswith((".pt", ".safetensors"))
                ]
                training_checkpoints = [f for f in all_checkpoints if f.startswith("model_") and f != "model_last.pt"]
                pretrained_checkpoints = [f for f in all_checkpoints if f.startswith("pretrained_")]

                latest_update = -1
                latest_training_ckpt = None
                if training_checkpoints: # Prioritize latest training checkpoint
                    try:
                        for ckpt_name in training_checkpoints:
                            match = re.search(r'model_(\d+)', ckpt_name)
                            if match:
                                update_num = int(match.group(1))
                                if update_num > latest_update:
                                     latest_update = update_num
                                     latest_training_ckpt = ckpt_name
                        if latest_training_ckpt:
                             latest_checkpoint_name = latest_training_ckpt
                             self.accelerator.print(f"Found latest training checkpoint: {latest_checkpoint_name}")
                    except Exception as e: # Fallback if sorting fails
                         self.accelerator.print(f"Warning: Failed to sort training checkpoints by update number: {e}. Using last alphabetically.")
                         latest_checkpoint_name = sorted(training_checkpoints)[-1] if training_checkpoints else None
                elif pretrained_checkpoints: # If no training checkpoints, use the first found pretrained one
                    latest_checkpoint_name = sorted(pretrained_checkpoints)[0] # Sort for consistency
                    self.accelerator.print(f"No training checkpoints found. Using pretrained checkpoint: {latest_checkpoint_name}")
                else:
                     self.accelerator.print("No suitable checkpoints (last, training, or pretrained) found.")

            if latest_checkpoint_name:
                checkpoint_path_to_load = os.path.join(self.checkpoint_path, latest_checkpoint_name)

        if checkpoint_path_to_load is None or not os.path.exists(checkpoint_path_to_load):
            self.accelerator.print("No valid checkpoint found. Starting from scratch.")
            return 0 # Return 0 updates processed

        # --- Load the Checkpoint File ---
        self.accelerator.print(f"Loading checkpoint: {checkpoint_path_to_load}")
        checkpoint = None
        try:
            if checkpoint_path_to_load.endswith(".safetensors"):
                 from safetensors.torch import load_file
                 loaded_data = load_file(checkpoint_path_to_load, device="cpu")
                 checkpoint = {'state_dict_loaded_from_safetensors': loaded_data}
                 self.accelerator.print("Loaded state_dict from .safetensors file.")
            elif checkpoint_path_to_load.endswith(".pt"):
                 self.accelerator.print("Loading .pt file with weights_only=False. Ensure checkpoint source is trusted.")
                 checkpoint = torch.load(checkpoint_path_to_load, map_location="cpu", weights_only=False)
                 self.accelerator.print("Loaded checkpoint from .pt file.")
            else:
                 raise ValueError(f"Unsupported checkpoint file extension: {checkpoint_path_to_load}")
            if not isinstance(checkpoint, dict):
                 raise TypeError("Loaded checkpoint is not a dictionary.")
        except Exception as e:
            self.accelerator.print(f"Error loading checkpoint file: {e}. Starting from scratch.")
            return 0

        # --- Find and Prepare Model State Dictionary ---
        model_sd_raw = None
        loaded_from_key = None
        is_ema_source = False

        search_keys = ['model_state_dict', 'ema_model_state_dict', 'state_dict', 'model', 'state_dict_loaded_from_safetensors']
        for key in search_keys:
            if key in checkpoint:
                potential_sd = checkpoint[key]
                if isinstance(potential_sd, dict) and potential_sd:
                    model_sd_raw = potential_sd
                    loaded_from_key = key
                    if key == 'ema_model_state_dict': is_ema_source = True
                    self.accelerator.print(f"Found model weights under key: '{loaded_from_key}' (Is EMA source: {is_ema_source})")
                    break
                elif not isinstance(potential_sd, dict):
                     self.accelerator.print(f"Warning: Key '{key}' found but value is not a dict (type: {type(potential_sd)}). Skipping.")
                elif not potential_sd:
                     self.accelerator.print(f"Warning: Key '{key}' found but dictionary is empty. Skipping.")

        if model_sd_raw is None:
            self.accelerator.print(f"ERROR: Could not find usable model state_dict in checkpoint: {checkpoint_path_to_load}")
            self.accelerator.print(f"Available top-level keys: {list(checkpoint.keys())}")
            self.accelerator.print("Starting from scratch as state_dict was not found.")
            return 0

        # --- Clean Prefixes and Metadata ---
        model_sd_cleaned = {}
        prefixes_to_strip = ["module.", "model.", "_orig_mod."]
        if is_ema_source:
             ema_prefix = "ema_model."
             if any(k.startswith(ema_prefix) for k in model_sd_raw): prefixes_to_strip.insert(0, ema_prefix)

        used_prefix = None
        first_key = next(iter(model_sd_raw.keys()), None)
        if first_key:
            for prefix in prefixes_to_strip:
                if first_key.startswith(prefix):
                    prefix_count = sum(1 for k in model_sd_raw if k.startswith(prefix))
                    if prefix_count >= 0.8 * len(model_sd_raw): used_prefix = prefix; break

        ignore_keys = {"initted", "step"}
        if used_prefix:
            self.accelerator.print(f"Stripping prefix '{used_prefix}' from state_dict keys.")
            prefix_len = len(used_prefix)
            for k, v in model_sd_raw.items():
                final_key = k[prefix_len:] if k.startswith(used_prefix) else k
                if final_key not in ignore_keys: model_sd_cleaned[final_key] = v
                else: self.accelerator.print(f"Ignoring metadata key while cleaning: {k}")
        else:
            self.accelerator.print("No common prefix found or stripping not applicable.")
            for k, v in model_sd_raw.items():
                 if k not in ignore_keys: model_sd_cleaned[k] = v
                 else: self.accelerator.print(f"Ignoring metadata key: {k}")

        if not model_sd_cleaned:
             self.accelerator.print("ERROR: State dictionary became empty after cleaning. Check original checkpoint.")
             return 0

        # --- Load Cleaned State Dict into the Main Model ---
        load_successful = False
        try:
            unwrapped_model = self.accelerator.unwrap_model(self.model)
            incompatible_keys = unwrapped_model.load_state_dict(model_sd_cleaned, strict=False)
            self.accelerator.print("Successfully loaded model weights into main model.")
            if incompatible_keys.missing_keys:
                 self.accelerator.print(f"Note: Missing keys when loading state_dict (expected if model structure changed): {incompatible_keys.missing_keys}")
            if incompatible_keys.unexpected_keys:
                 self.accelerator.print(f"Note: Unexpected keys when loading state_dict (expected if checkpoint has extra keys): {incompatible_keys.unexpected_keys}")
            load_successful = True
        except Exception as e:
            self.accelerator.print(f"ERROR loading state_dict into model: {e}") # Use accelerator.print

        if not load_successful:
             self.accelerator.print("Weights could not be loaded. Starting from scratch.")
             del checkpoint; del model_sd_raw; del model_sd_cleaned
             if 'loaded_data' in locals(): del loaded_data
             self.accelerator.wait_for_everyone()
             if self.accelerator.device.type == 'cuda': torch.cuda.empty_cache()
             elif self.accelerator.device.type == 'xpu': torch.xpu.empty_cache()
             gc.collect()
             return 0

        # --- Handle Resuming Full Training State ---
        is_resuming_full_state = 'optimizer_state_dict' in checkpoint and 'update' in checkpoint

        start_update = 0
        if is_resuming_full_state:
             self.accelerator.print("Attempting to load full training state (optimizer, scheduler, EMA, step)...")
             try:
                 self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                 self.accelerator.print("Optimizer state loaded.")
             except Exception as e:
                 self.accelerator.print(f"Warning: Failed to load optimizer state: {e}. Optimizer will start fresh.")

             if hasattr(self, 'scheduler') and self.scheduler and 'scheduler_state_dict' in checkpoint:
                 try:
                     self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                     self.accelerator.print("Scheduler state loaded.")
                 except Exception as e:
                     self.accelerator.print(f"Warning: Failed to load scheduler state: {e}.")

             if self.is_main and hasattr(self, 'ema_model') and 'ema_model_state_dict' in checkpoint:
                 try:
                      ema_sd_to_load = checkpoint['ema_model_state_dict']
                      ema_model_unwrapped = self.ema_model.module if hasattr(self.ema_model, 'module') else self.ema_model
                      incompatible_ema_keys = ema_model_unwrapped.load_state_dict(ema_sd_to_load, strict=False)
                      self.accelerator.print("Trainer EMA state loaded.")
                      if incompatible_ema_keys.missing_keys: self.accelerator.print(f"Note: Missing EMA keys: {incompatible_ema_keys.missing_keys}")
                      if incompatible_ema_keys.unexpected_keys: self.accelerator.print(f"Note: Unexpected EMA keys: {incompatible_ema_keys.unexpected_keys}")
                 except Exception as e:
                      self.accelerator.print(f"Warning: Failed to load EMA state into trainer: {e}. EMA will start fresh.")
             elif self.is_main and hasattr(self, 'ema_model'):
                  self.accelerator.print("EMA state not found in checkpoint for resume. EMA will start fresh.")

             start_update = checkpoint.get('update', checkpoint.get('step', -1))
             if start_update == -1:
                  self.accelerator.print("Warning: Resuming checkpoint missing 'update' or 'step'. Starting from update 0.")
                  start_update = 0
             else:
                  if 'step' in checkpoint and 'update' not in checkpoint and self.grad_accumulation_steps > 1:
                       start_update = start_update // self.grad_accumulation_steps
                       self.accelerator.print("Converted loaded 'step' to 'update' based on grad_accumulation_steps.")
                  start_update += 1
             self.step = start_update
             self.accelerator.print(f"Resuming training from update {start_update}")
        else:
             self.step = 0
             start_update = 0
             # Use f-string correctly for loaded_from_key
             self.accelerator.print(f"Loaded pre-trained weights (from key '{loaded_from_key}'). Starting fine-tuning from update 0.")

        # --- Cleanup ---
        del checkpoint
        del model_sd_raw
        del model_sd_cleaned
        if 'loaded_data' in locals(): del loaded_data
        self.accelerator.wait_for_everyone()
        if self.accelerator.device.type == 'cuda': torch.cuda.empty_cache()
        elif self.accelerator.device.type == 'xpu': torch.xpu.empty_cache()
        gc.collect()

        self.accelerator.print("Checkpoint loading process finished.")
        return start_update

    def train(self, train_dataset: Dataset, num_workers=16, resumable_with_seed: int = None):
        
        self.clear_GPU_steps = 100
        
        if self.log_samples:
            from f5_tts.infer.utils_infer import cfg_strength, load_vocoder, nfe_step, sway_sampling_coef

            vocoder = load_vocoder(
                vocoder_name=self.vocoder_name, is_local=self.is_local_vocoder, local_path=self.local_vocoder_path
            )
            target_sample_rate = self.accelerator.unwrap_model(self.model).mel_spec.target_sample_rate
            log_samples_path = f"{self.checkpoint_path}/samples"
            os.makedirs(log_samples_path, exist_ok=True)

        if exists(resumable_with_seed):
            generator = torch.Generator()
            generator.manual_seed(resumable_with_seed)
        else:
            generator = None

        if self.batch_size_type == "sample":
            train_dataloader = DataLoader(
                train_dataset,
                collate_fn=collate_fn,
                num_workers=8,
                pin_memory=True,
                persistent_workers=True,
                prefetch_factor=2,
                batch_size=self.batch_size_per_gpu,
                shuffle=True,
                generator=generator,
            )
        elif self.batch_size_type == "frame":
            self.accelerator.even_batches = False
            sampler = SequentialSampler(train_dataset)
            batch_sampler = DynamicBatchSampler(
                sampler,
                self.batch_size_per_gpu,
                max_samples=self.max_samples,
                random_seed=resumable_with_seed,  # This enables reproducible shuffling
                drop_residual=False,
            )
            train_dataloader = DataLoader(
                train_dataset,
                collate_fn=collate_fn,
                num_workers=8,
                pin_memory=True,
                prefetch_factor=2,
                persistent_workers=True,
                batch_sampler=batch_sampler,
            )
        else:
            raise ValueError(f"batch_size_type must be either 'sample' or 'frame', but received {self.batch_size_type}")

        #  accelerator.prepare() dispatches batches to devices;
        #  which means the length of dataloader calculated before, should consider the number of devices
        warmup_updates = (
            self.num_warmup_updates * self.accelerator.num_processes
        )  # consider a fixed warmup steps while using accelerate multi-gpu ddp
        # otherwise by default with split_batches=False, warmup steps change with num_processes
        total_updates = math.ceil(len(train_dataloader) / self.grad_accumulation_steps) * self.epochs
        decay_updates = total_updates - warmup_updates
        warmup_scheduler = LinearLR(self.optimizer, start_factor=1e-8, end_factor=1.0, total_iters=warmup_updates)
        decay_scheduler = LinearLR(self.optimizer, start_factor=1.0, end_factor=1e-8, total_iters=decay_updates)
        self.scheduler = SequentialLR(
            self.optimizer, schedulers=[warmup_scheduler, decay_scheduler], milestones=[warmup_updates]
        )
        train_dataloader, self.scheduler = self.accelerator.prepare(
            train_dataloader, self.scheduler
        )  # actual multi_gpu updates = single_gpu updates / gpu nums
        start_update = self.load_checkpoint()
        global_update = start_update

        if exists(resumable_with_seed):
            orig_epoch_step = len(train_dataloader)
            start_step = start_update * self.grad_accumulation_steps
            skipped_epoch = int(start_step // orig_epoch_step)
            skipped_batch = start_step % orig_epoch_step
            skipped_dataloader = self.accelerator.skip_first_batches(train_dataloader, num_batches=skipped_batch)
        else:
            skipped_epoch = 0

        for epoch in range(skipped_epoch, self.epochs):
            self.model.train()
            if exists(resumable_with_seed) and epoch == skipped_epoch:
                progress_bar_initial = math.ceil(skipped_batch / self.grad_accumulation_steps)
                current_dataloader = skipped_dataloader
            else:
                progress_bar_initial = 0
                current_dataloader = train_dataloader

            # Set epoch for the batch sampler if it exists
            if hasattr(train_dataloader, "batch_sampler") and hasattr(train_dataloader.batch_sampler, "set_epoch"):
                train_dataloader.batch_sampler.set_epoch(epoch)

            progress_bar = tqdm(
                range(math.ceil(len(train_dataloader) / self.grad_accumulation_steps)),
                desc=f"Epoch {epoch + 1}/{self.epochs}",
                unit="update",
                disable=not self.accelerator.is_local_main_process,
                initial=progress_bar_initial,
            )

            for batch in current_dataloader:
                with self.accelerator.accumulate(self.model):
                    text_inputs = batch["text"]
                    mel_spec = batch["mel"].permute(0, 2, 1)
                    mel_lengths = batch["mel_lengths"]
                    text_lengths = batch.get("text_lengths")
                    attn = batch.get("attn")

                    # Forward pass through the main model
                    loss, cond, pred = self.model(
                        mel_spec, text=text_inputs, lens=mel_lengths, noise_scheduler=self.noise_scheduler
                    )
                    
                    # Duration prediction loss calculation if predictor exists and attn is available
                    dur_loss = None
                    if self.duration_predictor is not None and attn is not None and text_lengths is not None:
                        # Get text tokens from the model (we might need to access these differently based on model impl)
                        text_tokens = batch["text"]
                        if hasattr(self.model, 'get_text_tokens'):
                            text_tokens = self.model.get_text_tokens(text_inputs)
                        
                        # Create mask for text tokens
                        b, nt = text_tokens.shape  # Get batch size and sequence length
                        range_tensor = torch.arange(nt, device=text_tokens.device).unsqueeze(0)  # Shape [1, nt]
                        text_tokens_mask = (range_tensor < text_lengths.unsqueeze(1)).int()
                        
                        # Calculate durations from attention
                        w = attn.sum(dim=2) # [b nt]
                        logw_ = torch.log(w + 1e-6) * text_tokens_mask
                        
                        # Get duration predictions
                        logw = self.duration_predictor(text_tokens, text_tokens_mask)
                        
                        # Calculate duration loss
                        l_length = torch.sum((logw - logw_)**2, [1,2]) / torch.sum(text_tokens_mask)
                        dur_loss = torch.sum(l_length.float())
                        
                        # Add weighted duration loss to main loss
                        if self.duration_loss_weight > 0:
                            loss = loss + self.duration_loss_weight * dur_loss
                        
                        # Log duration loss
                        if self.accelerator.is_local_main_process:
                            self.accelerator.log({"duration_loss": dur_loss.item()}, step=global_update)
                    
                    # Backward pass
                    self.accelerator.backward(loss)

                    if self.max_grad_norm > 0 and self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

                if self.accelerator.sync_gradients:
                    if self.is_main:
                        self.ema_model.update()

                    global_update += 1
                    progress_bar.update(1)
                    progress_bar.set_postfix(update=str(global_update), loss=loss.item())

                if self.accelerator.is_local_main_process:
                    log_dict = {"loss": loss.item(), "lr": self.scheduler.get_last_lr()[0]}
                    if dur_loss is not None:
                        log_dict["duration_loss"] = dur_loss.item()
                    
                    self.accelerator.log(log_dict, step=global_update)
                    
                    if self.logger == "tensorboard":
                        self.writer.add_scalar("loss", loss.item(), global_update)
                        if dur_loss is not None:
                            self.writer.add_scalar("duration_loss", dur_loss.item(), global_update)
                        self.writer.add_scalar("lr", self.scheduler.get_last_lr()[0], global_update)

                if global_update % self.save_per_updates == 0 and self.accelerator.sync_gradients:
                    self.save_checkpoint(global_update)
                    import gc
                    # Trigger garbage collection manually
                    gc.collect()
                
                    # Clear GPU memory for every clear_GPU_steps (100 - 500) 
                    if global_update % self.clear_GPU_steps == 0:
                        torch.cuda.empty_cache()
                    
                    if self.log_samples and self.accelerator.is_local_main_process:
                        # Generate training batch sample as usual
                        ref_audio_len = mel_lengths[0]
                        infer_text = [
                            text_inputs[0] + ([" "] if isinstance(text_inputs[0], list) else " ") + text_inputs[0]
                        ]
                        with torch.inference_mode():
                            generated, _ = self.accelerator.unwrap_model(self.model).sample(
                                cond=mel_spec[0][:ref_audio_len].unsqueeze(0),
                                text=infer_text,
                                duration=ref_audio_len * 2,
                                steps=nfe_step,
                                cfg_strength=cfg_strength,
                                sway_sampling_coef=sway_sampling_coef,
                            )
                            generated = generated.to(torch.float32)
                            gen_mel_spec = generated[:, ref_audio_len:, :].permute(0, 2, 1).to(self.accelerator.device)
                            ref_mel_spec = batch["mel"][0].unsqueeze(0)
                            if self.vocoder_name == "vocos":
                                gen_audio = vocoder.decode(gen_mel_spec).cpu()
                                ref_audio = vocoder.decode(ref_mel_spec).cpu()
                            elif self.vocoder_name == "bigvgan":
                                gen_audio = vocoder(gen_mel_spec).squeeze(0).cpu()
                                ref_audio = vocoder(ref_mel_spec).squeeze(0).cpu()
                        torchaudio.save(
                            f"{log_samples_path}/update_{global_update}_gen.wav", gen_audio, target_sample_rate
                        )
                        torchaudio.save(
                            f"{log_samples_path}/update_{global_update}_ref.wav", ref_audio, target_sample_rate
                        )
                        
                        # Generate fixed reference samples
                        self.generate_reference_samples(global_update, vocoder, nfe_step, cfg_strength, sway_sampling_coef)
                        
                        self.model.train()
                
                if global_update % self.last_per_updates == 0 and self.accelerator.sync_gradients:
                    self.save_checkpoint(global_update, last=True)  
                    
        self.save_checkpoint(global_update, last=True)

        self.accelerator.end_training()