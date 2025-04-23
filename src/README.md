Okay, here is a professional English translation suitable for a GitHub README, incorporating the technical details and explanations you provided.

---

## Enhanced F5-TTS Fork

This repository is a significantly enhanced fork of the original F5-TTS project. We have introduced several major improvements and features, focusing on model efficiency, training stability, and output quality. Key additions include advanced pruning techniques, a sophisticated duration predictor, improved checkpointing, knowledge distillation capabilities, and the option to train models entirely from scratch.

### ✨ Model Pruning ✨

Optimize your model size and inference speed using our flexible pruning options. You can choose between automated, evenly distributed layer removal or manual, fine-grained block selection.

*   **Fully Automated Pruning:**
    Specify the total number of encoder/decoder blocks to keep. The pruner will select them evenly.
    ```bash
    python excellent_definitive-f5tts-pruner.py \
      --model_path "/path/to/your/base_model.safetensors" \
      --output_dir "/path/to/your/pruned_models" \
      --output_name "pruned_model_16_layers_auto.pt" \
      --layers_keep 16 # Keep 16 blocks, selected evenly
    ```

*   **Manual Block Selection:**
    Provide a comma-separated list of the specific block indices you wish to retain for precise control.
    ```bash
    python excellent_definitive-f5tts-pruner.py \
      --model_path "/path/to/your/base_model.safetensors" \
      --output_dir "/path/to/your/pruned_models" \
      --output_name "pruned_model_14_layers_manual.pt" \
      --manual_blocks "0,1,2,3,4,6,8,10,12,14,16,18,20,21" # Keep these specific blocks
    ```

**Note:** After pruning, proceed with vocabulary extension as needed for your specific setup. Please double-check file paths, as some may be hard-coded in scripts for rapid development and testing purposes.

### ✨ Sophisticated Training w/ Duration Predictor ✨

We have integrated a **Duration Predictor** module into the F5-TTS architecture. This component is trained alongside the main model to explicitly predict the duration of each input token (e.g., phoneme or character).

**Why is this important?** By learning token durations, the model gains much better control over the rhythm, pacing, and prosody of the generated speech.

**Benefits:** This significantly mitigates common TTS artifacts such as:
*   Skipped words or syllables
*   Unnatural pauses or hesitations ("uhm", "err" sounds)
*   Dropped words
*   Repetitive sounds or babbling
*   Mumbling or unclear articulation

The duration predictor adds a secondary loss component during training, controlled by `--duration_loss_weight`.

*   **Finetuning an Existing Model (with Duration Predictor):**
    Adapt a pre-trained (potentially pruned) model using your custom dataset.
    ```bash
    accelerate launch --mixed_precision=bf16 /path/to/f5_tts/train/finetune_cli.py \
      --exp_name MyFinetune_WithDuration \
      --model_path "/path/to/your/pruned_or_base_model.pt" \ # Specify model if not default
      --learning_rate 2e-4 --weight_decay 0.001 \
      --batch_size_per_gpu 16384 --batch_size_type frame \
      --max_samples 64 --grad_accumulation_steps 4 --max_grad_norm 1.0 \
      --use_duration_predictor \ # Enable the duration predictor
      --duration_loss_weight 0.2 \ # Set weight for duration loss
      --epochs 500 --num_warmup_updates 10000 \
      --save_per_updates 2000 --keep_last_n_checkpoints 50 --last_per_updates 2000 \
      --dataset_name your_dataset_name --tokenizer char \
      --logger tensorboard --log_samples \
      --finetune \ # Indicate finetuning mode
      --ref_audio_paths "/path/to/reference.wav" \
      --ref_texts "Reference text corresponding to the audio." \
      --ref_sample_text_prompts "Example prompt text for generating samples during training."
    ```

*   **Training From Scratch (with Duration Predictor):**
    Train a new model entirely from your dataset.
    ```bash
    accelerate launch --mixed_precision=bf16 /path/to/f5_tts/train/finetune_cli.py \
      --exp_name MyTrainingFromScratch_WithDuration \
      # No --model_path needed unless specifying a non-default config
      --learning_rate 2e-4 --weight_decay 0.001 \
      --batch_size_per_gpu 16384 --batch_size_type frame \
      --max_samples 64 --grad_accumulation_steps 4 --max_grad_norm 1.0 \
      --use_duration_predictor \ # Enable the duration predictor
      --duration_loss_weight 0.2 \ # Set weight for duration loss
      --epochs 500 --num_warmup_updates 10000 \
      --save_per_updates 2000 --keep_last_n_checkpoints 50 --last_per_updates 2000 \
      --dataset_name your_dataset_name --tokenizer char \
      --logger tensorboard --log_samples \
      --from_scratch \ # Indicate training from scratch
      --ref_audio_paths "/path/to/reference.wav" \
      --ref_texts "Reference text corresponding to the audio." \
      --ref_sample_text_prompts "Example prompt text for generating samples during training."
    ```

### Other Enhancements

Beyond pruning and the duration predictor, this fork includes:

*   **Improved Checkpoint Save/Load:** More robust mechanisms for managing training checkpoints.
*   **Knowledge Distillation:** Support for distilling knowledge from larger teacher models.
*   **Training From Scratch:** Added capability to initialize and train models without pre-trained weights.
*   Various stability improvements and other advanced features.

---

We hope these enhancements prove valuable for your text-to-speech projects. Good luck with your experiments, and we welcome any feedback or contributions!
Contact us for discussion on further collaboration at nguyen@hatto.com
