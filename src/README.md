<p align="center">
  <img src="https://cdn-uploads.huggingface.co/production/uploads/63d8d8879dfcfa941d4d7cd9/GsQKdaTyn2FFx_cZvVHk3.png" alt="Logo">
</p>

# Updated May 3, 2025: Complete overhaul of training methodology with phoneme-based duration prediction for superior speech quality

# Introduce an Highly Enhanced F5-TTS fully open source.

**MIT license** for everything except if you use the **pretrained F5-TTS model**, then you are bound with Emilia dataset which is under **BY-NC 4.0 license**. 

**Celebrating the 50th Anniversary of Reunification Day (April 30, 1975 - April 30, 2025) 🇻🇳 🇻🇳 🇻🇳** 

This repository is a significantly enhanced fork of the original F5-TTS project. We have introduced several major improvements and features, focusing on model efficiency, training stability, and output quality. Key additions include advanced pruning techniques, a sophisticated duration predictor, improved checkpointing, knowledge distillation capabilities, and the option to train models entirely from scratch.

I'll rewrite that specific section of your document in a more professional style for a model card. Here's a polished English version:

## Advanced Training Strategy with Phoneme-Based Duration Prediction

Our implementation fundamentally improves upon the original F5-TTS architecture by introducing a phoneme-level duration predictor that significantly enhances naturalness and articulation clarity.

### Phoneme Preprocessing Pipeline

1. **Phonemization Setup**:
   - The system utilizes an enhanced phonemizer library with espeak language support to convert text into phonemic representations
   - All preprocessed phonemes are stored in a structured `phonemes_metadata.jsonl` file

2. **Prerequisites**:
   ```bash
   # Install required system dependencies
   sudo apt-get install festival espeak-ng mbrola
   
   # Use our recommended high-performance phonemizer fork
   git clone https://github.com/thewh1teagle/phonemizer-fork.git
   cd phonemizer-fork
   pip install .
   ```

3. **Generating Phoneme Data**:
   ```bash
   # Change path to your metadata.csv and path to output phonemes_metadata.jsonl
   # Execute preprocessing script
   python /path/to/f5_tts/model/preprocess_phoneme.py
   ```
   This process creates a comprehensive `phonemes_metadata.jsonl` file in your designated data directory. Run F5-TTS Gradio to extend vocab/prepare data (validate audio files and generate `duration.json` & `raw.arrow`) normally, the new codes will leverage this `phonemes_metadata.jsonl` instead of origin `metadata.csv` file.

### Optimized Training Approach

Our training strategy implements a two-phase process:

1. **Duration Predictor Warm-up Phase**: 
   - Sử dụng Dynamic Programming Window algo to calculate monotonic alignment to balance training speed & quality. You can also try Viterbi Vectorized (also fast but lower quality) or Progressive Refinement (a bit higher quality ut slower).  
   - Initial 3 epochs focus on training the duration predictor while freezing audio model parameters
   - This approach ensures stability and prevents initial training divergence

2. **Full Model Fine-tuning**:
   - Following warm-up, all model parameters are unfrozen for comprehensive optimization
   - A dedicated TensorBoard dashboard tracks duration loss metrics alongside other model performance indicators

3. **Model Persistence**:
   - The duration predictor is saved as a separate model component for portability
   - This allows for flexible reuse across different model configurations

### Example Training Configuration with 1.2 million/female-only samples w/ only 2xRTX3090 24G GPUs. The model was pruned from 22 to 12 layers and then trained from scratch.

```bash
accelerate launch --mixed_precision=bf16 /path/to/f5_tts/train/finetune_cli.py \
  --exp_name F5TTS_Phoneme_Duration_Model \
  --learning_rate 2e-4 \
  --weight_decay 0.001 \
  --batch_size_per_gpu 16384 \
  --batch_size_type frame \
  --max_samples 64 \
  --grad_accumulation_steps 4 \
  --max_grad_norm 1.0 \
  --duration_focus_updates 12000 \
  --duration_focus_weight 1.5 \
  --use_duration_predictor \
  --duration_loss_weight 0.5 \
  --epochs 300 \
  --num_warmup_updates 32000 \
  --save_per_updates 3000 \
  --keep_last_n_checkpoints 50 \
  --last_per_updates 3000 \
  --dataset_name vietnamese_tts_dataset \
  --tokenizer char \
  --logger tensorboard \
  --log_samples \
  --from_scratch \
  --ref_audio_paths "/path/to/reference/samples/female-reference.wav" \
  --ref_texts "Phoneme training reference text with natural pronunciation patterns." \
  --ref_sample_text_prompts "Example text for evaluating model performance during training."
```

This phoneme-based duration prediction approach resolves common TTS artifacts including word skipping, unnatural pauses, and unclear articulation, resulting in significantly more natural speech synthesis.

### ✨ Model Pruning ✨

Optimize your model size and inference speed using our flexible pruning options. You can choose between automated, evenly distributed layer removal or manual, fine-grained block selection.

*   **Fully Automated Pruning:**
    Specify the total number of encoder/decoder blocks to keep. The pruner will select them evenly.
    ```bash
    python excellent_definitive-f5tts-pruner.py \
      --model_path "/path/to/your/base_model.safetensors" \
      --output_dir "/path/to/your/pruned_models" \
      --output_name "pruned_model_16_layers_auto.pt" \
      --target_layers 16 # Keep 16 blocks, selected evenly
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

*   **Finetuning an Existing Model (with 🇻🇳 Duration Predictor 🇻🇳):**
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

*   **Training From Scratch (with 🇻🇳 Duration Predictor 🇻🇳):**
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

### ✨ Sophisticated 🇻🇳 Distillation 🇻🇳 w/ Duration Predictor ✨
```
accelerate launch --mixed_precision=bf16 /root/18April/F5-TTS/src/f5_tts/train/distil_reload.py \
--teacher_ckpt_path /root/18April/Model_Pruning/full_finetune_ckpt/models/model_42000.safetensors \ <--- teacher model 
--student_exp_name F5TTS_v1_Custom_Prune_12 \
--student_init_ckpt_path /root/18April/F5-TTS/ckpts/steve_combined_multi/model_last.pt \ <-- student model (prune or origin)
--dataset_name steve_combined_multi \
--output_dir /root/18April/F5-TTS/ckpts/steve_combined_multi \
--distill_loss_weight 0.5 \
--distill_loss_type mse \
--use_duration_predictor \
--duration_loss_weight 0.25 \
--learning_rate 2e-5 \
--weight_decay 0.01 \
--batch_size_per_gpu 16384 \
--batch_size_type frame \
--max_samples 128 \
--grad_accumulation_steps 1 \
--max_grad_norm 1.0 \
--epochs 200 \
--num_warmup_updates 20000 \
--save_per_updates 2000 \
--last_per_updates 2000 \
--keep_last_n_checkpoints 50 \
--log_samples \
--logger tensorboard \
--logging_dir /root/18April/F5-TTS/ckpts/steve_combined_multi/runs \
--use_ema \
--ema_decay 0.9999 \
--tokenizer char \
--tokenizer_path /root/18April/F5-TTS/data/steve_combined_multi_char/vocab.txt \
--ref_audio_paths "/root/18April/Model_Pruning/female-vts.wav" \
--ref_texts "ai đã đến hàng dương , đều không thể cầm lòng về những nấm mộ chen nhau , nhấp nhô trải khắp một vùng đồi . những nấm mộ có tên và không tên , nhưng nấm mộ lấp ló trong lùm cây , bụi cỏ ." \
--ref_sample_text_prompts "sáng mười tám tháng bốn , cơ quan chức năng quảng ninh cho biết hiện cơ quan cảnh sát điều tra công an tỉnh quảng ninh đang tiếp tục truy bắt bùi đình khánh , ba mươi mốt tuổi , tay buôn ma túy đã xả súng làm một chiến sĩ công an hi sinh ."
```
### Streaming
```
# pip install fastapi uvicorn piper-tts langchain underthesea vinorm
# Edit f5tts-fastapi-server.py and change model checkpoint, reference audio and reference text (as many as you like and name them well), then simply call:

python f5tts-fastapi-server.py <-- check port inside this file
```
From browser call the host/port and you are up and running.

### Other Enhancements

Beyond pruning and the duration predictor, this fork includes:

*   **Improved Checkpoint Save/Load:** More robust mechanisms for managing training checkpoints.
*   **Knowledge Distillation:** Support for distilling knowledge from larger teacher models.
*   **Training From Scratch:** Added capability to initialize and train models without pre-trained weights.
*   Various stability improvements and other advanced features.

---

We hope these enhancements prove valuable for your text-to-speech projects. Good luck with your experiments, and we welcome any feedback or contributions!
Contact us for discussion on further collaboration at nguyen@hatto.com
