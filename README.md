---
license: mit
language:
- vi
base_model:
- SWivid/F5-TTS
pipeline_tag: text-to-speech
widget:
- src: "https://huggingface.co/erax-ai/EraX-Smile-Female-F5-V1.0/resolve/main/model/update_213000_ref.wav"
  text: "Thậm chí không ăn thì cũng có cảm giác rất là cứng bụng, chủ yếu là cái phần rốn...trở lên. Em có cảm giác khó thở, và ngủ cũng không ngon, thường bị ợ hơi rất là nhiều"
- src: "https://huggingface.co/erax-ai/EraX-Smile-Female-F5-V1.0/resolve/main/model/generated_non_ema.wav"
  text: "Trong khi đó, tại một chung cư trên địa bàn P.Vĩnh Tuy (Q.Hoàng Mai), nhiều người sống trên tầng cao giật mình khi thấy rung lắc mạnh nên đã chạy xuống sảnh tầng 1. Cư dân tại đây cho biết, họ chưa bao giờ cảm thấy ảnh hưởng của động đất mạnh như hôm nay"
---

<p align="left">
  <img src="https://cdn-uploads.huggingface.co/production/uploads/63d8d8879dfcfa941d4d7cd9/GsQKdaTyn2FFx_cZvVHk3.png" alt="Logo">
</p>

# EraX-Smile-Female-F5-V1.0: Giving F5-TTS a Vietnamese Twist (with Online one-shot Voice Cloning!) ✨

Hey there, fellow Vietnamese AI explorers! 👋

We took the rather clever F5-TTS model, with ++800,000 samples from public repository and from a huge 500h private dataset whom was kindly giving us a right to use it. 
We gave it a nudge towards Vietnamese TTS, and sprinkled in some voice cloning capabilities because... well, why not? We're calling this little experiment **EraX-Smile-Female-F5-V1.0**. We hope it brings a smile (or at least doesn't make you frown *too* much).

## Does it actually work? Let's listen! 🎧

Okay, moment of truth. Here's a sample voice we fed into the model (the "reference"):

*   **Reference Audio & Text:**
    <audio controls src="https://huggingface.co/erax-ai/EraX-Smile-Female-F5-V1.0/resolve/main/model/update_213000_ref.wav"></audio>
    > *"Thậm chí không ăn thì cũng có cảm giác rất là cứng bụng, chủ yếu là cái phần rốn...trở lên. Em có cảm giác khó thở, và ngủ cũng không ngon, thường bị ợ hơi rất là nhiều"*

And here's our model trying its best to mimic that voice while reading completely different text. Fingers crossed! 🤞

*   **Text to Generate:**
    > *"Trong khi đó, tại một chung cư trên địa bàn P.Vĩnh Tuy (Q.Hoàng Mai), nhiều người sống trên tầng cao giật mình khi thấy rung lắc mạnh nên đã chạy xuống sảnh tầng 1. Cư dân tại đây cho biết, họ chưa bao giờ cảm thấy ảnh hưởng của động đất mạnh như hôm nay"*

*   **Generated Audio:**
    <audio controls src="https://huggingface.co/erax-ai/EraX-Smile-Female-F5-V1.0/resolve/main/model/generated_non_ema.wav"></audio>

## Wanna try this magic (or madness) yourself? 🧙‍♂️

The code that wrangles this thing lives over on our GitHub: ([EraX Smile Github](https://github.com/EraX-JS-Company/EraX-Smile-F5TTS)). Give it a visit!

Getting started is hopefully not *too* painful. After downloading this repo and cloning our GitHub, you can try something like this:

```python
import os
os.environ["CUDA_VISIBLE_DEVICES"] =  "0" # Tell it which GPU to use (or ignore if you're CPU-bound and patient!)

from vinorm import TTSnorm # Gotta normalize that Vietnamese text first
from f5tts_wrapper import F5TTSWrapper # Our handy wrapper class

# --- Config ---
# Path to the model checkpoint you downloaded from *this* repo
# MAKE SURE this path points to the actual .pth or .ckpt file!
eraX_ckpt_path = "path/to/your/downloaded/EraX-Smile-Female-F5-V1.0/model.pth" # <-- CHANGE THIS!

# Path to the voice you want to clone
ref_audio_path = "path/to/your/reference_voice.wav" # <-- CHANGE THIS!

# Path to the vocab file from this repo
vocab_file = "path/to/your/downloaded/EraX-Smile-Female-F5-V1.0/vocab.txt" # <-- CHANGE THIS!

# Where to save the generated sound
output_dir = "output_audio"

# --- Texts ---
# Text matching the reference audio (helps the model learn the voice). Please make sure it match with the referrence audio!
ref_text = "Thậm chí không ăn thì cũng có cảm giác rất là cứng bụng, chủ yếu là cái phần rốn...trở lên. Em có cảm giác khó thở, và ngủ cũng không ngon, thường bị ợ hơi rất là nhiều"

# The text you want the cloned voice to speak
text_to_generate = "Trong khi đó, tại một chung cư trên địa bàn P.Vĩnh Tuy (Q.Hoàng Mai), nhiều người sống trên tầng cao giật mình khi thấy rung lắc mạnh nên đã chạy xuống sảnh tầng 1. Cư dân tại đây cho biết, họ chưa bao giờ cảm thấy ảnh hưởng của động đất mạnh như hôm nay."

# --- Let's Go! ---
print("Initializing the TTS engine... (Might take a sec)")
tts = F5TTSWrapper(
    vocoder_name="vocos", # Using Vocos vocoder
    ckpt_path=eraX_ckpt_path,
    vocab_file=vocab_file,
    use_ema=False, # Set True if you trained *with* EMA and want to use those weights
)

# Normalize the reference text (makes it easier for the model)
ref_text_norm = TTSnorm(ref_text)

# Prepare the output folder
os.makedirs(output_dir, exist_ok=True)

print("Processing the reference voice...")
# Feed the model the reference voice ONCE
# Provide ref_text for better quality, or set ref_text="" to use Whisper for auto-transcription (if installed)
tts.preprocess_reference(
    ref_audio_path=ref_audio_path,
    ref_text=ref_text_norm,
    clip_short=True # Keeps reference audio to a manageable length (~12s)
)
print(f"Reference audio duration used: {tts.get_current_audio_length():.2f} seconds")

# --- Generate New Speech ---
print("Generating new speech with the cloned voice...")

# Normalize the text we want to speak
text_norm = TTSnorm(text_to_generate)

# You can generate multiple sentences easily
# Just add more normalized strings to this list
sentences = [text_norm]

for i, sentence in enumerate(sentences):
    output_path = os.path.join(output_dir, f"generated_speech_{i+1}.wav")

    # THE ACTUAL GENERATION HAPPENS HERE!
    tts.generate(
        text=sentence,
        output_path=output_path,
        nfe_step=20,               # Denoising steps. More = slower but potentially better? (Default: 32)
        cfg_strength=2.0,          # How strongly to stick to the reference voice style? (Default: 2.0)
        speed=1.0,                 # Make it talk faster or slower (Default: 1.0)
        cross_fade_duration=0.15,  # Smooths transitions if text is split into chunks (Default: 0.15)
    )
    print(f"Boom! Audio saved to: {output_path}")

print("\nAll done! Check your output folder.")
```

* For full Web interface and control with Gradio, please clone and use the original repository of [F5-TTS Github](https://github.com/SWivid/F5-TTS)
* We use the cool library from [Vnorm Team](https://github.com/v-nhandt21/Vinorm) for Vietnamese text normalization.
  
```bibtex
@misc{EraXSmileF5_2024,
  author       = {Nguyễn Anh Nguyên and The EraX Team},
  title        = {EraX-Smile-Female-F5-V1.0: Người Việt sành tiếng Việt.},
  year         = {2024},
  publisher    = {Hugging Face},
  journal      = {Hugging Face Model Hub},
  howpublished = {\url{https://github.com/EraX-JS-Company/EraX-Smile-F5TTS}}
}
```
