<p align="left">
  <img src="https://cdn-uploads.huggingface.co/production/uploads/63d8d8879dfcfa941d4d7cd9/GsQKdaTyn2FFx_cZvVHk3.png" alt="Logo">
</p>

# UPDATE: 

## Streaming
```
# pip install fastapi uvicorn piper-tts langchain underthesea vinorm
# Edit f5tts-fastapi-server.py and change model checkpoint, reference audio and reference text (as many as you like and name them well), then simply call:

python f5tts-fastapi-server.py <-- check port inside this file
```
From browser call the host/port and you are up and running.

## Celebrating the 50th Anniversary of Reunification Day (April 30, 1975 - April 30, 2025): Highly enhanced source codes are released [(here)](https://github.com/EraX-AI/viF5TTS/tree/main/src) ✨✨✨

## EraX-Smile-UnixSex-F5: Unisex is now available at https://huggingface.co/erax-ai/EraX-Smile-UnixSex-F5 ✨✨✨

---------------
# EraX-Smile-Female-F5-V1.0: Giving F5-TTS a Vietnamese Twist (with Online Zero-Shot Voice Cloning!) ✨

Hey there, fellow Vietnamese AI explorers! 👋

We introduce **EraX-Smile-Female-F5-V1.0**, a Vietnamese text-to-speech model developed based on the F5-TTS architecture [arXiv:2410.06885](https://arxiv.org/abs/2410.06885).
To adapt this model for Vietnamese, we utilized a substantial dataset combining over 800,000 samples, of which some are from public repositories and with an extensive 500-hour private dataset, for which we gratefully acknowledge obtaining usage rights. 
The model underwent significant training, involving approximately 1 million update steps on a 4x RTX 3090 configuration. It tooks almost a week with some crashes and burns too 🔥

Our hope is that EraX-Smile-Female-F5-V1.0 (soon UniSex) proves to be a useful contribution to the community for ethical and creative purposes.

## Does it actually work? Let's listen! 🎧

Okay, moment of truth. Here's a sample voice we fed into the model (the "reference"):

# EraX-Smile-Female-F5-V1.0

## Reference Audio (< 15s) & Text
**Reference Audio:** [Download and play reference audio](https://huggingface.co/erax-ai/EraX-Smile-Female-F5-V1.0/resolve/main/update_213000_ref.wav)

**Reference Text:**
> *"Thậm chí không ăn thì cũng có cảm giác rất là cứng bụng, chủ yếu là cái phần rốn...trở lên. Em có cảm giác khó thở, và ngủ cũng không ngon, thường bị ợ hơi rất là nhiều"*

And here's our model trying its best to mimic that voice while reading completely different text. Fingers crossed! 🤞

## Text to Generate
> *"Trong khi đó, tại một chung cư trên địa bàn P.Vĩnh Tuy (Q.Hoàng Mai), nhiều người sống trên tầng cao giật mình khi thấy rung lắc mạnh nên đã chạy xuống sảnh tầng 1. Cư dân tại đây cho biết, họ chưa bao giờ cảm thấy ảnh hưởng của động đất mạnh như hôm nay"*

**Generated Audio:** [Download and play generated audio](https://huggingface.co/erax-ai/EraX-Smile-Female-F5-V1.0/resolve/main/generated_429000.wav)

## Attention: 
Just a gentle observation regarding the selected reference audio – it seems there might be a **noticeable pause** within it. Please be aware that the zero-shot cloning process will likely replicate this characteristic in the synthesized output. If a more continuous flow without pauses is desired, you might consider using a reference recording that is clean and free of significant delays, unless reproducing the pause is intentional.

## Audio Samples

If you'd like to listen to the audio samples directly:

1. **Reference Audio**: Download the [reference audio file](https://huggingface.co/erax-ai/EraX-Smile-Female-F5-V1.0/resolve/main/model/update_213000_ref.wav) and play it on your device.

2. **Generated Audio**: Download the [generated audio file](https://huggingface.co/erax-ai/EraX-Smile-Female-F5-V1.0/resolve/main/model/generated_429000.wav) and play it on your device.

Alternatively, you can visit our [Hugging Face model page](https://huggingface.co/erax-ai/EraX-Smile-Female-F5-V1.0) to access and play these audio files directly.

## Wanna try this magic (or madness) yourself? 🧙‍♂️

Getting started is hopefully not *too* painful. After downloading this repo and cloning our GitHub, you can try something like this:

```python
# Ubuntu: sudo apt install ffmpeg
# Windows please refer to https://www.geeksforgeeks.org/how-to-install-ffmpeg-on-windows/

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
    use_ema=False, # ALWAYS False as we converted from .pt to safetensors and EMA (where there is or not) was in there
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
* We use the cool library from [Vinorm Team](https://github.com/v-nhandt21/Vinorm) for Vietnamese text normalization.

* **What's Next?** 🤔
The EraX Team (that's us!) are always tinkering and trying to make things better (or at least, less broken!).
We hope to bring more updates your way. Let us know what you think – feedback, bug reports, or even just saying hi is always welcome!
- [ ] ⭐ Release checkpoints for Vietnamese male
- [ ] 📝 Codes for real-time TTS streaming
- [ ] 🔥 Release Piper-based model that can run on ...Rasberry Pi 4 🔥

⚠️ **Important Note on Responsible Use** ⚠️
- Voice cloning technology is powerful and comes with significant ethical responsibilities.
- Intended Use: This model is intended for creative purposes, accessibility tools, personal projects, and applications where consent is explicit and ethical considerations are prioritized.
- **Prohibited Use**: We strongly condemn and strictly prohibit the use of this model for any malicious or unethical purposes, including but not limited to:
  - Creating non-consensual deepfakes or impersonating individuals without permission.
  - Generating misinformation, fraudulent content, or defamatory material.
  - Harassment, abuse, or any form of criminal activity.
- User Responsibility: By using this model, you agree to do so responsibly and ethically. You are solely responsible for the content you generate and ensuring it complies with all applicable laws and ethical standards. The creators (EraX Team) disavow any responsibility for misuse of this model.

  Please use this technology thoughtfully and ethically.

⚠️ **Lưu ý Quan trọng về Việc Sử dụng có Trách nhiệm** ⚠️

- Sức mạnh và Trách nhiệm: Công nghệ nhân bản giọng nói sở hữu sức mạnh to lớn và đi kèm với những trách nhiệm đạo đức hết sức quan trọng.
- Mục đích Sử dụng Dự kiến: Mô hình này được tạo ra nhằm phục vụ các mục đích sáng tạo, phát triển công cụ hỗ trợ tiếp cận, thực hiện dự án cá nhân và các ứng dụng khác nơi có sự đồng thuận rõ ràng từ các bên liên quan và các yếu tố đạo đức luôn được đặt lên hàng đầu.
- Nghiêm cấm Sử dụng Sai trái: Chúng tôi cực lực lên án và nghiêm cấm tuyệt đối việc sử dụng mô hình này cho bất kỳ mục đích xấu xa, phi đạo đức nào, bao gồm nhưng không giới hạn ở:
  - Tạo ra deepfake hoặc mạo danh người khác khi chưa được sự cho phép hoặc đồng thuận rõ ràng.
  - Phát tán thông tin sai lệch, tạo nội dung lừa đảo hoặc các tài liệu mang tính phỉ báng, bôi nhọ.
  - Thực hiện hành vi quấy rối, lạm dụng hoặc bất kỳ hoạt động tội phạm nào khác.

- Trách nhiệm của Người dùng: Khi sử dụng mô hình này, bạn cam kết hành động một cách có trách nhiệm và tuân thủ các chuẩn mực đạo đức. Bạn phải chịu trách nhiệm hoàn toàn về nội dung do mình tạo ra và đảm bảo rằng nội dung đó tuân thủ mọi quy định pháp luật hiện hành và các tiêu chuẩn đạo đức. Đội ngũ phát triển (Nhóm EraX) hoàn toàn không chịu trách nhiệm cho bất kỳ hành vi lạm dụng nào đối với mô hình này.

  Lời kêu gọi: Xin hãy sử dụng công nghệ này một cách có suy xét, thận trọng và đạo đức.

**License Stuff** 📜
We're keeping it simple with the MIT License, following in the footsteps of giants like Whisper. Use it, break it, hopefully make cool things with it!

**Feeling Generous? (Citation)** 🙏
Did this model actually help you? Or maybe just provide a moment's amusement? If so, a star ⭐ on our GitHub repo would totally make our day!
Don't forget to like, share and star us on Github and HuggingFace!
And if you're writing something fancy (like a research paper) and want to give us a nod, here's the bibtex snippet:

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
