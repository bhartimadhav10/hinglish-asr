---
title: Hinglish ASR
emoji: 🎙️
colorFrom: yellow
colorTo: red
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: mit
models:
  - hinglish-asr-dataset/whisper-base-hinglish-1105-en-mix
---

# Hinglish ASR — Hugging Face Space

This folder is the Hugging Face **Space** deployment of the Hinglish ASR demo.

Connected links (also shown inside the Space once deployed):
- Model: https://huggingface.co/hinglish-asr-dataset/whisper-base-hinglish-1105-en-mix
- Dataset: https://huggingface.co/datasets/hinglish-asr-dataset/hinglish-asr
- Code + full docs: https://github.com/bhartimadhav10/hinglish-asr

## Deploy it

1. Create a new Space on huggingface.co (SDK: Gradio).
2. Copy this folder's contents to the Space root, together with:
   - `app.py` (from the repository root)
   - `src/` (the shared model package)
   - `requirements.txt`
3. The block above is the Space's `README.md` front-matter — keep it at the
   top of the Space's README.
4. The Space installs `requirements.txt`, loads
   `hinglish-asr-dataset/whisper-base-hinglish-1105-en-mix`, and serves the
   microphone/upload demo. The base-sized model (72.6M params) runs
   comfortably on a free CPU Space.

## Run it locally instead

```bash
pip install --index-url https://download.pytorch.org/whl/cpu torch
pip install -r requirements.txt
python app.py
```
