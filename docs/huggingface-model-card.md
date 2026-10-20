---
language:
  - hi
  - en
license: mit
base_model: openai/whisper-base
tags:
  - automatic-speech-recognition
  - whisper
  - hinglish
  - code-switching
  - hindi
metrics:
  - wer
  - cer
datasets:
  - hinglish-asr-dataset/hinglish-asr
pipeline_tag: automatic-speech-recognition
---

# Whisper Base — Hinglish (1105-en-mix)

A fine-tune of `openai/whisper-base` for **Hinglish** — Hindi–English
code-switched speech, the everyday spoken language of 500M+ people in
India that monolingual ASR systems handle poorly.

- **Parameters:** 72.6M · **Format:** safetensors (F32)
- **Generation config:** language `hi`, task `transcribe`, no timestamps
- **Code, scripts, and full documentation:**
  [github.com/bhartimadhav10/hinglish-asr](https://github.com/bhartimadhav10/hinglish-asr)
  (runnable CLI, evaluation harness, Gradio demo, full benchmark data)
- **Dataset:**
  [hinglish-asr-dataset/hinglish-asr](https://huggingface.co/datasets/hinglish-asr-dataset/hinglish-asr)

## Training data

Fine-tuned on [hinglish-asr-dataset/hinglish-asr](https://huggingface.co/datasets/hinglish-asr-dataset/hinglish-asr)
— 40,231 real-world Hinglish audio clips (27.8 GB) collected from public
sources with an automated Gemma-based labelling pipeline.

## Evaluation

Held-out test sets (WER / CER in %, lowercase + punctuation-stripped
normalisation). Baseline: Oriserve Swift (commercial ASR).

| Test set | n | This model WER | This model CER | Oriserve WER | Oriserve CER |
|---|---:|---:|---:|---:|---:|
| test_gemma_dataset | 1,283 | **16.46** | **8.05** | 32.79 | 18.06 |

The fine-tuned model roughly **halves the commercial baseline's WER**
on held-out Hinglish speech. Raw evaluation artefact:
`checkpoint-1356/plots/compare_whisper_hinglish_1105_vs_oriserve.json`
in this repository; machine-readable copy and reproduction scripts in
the [GitHub repository](https://github.com/bhartimadhav10/hinglish-asr).

## Usage

```python
from transformers import WhisperForConditionalGeneration, WhisperProcessor
import soundfile as sf

model_id = "hinglish-asr-dataset/whisper-base-hinglish-1105-en-mix"
processor = WhisperProcessor.from_pretrained(model_id)
model = WhisperForConditionalGeneration.from_pretrained(model_id)

audio, sr = sf.read("audio.wav")          # 16 kHz mono expected
inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
ids = model.generate(inputs.input_features)   # uses the model's generation config
print(processor.batch_decode(ids, skip_special_tokens=True)[0])
```

A maintained, packaged version of this snippet (batch files, folders,
resampling, long-audio windowing, GPU support) lives in the project's
GitHub repository as `transcribe.py`.

## Limitations

Noisy/far-field speech remains difficult; long-form audio is processed
in 30 s windows without timestamps or diarisation; the configuration
targets Hindi-anchored Hinglish rather than English-dominant speech.

## Author

Madhav Bharti — research on ASR for underserved, code-switched language
populations.
