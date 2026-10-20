---
language:
  - hi
  - en
license: mit
task_categories:
  - automatic-speech-recognition
tags:
  - hinglish
  - code-switching
  - speech
  - hindi
size_categories:
  - 10K<n<100K
---

# Hinglish ASR Dataset

A dataset of **real-world Hinglish (Hindi–English code-switched) speech**
built for training and evaluating automatic speech recognition on the way
500M+ people in India actually speak — both languages interleaved in the
same sentence, where monolingual Hindi or English ASR degrades sharply.

## Dataset summary

- **Rows:** 40,231 (single `train` split as released)
- **Total size:** 27.8 GB of audio
- **Content:** real-world Hinglish audio clips collected from public
  sources (YouTube, podcasts) as part of the Hinglish ASR research
  project. The public release contains the audio; the reference
  transcripts produced by the project's Gemma-based labelling pipeline
  for training are not part of this release.

## Links

- **Models trained on this data:**
  [huggingface.co/hinglish-asr-dataset](https://huggingface.co/hinglish-asr-dataset)
  (7 fine-tunes, Whisper-base and Whisper-large-v3)
- **Best model:**
  [whisper-base-hinglish-1105-en-mix](https://huggingface.co/hinglish-asr-dataset/whisper-base-hinglish-1105-en-mix)
- **Code, evaluation harness, and documentation:**
  [github.com/bhartimadhav10/hinglish-asr](https://github.com/bhartimadhav10/hinglish-asr)

## What it was built for

- Fine-tuning Whisper-family models for code-switched speech — the
  models under this account (including
  `whisper-base-hinglish-1105-en-mix`, WER 16.46% vs a commercial
  baseline's 32.79% on a held-out set) were trained on this data.
- Benchmarking ASR systems on realistic, non-studio Hinglish audio.

## Considerations

- Speech was sourced from publicly available material for research on an
  underserved language population; users are responsible for ensuring
  their use complies with applicable terms and law.
- Real-world audio is noisy and unevenly distributed across speakers,
  topics, and recording conditions; expect domain shift versus clean
  studio corpora, and evaluate on held-out data before deployment.

## Author

Madhav Bharti — Hinglish ASR research project.
