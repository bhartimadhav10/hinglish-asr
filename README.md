# Hinglish ASR

**Speech recognition for the way India actually talks — Hindi and English mixed in the same sentence.**

A Whisper-base model fine-tuned on Hinglish (Hindi–English code-switched) speech, packaged with everything needed to run it, evaluate it, and demo it:

- 🖥️ **On-device transcription CLI** — runs fully offline after a one-time download (CPU or GPU)
- 📊 **A reproducible WER/CER evaluation harness** — local manifests or the public dataset
- 🎙️ **A Gradio web demo** — the same file runs locally and as a Hugging Face Space
- 📈 **Published benchmark results** — **16.46% WER** vs 32.79% for a commercial ASR baseline on a held-out test set of 1,283 clips

| | |
|---|---|
| **Model** | [hinglish-asr-dataset/whisper-base-hinglish-1105-en-mix](https://huggingface.co/hinglish-asr-dataset/whisper-base-hinglish-1105-en-mix) |
| **Base model** | openai/whisper-base |
| **Parameters** | 72.6M (F32, safetensors) |
| **Dataset** | [hinglish-asr-dataset/hinglish-asr](https://huggingface.co/datasets/hinglish-asr-dataset/hinglish-asr) — 40,231 clips, 27.8 GB |
| **Task / language** | Transcription, Hindi (`hi`) forced by the model's generation config, no timestamps |
| **Author** | Madhav Bharti — [LinkedIn](https://www.linkedin.com/in/madhav-bharti-09306b263/) · [GitHub](https://github.com/bhartimadhav10) |

## 🔗 Everything is connected

| What | Where |
|---|---|
| 🤗 **Model weights + model card** | [huggingface.co/hinglish-asr-dataset/whisper-base-hinglish-1105-en-mix](https://huggingface.co/hinglish-asr-dataset/whisper-base-hinglish-1105-en-mix) |
| 🤗 **Dataset (40,231 clips)** | [huggingface.co/datasets/hinglish-asr-dataset/hinglish-asr](https://huggingface.co/datasets/hinglish-asr-dataset/hinglish-asr) |
| 🤗 **All 7 fine-tuned models** | [huggingface.co/hinglish-asr-dataset](https://huggingface.co/hinglish-asr-dataset) |
| 🎙️ **Live demo (HF Space)** | Create from the [`space/`](space/) folder in this repo — it will live at `huggingface.co/spaces/hinglish-asr-dataset/hinglish-asr` (steps in [`space/README.md`](space/README.md)) |
| 💻 **This code repository** | [github.com/bhartimadhav10/hinglish-asr](https://github.com/bhartimadhav10/hinglish-asr) |
| 📄 **Paste-ready HF cards** | [`docs/huggingface-model-card.md`](docs/huggingface-model-card.md) · [`docs/huggingface-dataset-card.md`](docs/huggingface-dataset-card.md) |

**The flow:** the model card on Hugging Face links here → this README's quickstart gets you transcribing in minutes → `evaluate.py` reproduces the benchmark → the Space demo lets anyone try the model in a browser. Every page points to the others, so a visitor landing anywhere finds everything.

---

## Why Hinglish ASR is a real problem

500M+ people in India speak Hinglish — Hindi grammar and vocabulary interleaved with English words, often mid-sentence: *"Maine kal meeting mein bola tha ki file upload kar dunga."* Mainstream ASR systems are trained on monolingual Hindi **or** English and degrade sharply on code-switched speech: they force one language, mistranscribe the other, and produce output no downstream system can use. This project attacks that gap directly with a purpose-built dataset and fine-tunes, rather than prompting a general model and hoping.

## Benchmark results

Held-out test sets, scored with lowercase + punctuation-stripped normalisation (WER / CER, %; lower is better). Baseline: **Oriserve Swift**, a commercial ASR service. Full data in [`results/benchmark_results.json`](results/benchmark_results.json); evaluation artefacts are published inside the model repository on Hugging Face (`checkpoint-1356/plots/`).

| Test set | n | This model WER | This model CER | Oriserve Swift WER | Oriserve Swift CER |
|---|---:|---:|---:|---:|---:|
| test_gemma_dataset | 1,283 | **16.46** | **8.05** | 32.79 | 18.06 |

On this held-out set the fine-tuned 72.6M model roughly **halves the word error rate of the commercial Oriserve Swift ASR service** (16.46% vs 32.79% WER; 8.05% vs 18.06% CER) — a base-sized open model outperforming a paid production API on real-world code-switched speech. Machine-readable results: [`results/benchmark_results.json`](results/benchmark_results.json). Reproduce the methodology on your own data with `evaluate.py` (below).

> ✅ **Verified end-to-end (2026-10-21):** every script in this repo was executed against the published model on a CPU-only machine — 8 real clips streamed from the public dataset, transcribed at a mean RTF of ≈ 0.25 (≈ 4× faster than real time, no GPU). Unedited outputs: [`examples/sample_transcripts.md`](examples/sample_transcripts.md) · full record: [`results/local_verification.json`](results/local_verification.json).

## Quickstart

### 1. Install

```bash
git clone https://github.com/bhartimadhav10/hinglish-asr.git
cd hinglish-asr

# CPU-only PyTorch (enough for this 72.6M model):
pip install --index-url https://download.pytorch.org/whl/cpu torch
# ...or install a CUDA build of torch for GPU use.

pip install -e .            # core
pip install -e ".[eval]"   # + evaluation harness
pip install -e ".[demo]"   # + Gradio demo
```

### 2. Transcribe on your own device

```bash
python transcribe.py interview.wav
python transcribe.py call1.wav call2.mp3 --output transcripts.tsv
python transcribe.py --input-dir ./clips --output-dir ./transcripts
python transcribe.py long_call.wav --device cuda
```

The model (~290 MB) downloads once from Hugging Face and is cached; transcription after that is fully offline. Audio is resampled to 16 kHz automatically and long files are processed in 30-second windows.

### 3. Use it from Python

```python
from hinglish_asr import HinglishASR

asr = HinglishASR()                       # auto-selects GPU if available
result = asr.transcribe_file("audio.wav")
print(result.text)
print(f"{result.audio_seconds:.1f}s of audio in {result.processing_seconds:.1f}s "
      f"(RTF {result.real_time_factor:.3f})")
```

### 4. Web demo (local or Hugging Face Space)

```bash
python app.py        # Gradio UI: upload a file or record from the mic
```

To publish the same demo as a Hugging Face Space, follow [`space/README.md`](space/README.md) — the Space front-matter and file list are prepared for you.

### 5. Evaluate (reproduce the numbers)

```bash
# Labelled data you control (JSONL: {"audio": "...", "text": "..."} per line)
python evaluate.py --manifest data/test.jsonl --limit 200

# Any Hugging Face dataset that has audio + transcript columns
python evaluate.py --dataset <org>/<dataset> \
    --audio-column audio --text-column text --limit 50
```

> Note: the public release of `hinglish-asr-dataset/hinglish-asr` is
> **audio-only** (reference transcripts are not published with it), so it
> cannot be scored directly — `evaluate.py` detects this and explains it.
> The published benchmark above was produced on held-out labelled test
> sets with this same harness and normalisation.

## How it works

```
audio file ──► load + mono mixdown ──► resample to 16 kHz
          ──► 30 s windows ──► log-Mel features (WhisperProcessor)
          ──► WhisperForConditionalGeneration.generate()
               (model's own generation config: <|hi|> · transcribe · no timestamps)
          ──► token decode ──► transcript
```

All entry points (`transcribe.py`, `evaluate.py`, `app.py`) share one implementation — `src/hinglish_asr/model.py` — so local, evaluation, and Space behaviour cannot drift apart. The language override (`--language`) exists for experiments but is **off by default**: the published benchmarks were produced with the model's own configuration, and that is what runs here.

## Repository structure

```
.
├── README.md                  # this file
├── transcribe.py              # on-device CLI
├── evaluate.py                # WER/CER harness (manifest or HF dataset)
├── app.py                     # Gradio demo (local + HF Space entry point)
├── src/hinglish_asr/
│   ├── model.py               # shared loader + transcription logic
│   └── __init__.py
├── results/
│   ├── benchmark_results.json # published held-out benchmark vs Oriserve Swift
│   └── local_verification.json# the 2026-10-21 end-to-end run of this repo's code
├── examples/
│   └── sample_transcripts.md  # unedited model outputs on public-dataset clips
├── space/README.md            # Hugging Face Space config + deploy steps
├── docs/
│   ├── huggingface-model-card.md   # paste-ready card for the HF model page
│   └── huggingface-dataset-card.md # paste-ready card for the HF dataset page
├── requirements.txt
├── pyproject.toml
└── LICENSE                    # MIT (code). Weights/dataset: see their HF pages.
```

## Training summary

- **Starting point:** `openai/whisper-base`, fine-tuned on the project's Hinglish corpus ([dataset on HF](https://huggingface.co/datasets/hinglish-asr-dataset/hinglish-asr)).
- **Data pipeline:** real-world Hinglish audio collected from public sources (YouTube, podcasts) with an automated Gemma-based labelling pipeline; the released dataset contains 40,231 clips (27.8 GB).
- **Lineage:** this `1105-en-mix` variant is the best-performing of a family of fine-tunes (Whisper-base and Whisper-large-v3 variants) published under the same HF account; intermediate training checkpoints (to step 4746) are preserved in the model repository, and the exact run configuration is in `training_args.bin` there.

## Limitations

- Noisy, far-field, or heavily accented speech is harder than the clean speech in the benchmark set.
- Long-form audio is transcribed in independent 30 s windows without timestamps or speaker diarisation.
- The generation config is tuned for Hindi-anchored Hinglish; primarily-English speech with occasional Hindi words is out of the primary distribution.
- No streaming/low-latency mode; this is an offline/batch model.

## Troubleshooting

- **`torch` install is huge / CUDA errors** — install the CPU build exactly as in the Quickstart (`--index-url https://download.pytorch.org/whl/cpu`). The model is 72.6M parameters; CPU inference is practical (see the RTF printed by `transcribe.py`).
- **First run is slow** — the ~290 MB model downloads once into the Hugging Face cache (`~/.cache/huggingface`); later runs are offline and fast.
- **mp3/m4a won't load** — `soundfile` covers wav/flac/ogg; for mp3/m4a also `pip install librosa` (the loader falls back to it automatically).
- **Weird output on pure-English audio** — expected: the model's generation config forces Hindi transcription for Hinglish. Experiment with `python transcribe.py file.wav --language en`.

## Responsible use

The dataset was curated from publicly available speech for research on an underserved language population. The model is intended for transcription of speech its users have the right to process; do not use it for surveillance of private conversations.

## Citation

```bibtex
@misc{bharti2026hinglishasr,
  title  = {Hinglish ASR: Fine-tuned Whisper for Hindi-English Code-Switched Speech},
  author = {Bharti, Madhav},
  year   = {2026},
  url    = {https://huggingface.co/hinglish-asr-dataset/whisper-base-hinglish-1105-en-mix}
}
```

## License

Code in this repository: MIT (see `LICENSE`). Model weights and dataset are distributed via Hugging Face under the terms stated on their respective pages; the base model is OpenAI Whisper.
