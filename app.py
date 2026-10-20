#!/usr/bin/env python3
"""Gradio app for the Hinglish ASR model.

Used in two places without modification:
  * Locally            ->  python app.py
  * Hugging Face Space ->  this file is the Space entry point
                           (see space/README.md for the Space config)

Upload a file or record from the microphone; the model transcribes
Hindi-English code-switched speech on the server and returns the text.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import gradio as gr
import numpy as np

from hinglish_asr import HinglishASR

asr = HinglishASR()  # loads once at startup; device auto-selected


def transcribe(audio: tuple[int, np.ndarray] | None) -> str:
    if audio is None:
        return "Please upload or record an audio clip first."
    sample_rate, waveform = audio
    if waveform.size == 0:
        return "Empty audio received — please try again."
    result = asr.transcribe_array(
        waveform.astype(np.float32).flatten() / 32768.0
        if waveform.dtype == np.int16
        else waveform.astype(np.float32).flatten(),
        sample_rate,
    )
    return result.text


demo = gr.Interface(
    fn=transcribe,
    inputs=gr.Audio(sources=["upload", "microphone"], type="numpy", label="Hinglish audio"),
    outputs=gr.Textbox(label="Transcript", lines=4),
    title="Hinglish ASR — Whisper fine-tuned for Hindi-English code-switching",
    description=(
        "Speech recognition for the way 500M+ people in India actually talk: "
        "Hindi and English mixed in the same sentence. Model: "
        "[whisper-base-hinglish-1105-en-mix]"
        "(https://huggingface.co/hinglish-asr-dataset/whisper-base-hinglish-1105-en-mix) "
        "— a Whisper-base fine-tuned on a 40,231-clip Hinglish dataset. "
        "On a held-out test set it scores 16.46% WER vs 32.79% for a commercial "
        "baseline. Code, dataset and full benchmark: "
        "[github.com/maddycodes123/hinglish-asr]"
        "(https://github.com/maddycodes123/hinglish-asr) · "
        "[dataset on Hugging Face]"
        "(https://huggingface.co/datasets/hinglish-asr-dataset/hinglish-asr)."
    ),
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch()
