"""Shared model loading and transcription logic for Hinglish ASR.

Single source of truth used by every entry point in this repository
(`transcribe.py`, `evaluate.py`, and the Gradio `app.py`), so the model
behaves identically on a local device and on a Hugging Face Space.

Model: hinglish-asr-dataset/whisper-base-hinglish-1105-en-mix
A Whisper-base (72.6M params) fine-tuned on Hinglish (Hindi-English
code-switched) speech. The model's own generation config forces
Hindi transcription with no timestamps; that default is respected
unless a caller explicitly overrides the language.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

MODEL_ID = "hinglish-asr-dataset/whisper-base-hinglish-1105-en-mix"
TARGET_SAMPLE_RATE = 16000

# Whisper processes audio in 30-second windows.
_CHUNK_SECONDS = 30


@dataclass
class TranscriptionResult:
    text: str
    audio_seconds: float
    processing_seconds: float

    @property
    def real_time_factor(self) -> float:
        """Processing time per second of audio (lower is faster)."""
        if self.audio_seconds <= 0:
            return 0.0
        return self.processing_seconds / self.audio_seconds


class HinglishASR:
    """Thin, well-defined wrapper around the fine-tuned Whisper model."""

    def __init__(
        self,
        model_id: str = MODEL_ID,
        device: str | None = None,
        language: str | None = None,
    ) -> None:
        """
        Args:
            model_id: Hugging Face model id or a local path.
            device: "cpu", "cuda", or None to auto-select (cuda if available).
            language: Optional language override (e.g. "hi", "en").
                Leave as None to use the model's own generation config,
                which is how the published benchmarks were produced.
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.language = language

        self.processor = WhisperProcessor.from_pretrained(model_id)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_id)
        self.model.to(self.device)
        self.model.eval()

        if language is not None:
            # Rebuild forced decoder ids for the requested language while
            # keeping the model's task (transcribe) and no-timestamps setup.
            tokenizer = self.processor.tokenizer
            forced = self.model.generation_config.forced_decoder_ids or []
            new_forced = []
            for pos, token_id in forced:
                if pos == 1:
                    lang_id = tokenizer.convert_tokens_to_ids(f"<|{language}|>")
                    new_forced.append([pos, lang_id])
                else:
                    new_forced.append([pos, token_id])
            self.model.generation_config.forced_decoder_ids = new_forced

    # ------------------------------------------------------------------
    def transcribe_array(
        self, audio: np.ndarray, sample_rate: int = TARGET_SAMPLE_RATE
    ) -> TranscriptionResult:
        """Transcribe a mono float waveform (values in [-1, 1]).

        Audio longer than 30 s is processed in 30-second windows and the
        transcripts are concatenated — the standard approach for Whisper
        models whose generation config does not return timestamps.
        """
        if sample_rate != TARGET_SAMPLE_RATE:
            audio = resample(audio, sample_rate, TARGET_SAMPLE_RATE)

        audio = np.asarray(audio, dtype=np.float32).flatten()
        total_seconds = len(audio) / TARGET_SAMPLE_RATE
        window = _CHUNK_SECONDS * TARGET_SAMPLE_RATE

        started = time.perf_counter()
        texts: list[str] = []
        for start in range(0, max(len(audio), 1), window):
            chunk = audio[start : start + window]
            if chunk.size == 0:
                continue
            inputs = self.processor(
                chunk, sampling_rate=TARGET_SAMPLE_RATE, return_tensors="pt"
            )
            input_features = inputs.input_features.to(self.device)
            with torch.no_grad():
                predicted_ids = self.model.generate(input_features)
            text = self.processor.batch_decode(
                predicted_ids, skip_special_tokens=True
            )[0].strip()
            if text:
                texts.append(text)
        elapsed = time.perf_counter() - started

        return TranscriptionResult(
            text=" ".join(texts),
            audio_seconds=total_seconds,
            processing_seconds=elapsed,
        )

    def transcribe_file(self, path: str) -> TranscriptionResult:
        """Load an audio file (wav/flac/mp3 via soundfile/ffmpeg fallbacks)
        and transcribe it."""
        audio, sample_rate = load_audio(path)
        return self.transcribe_array(audio, sample_rate)


# ----------------------------------------------------------------------
def resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample a waveform with a polyphase filter (scipy)."""
    from math import gcd

    from scipy.signal import resample_poly

    g = gcd(orig_sr, target_sr)
    return resample_poly(audio, target_sr // g, orig_sr // g).astype(np.float32)


def load_audio(path: str) -> tuple[np.ndarray, int]:
    """Read an audio file into a mono float32 waveform.

    Uses soundfile (wav/flac/ogg). For mp3/m4a, falls back to librosa
    if installed. Returns (waveform, sample_rate).
    """
    import soundfile as sf

    try:
        audio, sample_rate = sf.read(path, always_2d=False)
    except RuntimeError:
        try:
            import librosa
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                f"soundfile cannot read {path!r}. Install librosa "
                "(pip install librosa) for mp3/m4a support."
            ) from exc
        audio, sample_rate = librosa.load(path, sr=None, mono=True)
        return np.asarray(audio, dtype=np.float32), int(sample_rate)

    audio = np.asarray(audio)
    if audio.ndim > 1:  # stereo -> mono
        audio = audio.mean(axis=1)
    return audio.astype(np.float32), int(sample_rate)
