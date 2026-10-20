"""Hinglish ASR — fine-tuned Whisper for Hindi-English code-switched speech."""

from .model import MODEL_ID, HinglishASR, TranscriptionResult

__all__ = ["MODEL_ID", "HinglishASR", "TranscriptionResult"]
__version__ = "1.0.0"
