#!/usr/bin/env python3
"""Transcribe audio files with the Hinglish ASR model — runs fully on-device.

Usage:
    python transcribe.py audio.wav
    python transcribe.py audio1.wav audio2.mp3 --output results.txt
    python transcribe.py --input-dir ./clips --output-dir ./transcripts
    python transcribe.py audio.wav --device cuda        # force GPU
    python transcribe.py audio.wav --language en        # override language

The model is downloaded once from Hugging Face and cached locally;
after that, transcription works completely offline.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from hinglish_asr import MODEL_ID, HinglishASR  # noqa: E402

AUDIO_SUFFIXES = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".webm"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Transcribe Hinglish (Hindi-English code-switched) audio."
    )
    parser.add_argument("files", nargs="*", help="Audio file(s) to transcribe")
    parser.add_argument("--input-dir", help="Transcribe every audio file in a folder")
    parser.add_argument(
        "--output", help="Write all transcripts to this text file (file<TAB>text)"
    )
    parser.add_argument(
        "--output-dir", help="Write one .txt transcript per input file into this folder"
    )
    parser.add_argument("--model", default=MODEL_ID, help="HF model id or local path")
    parser.add_argument("--device", choices=["cpu", "cuda"], help="Force device")
    parser.add_argument(
        "--language",
        help="Language override (e.g. hi, en). Default: the model's own config (hi)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    files: list[Path] = [Path(f) for f in args.files]
    if args.input_dir:
        folder = Path(args.input_dir)
        files.extend(
            sorted(p for p in folder.iterdir() if p.suffix.lower() in AUDIO_SUFFIXES)
        )
    if not files:
        print("No audio files given. See --help.", file=sys.stderr)
        return 2

    print(f"Loading model {args.model} ...", file=sys.stderr)
    asr = HinglishASR(model_id=args.model, device=args.device, language=args.language)
    print(f"Model ready on {asr.device}.", file=sys.stderr)

    out_lines: list[str] = []
    for path in files:
        result = asr.transcribe_file(str(path))
        line = f"{path.name}\t{result.text}"
        print(f"\n=== {path.name} ===")
        print(result.text)
        print(
            f"[{result.audio_seconds:.1f}s audio in {result.processing_seconds:.1f}s "
            f"-> RTF {result.real_time_factor:.3f}]",
            file=sys.stderr,
        )
        out_lines.append(line)
        if args.output_dir:
            out_dir = Path(args.output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / f"{path.stem}.txt").write_text(result.text, encoding="utf-8")

    if args.output:
        Path(args.output).write_text("\n".join(out_lines) + "\n", encoding="utf-8")
        print(f"\nTranscripts written to {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
