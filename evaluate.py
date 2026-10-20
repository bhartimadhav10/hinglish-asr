#!/usr/bin/env python3
"""Evaluate the Hinglish ASR model with WER / CER.

Two modes:

1. Local manifest — a JSONL file with one object per row:
       {"audio": "path/to/clip.wav", "text": "reference transcript"}
       {"audio_filepath": "...", "transcription": "..."}   (also accepted)

       python evaluate.py --manifest data/test.jsonl --limit 100

2. Hugging Face dataset — streams rows from a dataset that has both audio
   and transcript columns (no full download):

       python evaluate.py --dataset <org>/<dataset> \
           --audio-column audio --text-column text --limit 50

   (The public hinglish-asr dataset release is audio-only; the tool detects
   a missing transcript column and explains — use --manifest there.)

Results are printed and saved as JSON alongside per-utterance predictions
so error analysis is reproducible.

Normalisation before scoring: lowercase, strip punctuation. This matches
the normalisation used for the published benchmark numbers in
results/benchmark_results.json.
"""

from __future__ import annotations

import argparse
import json
import re
import string
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import numpy as np  # noqa: E402

from hinglish_asr import MODEL_ID, HinglishASR  # noqa: E402

_PUNCT = str.maketrans("", "", string.punctuation)


def normalise(text: str) -> str:
    return " ".join(text.lower().translate(_PUNCT).split())


def score(references: list[str], hypotheses: list[str]) -> dict:
    import jiwer

    refs = [normalise(r) or " " for r in references]
    hyps = [normalise(h) or " " for h in hypotheses]
    return {
        "wer": round(jiwer.wer(refs, hyps) * 100, 2),
        "cer": round(jiwer.cer(refs, hyps) * 100, 2),
        "n_utterances": len(refs),
    }


def load_manifest(path: str, limit: int | None) -> list[tuple[str, str]]:
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        audio = obj.get("audio") or obj.get("audio_filepath") or obj.get("path")
        text = obj.get("text") or obj.get("transcription") or obj.get("sentence")
        if audio and text:
            rows.append((audio, text))
        if limit and len(rows) >= limit:
            break
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="WER/CER evaluation for Hinglish ASR")
    parser.add_argument("--manifest", help="Local JSONL manifest (audio + reference)")
    parser.add_argument("--dataset", help="HF dataset id to stream")
    parser.add_argument("--split", default="train")
    parser.add_argument("--audio-column", default="audio")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--device", choices=["cpu", "cuda"])
    parser.add_argument("--output", default="results/evaluation_run.json")
    args = parser.parse_args()

    if not args.manifest and not args.dataset:
        parser.error("Pass either --manifest or --dataset")

    asr = HinglishASR(model_id=args.model, device=args.device)
    references: list[str] = []
    hypotheses: list[str] = []
    per_item: list[dict] = []
    started = time.perf_counter()

    if args.manifest:
        items = load_manifest(args.manifest, args.limit)
        for i, (audio_path, reference) in enumerate(items, 1):
            result = asr.transcribe_file(audio_path)
            references.append(reference)
            hypotheses.append(result.text)
            per_item.append(
                {"audio": audio_path, "reference": reference, "hypothesis": result.text}
            )
            print(f"[{i}/{len(items)}] {Path(audio_path).name} -> {result.text[:80]}")
    else:
        from datasets import load_dataset

        ds = load_dataset(args.dataset, split=args.split, streaming=True)
        iterator = iter(ds)
        first = next(iterator, None)
        if first is None:
            print("Dataset split is empty.", file=sys.stderr)
            return 1
        missing = [c for c in (args.audio_column, args.text_column) if c not in first]
        if missing:
            print(
                f"Dataset rows do not contain column(s): {missing}. "
                f"Available columns: {list(first.keys())}.\n"
                "Note: the public hinglish-asr-dataset/hinglish-asr release is "
                "audio-only (no reference transcripts), so it cannot be scored "
                "directly — evaluate on labelled data with --manifest instead, "
                "or point --text-column at a transcript column that exists.",
                file=sys.stderr,
            )
            return 2
        import itertools

        count = 0
        for row in itertools.chain([first], iterator):
            if count >= args.limit:
                break
            sample = row[args.audio_column]
            reference = row[args.text_column]
            audio = np.asarray(sample["array"], dtype=np.float32)
            result = asr.transcribe_array(audio, sample["sampling_rate"])
            references.append(reference)
            hypotheses.append(result.text)
            per_item.append(
                {"reference": reference, "hypothesis": result.text}
            )
            count += 1
            print(f"[{count}/{args.limit}] ref: {reference[:60]} | hyp: {result.text[:60]}")

    metrics = score(references, hypotheses)
    metrics["elapsed_seconds"] = round(time.perf_counter() - started, 1)
    metrics["model"] = args.model
    metrics["device"] = asr.device
    metrics["normalisation"] = "lowercase, punctuation removed, whitespace collapsed"

    print("\n=== Results ===")
    print(json.dumps(metrics, indent=2))

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps({"metrics": metrics, "predictions": per_item}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Saved to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
