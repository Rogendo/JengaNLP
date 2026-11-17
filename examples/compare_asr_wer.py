"""Word/character error rate of several Whisper models on one ASR manifest (base vs fine-tuned).

    python examples/compare_asr_wer.py --manifest data/fleurs_sw_ke/test.jsonl --language sw \
        --models openai/whisper-small results/whisper-small-sw

Decoding is forced to ``--language``; text is lower-cased and punctuation is removed on both sides
before scoring, so models are compared on words, not formatting. Uses JengaNLP's audio loader and
WER/CER code. Writes a JSON report with a few example transcripts per model.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from jenga_nlp.asr.config import WhisperPreprocessingConfig
from jenga_nlp.asr.loaders import load_audio_file
from jenga_nlp.asr.metrics import compute_wer_cer


def normalise(text: str) -> str:
    return " ".join(re.sub(r"[^\w\s']", " ", text.lower()).split())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--language", default="sw")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--out", type=Path, default=Path("asr_comparison.json"))
    args = ap.parse_args()

    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    rows = [json.loads(line) for line in args.manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = rows[: args.limit] if args.limit else rows
    pre = WhisperPreprocessingConfig(min_audio_duration_s=0.0, max_audio_duration_s=30.0)
    audio = [load_audio_file(r["audio"], pre)["array"] for r in rows]
    refs = [normalise(r["text"]) for r in rows]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    report = {"manifest": str(args.manifest), "clips": len(rows), "models": {}}
    for name in args.models:
        processor = WhisperProcessor.from_pretrained(name)
        dtype = torch.float16 if device == "cuda" else torch.float32
        model = WhisperForConditionalGeneration.from_pretrained(name, torch_dtype=dtype).to(device).eval()
        preds = []
        for i in range(0, len(audio), args.batch_size):
            feats = processor.feature_extractor(audio[i:i + args.batch_size], sampling_rate=16000,
                                                return_tensors="pt").input_features.to(device, dtype)
            with torch.no_grad():
                ids = model.generate(feats, language=args.language, task="transcribe", max_new_tokens=200)
            preds += [normalise(p) for p in processor.batch_decode(ids, skip_special_tokens=True)]
        scores = {k: round(v, 2) for k, v in compute_wer_cer(preds, refs).items()}
        report["models"][name] = {**scores, "examples": [{"ref": r, "pred": p} for r, p in zip(refs[:5], preds[:5])]}
        print(f"{name:45} WER {scores['wer']:6.2f}  CER {scores['cer']:6.2f}")
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
    args.out.write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"report -> {args.out}")


if __name__ == "__main__":
    main()
