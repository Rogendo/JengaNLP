"""Download Google FLEURS for one language and write JengaNLP ASR manifests.

FLEURS (CC-BY-4.0, not gated) has read speech with transcripts in 102 languages.
Kenyan Swahili is ``sw_ke``: train 3,070 clips (3.1 GB), validation 211 (183 MB), test 487 (441 MB).

    pip install "jenga-nlp[asr]" pandas pyarrow
    python examples/prepare_fleurs_asr.py --lang sw_ke --out data/fleurs_sw_ke
    python examples/prepare_fleurs_asr.py --lang sw_ke --out data/fleurs_sw_ke --splits validation  # 183 MB only

Writes <out>/clips/<split>/*.wav and <out>/{train,validation,test}.jsonl with rows
``{"audio": path, "text": transcript, "duration_s": seconds}``, the format
``dataset.source: manifest`` reads. Audio paths are absolute, so configs work from any directory.
Sentences that occur in validation/test are removed from train (FLEURS repeats sentences across speakers).
"""

from __future__ import annotations

import argparse
import io
import json
import urllib.request
from pathlib import Path

PARQUET_API = "https://huggingface.co/api/datasets/google/fleurs/parquet/{lang}/{split}"


def download(lang: str, split: str, raw_dir: Path) -> list[Path]:
    with urllib.request.urlopen(PARQUET_API.format(lang=lang, split=split), timeout=60) as resp:
        urls = json.load(resp)
    paths = []
    for i, url in enumerate(urls):
        path = raw_dir / f"{lang}_{split}_{i}.parquet"
        if not path.exists():
            print(f"downloading {url}")
            tmp = path.with_suffix(".part")
            urllib.request.urlretrieve(url, tmp)
            tmp.rename(path)
        paths.append(path)
    return paths


def convert(parquets: list[Path], clip_dir: Path, max_s: float) -> list[dict]:
    import pandas as pd
    import soundfile as sf

    clip_dir.mkdir(parents=True, exist_ok=True)
    rows, seen = [], set()
    for parquet in parquets:
        for r in pd.read_parquet(parquet).itertuples(index=False):
            audio, rate = sf.read(io.BytesIO(r.audio["bytes"]))
            duration = len(audio) / rate
            text = str(r.transcription).strip()
            if not text or not 0.5 <= duration <= max_s or (r.id, text, duration) in seen:
                continue
            seen.add((r.id, text, duration))
            path = clip_dir / f"{r.id}_{len(rows):05d}.wav"
            if not path.exists():
                sf.write(path, audio, rate)
            rows.append({"audio": str(path.resolve()), "text": text, "duration_s": round(duration, 2),
                         "fleurs_id": int(r.id), "gender": int(r.gender)})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lang", default="sw_ke", help="FLEURS config, e.g. sw_ke, am_et, yo_ng, so_so")
    ap.add_argument("--out", type=Path, default=Path("data/fleurs_sw_ke"))
    ap.add_argument("--splits", nargs="+", default=["train", "validation", "test"])
    ap.add_argument("--max-seconds", type=float, default=30.0, help="Whisper's input window is 30 s")
    args = ap.parse_args()

    out = args.out
    (out / "raw").mkdir(parents=True, exist_ok=True)
    data = {s: convert(download(args.lang, s, out / "raw"), out / "clips" / s, args.max_seconds) for s in args.splits}
    if "train" in data:
        held_out = {r["text"] for s in ("validation", "test") for r in data.get(s, [])}
        before = len(data["train"])
        data["train"] = [r for r in data["train"] if r["text"] not in held_out]
        print(f"train: removed {before - len(data['train'])} clips whose sentence is in validation/test")
    for split, rows in data.items():
        (out / f"{split}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                                            encoding="utf-8")
        hours = sum(r["duration_s"] for r in rows) / 3600
        print(f"{split}: {len(rows)} clips, {hours:.1f} h -> {out / (split + '.jsonl')}")
    print(f"raw parquet files can be deleted to save space: {out / 'raw'}")


if __name__ == "__main__":
    main()
