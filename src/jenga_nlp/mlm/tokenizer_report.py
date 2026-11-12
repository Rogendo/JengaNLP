"""How badly does each model's tokenizer fragment your corpus?

Words that are split into many sub-word pieces (common for Sheng and
code-switched text) are harder for an encoder to learn. This report helps decide
whether vocabulary extension is worth it. Only tokenizers are downloaded.

Per model and source: average pieces per word (frequency-weighted) and the share
of word occurrences split into 3+ pieces. The most-fragmented frequent words are
listed only for non-sensitive sources, so no words from sensitive data appear.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from .config import MLMConfig
from .data import prepare_corpus

_WORD = re.compile(r"[^\W\d_](?:[\w'-]*\w)?", re.UNICODE)
_PLACEHOLDER = re.compile(r"\[[A-Z_]+\]")  # redaction tags such as [PHONE], [NAME]


def count_pieces(tokenizer: Any, word: str) -> int:
    pieces = [p for p in tokenizer.tokenize(word) if p not in ("▁", "Ġ", "")]
    return max(1, len(pieces))


def word_counts(texts: list[str]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for text in texts:
        counts.update(_WORD.findall(_PLACEHOLDER.sub(" ", text)))
    return counts


def fragmentation(tokenizer: Any, counts: Counter[str], top_n: int = 15) -> dict[str, Any]:
    total = sum(counts.values())
    if not total:
        return {"word_occurrences": 0}
    pieces = {word: count_pieces(tokenizer, word) for word in counts}
    weighted = sum(pieces[w] * c for w, c in counts.items())
    heavy = sum(c for w, c in counts.items() if pieces[w] >= 3)
    frequent = [w for w, c in counts.most_common() if c >= 2] or list(counts)
    worst = sorted(frequent, key=lambda w: (-pieces[w], -counts[w]))[:top_n]
    return {
        "word_occurrences": total,
        "unique_words": len(counts),
        "avg_pieces_per_word": round(weighted / total, 3),
        "pct_words_3plus_pieces": round(100 * heavy / total, 1),
        "most_fragmented": [{"word": w, "pieces": pieces[w], "count": counts[w]} for w in worst],
    }


def tokenizer_report(config: MLMConfig, models: list[str], max_examples_per_source: int = 5000) -> dict[str, Any]:
    from transformers import AutoTokenizer

    corpus = prepare_corpus(config)  # cleaned + redacted, each example once
    sensitive = {s.name for s in config.corpus if s.sensitive}
    per_source = {
        name: word_counts(texts[:max_examples_per_source]) for name, texts in corpus.texts_by_source().items()
    }
    all_counts = sum(per_source.values(), Counter())

    report: dict[str, Any] = {"models": {}}
    for model in models:
        tokenizer = AutoTokenizer.from_pretrained(model)
        overall = fragmentation(tokenizer, all_counts)
        if sensitive:
            overall.pop("most_fragmented", None)  # all-corpus words may include sensitive text
        model_report: dict[str, Any] = {"vocab_size": tokenizer.vocab_size, "overall": overall, "sources": {}}
        for name, counts in per_source.items():
            stats = fragmentation(tokenizer, counts)
            if name in sensitive:
                stats.pop("most_fragmented", None)
            model_report["sources"][name] = stats
        report["models"][model] = model_report
    return report


def write_tokenizer_report(report: dict[str, Any], output_dir: str | Path) -> tuple[Path, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "tokenizer_report.json"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    lines = [
        "# Tokenizer fragmentation (lower is better)",
        "",
        "| Model | Vocab | Avg pieces / word | Words in 3+ pieces |",
        "|---|---|---|---|",
    ]
    for model, rep in report["models"].items():
        o = rep["overall"]
        lines.append(f"| `{model}` | {rep['vocab_size']} | {o.get('avg_pieces_per_word')} | "
                     f"{o.get('pct_words_3plus_pieces')}% |")
    for model, rep in report["models"].items():
        worst = rep["overall"].get("most_fragmented")
        if worst:
            lines += ["", f"Most fragmented frequent words for `{model}`: "
                      + ", ".join(f"{w['word']} ({w['pieces']})" for w in worst[:10])]
    md_path = output_dir / "tokenizer_report.md"
    md_path.write_text("\n".join(lines) + "\n")
    return json_path, md_path
