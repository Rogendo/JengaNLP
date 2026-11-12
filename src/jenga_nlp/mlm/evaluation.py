"""Compare a base encoder and its domain-adapted version fairly.

Both models are scored on the *same* sentences with **pseudo-perplexity**
(PPPL): every token is masked once and the model's probability of the true
token is recorded. It is deterministic (no random masking), so the two models
see identical inputs. Lower is better.

Domains come from:
- ``heldout/<source>``: each corpus source's held-out split (rebuilt from the
  config with the same seed, so it is exactly the split training never saw);
- probe sentences: a JSONL file of ``{"domain": ..., "text": ...}`` lines.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from .config import MLMConfig
from .data import clean, prepare_corpus


def load_probes(path: str | Path) -> dict[str, list[str]]:
    domains: dict[str, list[str]] = defaultdict(list)
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            if isinstance(row.get("text"), str) and row["text"].strip():
                domains[str(row.get("domain", "probe"))].append(row["text"].strip())
    return dict(domains)


def pseudo_log_likelihood(
    model: Any, tokenizer: Any, text: str, device: Any, max_length: int = 256, batch_size: int = 64
) -> tuple[float, int]:
    """Sum of log-probabilities of each token when masked alone, and the number of tokens scored."""
    import torch

    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length,
                    return_special_tokens_mask=True)
    input_ids = enc["input_ids"][0]
    positions = [i for i, special in enumerate(enc["special_tokens_mask"][0].tolist()) if not special]
    if not positions:
        return 0.0, 0
    total = 0.0
    for start in range(0, len(positions), batch_size):
        chunk = positions[start : start + batch_size]
        batch = input_ids.repeat(len(chunk), 1)
        for row, pos in enumerate(chunk):
            batch[row, pos] = tokenizer.mask_token_id
        with torch.no_grad():
            logits = model(input_ids=batch.to(device), attention_mask=torch.ones_like(batch).to(device)).logits
        log_probs = torch.log_softmax(logits.float(), dim=-1)
        for row, pos in enumerate(chunk):
            total += log_probs[row, pos, input_ids[pos]].item()
    return total, len(positions)


def score_domains(model_name: str | Path, domains: dict[str, list[str]], device: str = "cpu") -> dict[str, float]:
    """Pseudo-perplexity per domain (token-weighted)."""
    import torch
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(model_name))
    model = AutoModelForMaskedLM.from_pretrained(str(model_name)).to(device).eval()
    scores = {}
    for domain, sentences in domains.items():
        log_likelihood, tokens = 0.0, 0
        for sentence in sentences:
            ll, n = pseudo_log_likelihood(model, tokenizer, sentence, device)
            log_likelihood += ll
            tokens += n
        scores[domain] = math.exp(-log_likelihood / tokens) if tokens else float("nan")
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return scores


def compare_models(
    config: MLMConfig,
    adapted_model: str | Path,
    base_model: str | None = None,
    max_sentences_per_domain: int = 200,
    device: str = "auto",
) -> dict[str, Any]:
    """Score base and adapted models on held-out splits and probes; return a comparison."""
    from ..utils.device import get_device

    resolved = str(get_device(device))
    domains: dict[str, list[str]] = {}
    corpus = prepare_corpus(config)
    for source, texts in corpus.eval_by_source.items():
        if texts:
            domains[f"heldout/{source}"] = texts[:max_sentences_per_domain]
    # Probe sentences that also occur in the training data measure memorisation, not adaptation.
    trained_on = {clean(t).casefold() for t in corpus.train_texts}
    in_training: dict[str, int] = {}
    if config.probes_path:
        for domain, texts in load_probes(config.probes_path).items():
            domains[domain] = texts[:max_sentences_per_domain]
            in_training[domain] = sum(clean(t).casefold() in trained_on for t in domains[domain])
    if not domains:
        raise ValueError("nothing to evaluate: no held-out examples and no probes_path")

    base = base_model or config.base_model
    base_scores = score_domains(base, domains, resolved)
    adapted_scores = score_domains(adapted_model, domains, resolved)
    return {
        "metric": "pseudo_perplexity",
        "lower_is_better": True,
        "base_model": base,
        "adapted_model": str(adapted_model),
        "domains": {
            name: {
                "sentences": len(texts),
                "in_training": in_training.get(name, 0),
                "base": base_scores[name],
                "adapted": adapted_scores[name],
            }
            for name, texts in domains.items()
        },
    }


def write_comparison(comparison: dict[str, Any], output_dir: str | Path) -> tuple[Path, Path]:
    from .hub import comparison_table

    output_dir = Path(output_dir)
    json_path = output_dir / "comparison.json"
    json_path.write_text(json.dumps(comparison, indent=2))
    md_path = output_dir / "comparison.md"
    md_path.write_text(
        f"# Base vs adapted: pseudo-perplexity (lower is better)\n\n"
        f"- Base: `{comparison['base_model']}`\n- Adapted: `{comparison['adapted_model']}`\n\n"
        + comparison_table(comparison)
        + "\n\nDomains with few sentences are noisy; treat small differences with caution.\n"
        + _overlap_warning(comparison)
    )
    return json_path, md_path


def _overlap_warning(comparison: dict[str, Any]) -> str:
    flagged = {d: v["in_training"] for d, v in comparison["domains"].items() if v.get("in_training")}
    if not flagged:
        return ""
    detail = ", ".join(f"{d}: {n}/{comparison['domains'][d]['sentences']}" for d, n in flagged.items())
    return (
        f"\n**Warning:** some probe sentences also appear in the training data ({detail}). "
        "Their scores reflect memorisation; use probe sentences the model has not been trained on.\n"
    )
