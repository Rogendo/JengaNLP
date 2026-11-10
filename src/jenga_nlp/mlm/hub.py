"""Model card and Hub publishing for domain-adapted MLM models."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import MLMConfig


def write_model_card(
    output_dir: str | Path,
    config: MLMConfig,
    results: dict[str, Any],
    corpus_report: dict[str, Any],
    comparison: dict[str, Any] | None = None,
) -> Path:
    """Write README.md describing exactly how the model was trained and evaluated."""
    lines = [
        "---",
        "library_name: transformers",
        "pipeline_tag: fill-mask",
        f"base_model: {config.base_model}",
        "tags: [jenga-nlp, domain-adaptation, masked-language-modeling]",
        "---",
        "",
        f"# {config.project_name}",
        "",
        f"Continued masked-language-model pretraining of `{config.base_model}` with "
        "[JengaNLP](https://github.com/Rogendo/JengaNLP) (`jenga-nlp mlm train`).",
        "",
        "## Training data",
        "",
        "| Source | Origin | Sensitive | Train | Held-out | Repeat |",
        "|---|---|---|---|---|---|",
    ]
    for src in corpus_report.get("sources", []):
        lines.append(
            f"| {src['name']} | {src.get('origin') or '-'} | {'yes' if src['sensitive'] else 'no'} | "
            f"{src['train_examples']} | {src['eval_examples']} | x{src['repeat']} |"
        )
    redactions = corpus_report.get("totals", {}).get("redactions", {})
    lines += [
        "",
        f"- PII redaction: {'on' if corpus_report.get('pii_redaction') else 'off'}"
        f"{' (with a names list)' if corpus_report.get('names_list_used') else ''}; "
        f"redacted items by type: {redactions or 'none'}",
        "- Held-out examples were split from each source **before** repetition, so they never appear in training.",
        "",
        "## Results",
        "",
        f"- Held-out MLM loss: {results.get('eval_loss', 'n/a')}",
        f"- Held-out perplexity: {results.get('eval_perplexity', 'n/a')}",
        f"- Training steps: {results.get('global_step')}",
    ]
    if comparison:
        lines += ["", "### Base vs adapted (pseudo-perplexity, lower is better)", "", comparison_table(comparison)]
    lines += [
        "",
        "## Use as a JengaNLP encoder",
        "",
        "```yaml",
        "model:",
        f"  base_model: {config.hub_model_id or config.output_dir}",
        "```",
        "",
        "## Limitations",
        "",
        "- Regex PII redaction and a names list reduce, but do not eliminate, personal data in the corpus.",
        "- Perplexity measures language-model fit, not downstream task quality; compare on your tasks.",
    ]
    path = Path(output_dir) / "README.md"
    path.write_text("\n".join(lines) + "\n")
    return path


def comparison_table(comparison: dict[str, Any]) -> str:
    rows = [
        "| Domain | Sentences | Base | Adapted | Change |",
        "|---|---|---|---|---|",
    ]
    for domain, values in comparison.get("domains", {}).items():
        base, adapted = values["base"], values["adapted"]
        change = f"{(adapted - base) / base * 100:+.1f}%" if base else "-"
        seen = f" ({values['in_training']} in training)" if values.get("in_training") else ""
        rows.append(f"| {domain} | {values['sentences']}{seen} | {base:.2f} | {adapted:.2f} | {change} |")
    return "\n".join(rows)


def push_mlm_model(output_dir: str | Path, config: MLMConfig, token: str | None = None) -> str:
    """Push the model folder (without training checkpoints) to the Hub."""
    from huggingface_hub import HfApi

    private = True if config.has_sensitive_sources else config.hub_private
    api = HfApi(token=token)
    api.create_repo(repo_id=config.hub_model_id, private=private, exist_ok=True)
    api.upload_folder(
        folder_path=str(output_dir),
        repo_id=config.hub_model_id,
        ignore_patterns=["checkpoints/*", "logs/*", "activity.jsonl"],
        commit_message="Upload JengaNLP domain-adapted MLM model",
    )
    return f"https://huggingface.co/{config.hub_model_id}"
