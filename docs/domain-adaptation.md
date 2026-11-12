---
title: "Domain-Adaptive Pretraining for AfriBERTa and AfroXLMR (Swahili, Sheng)"
description: "Continue masked-language-model training of AfriBERTa or AfroXLMR on Kenyan text - Swahili, Sheng, code-switching - with NER-based PII redaction and a recall gate, using JengaNLP."
---

# Domain-adaptive pretraining (MLM)

Pretrained African-language encoders such as **AfriBERTa** and **AfroXLMR** learned Swahili mostly from news and Wikipedia. They have seen little Sheng, Swahili–English code-switching, M-PESA vocabulary or helpline language. **Domain-adaptive pretraining** fixes this: you continue the model's original *masked-language-model* (MLM) training on your own text before fine-tuning it on any task.

`jenga-nlp mlm` does this for any XLM-R-family encoder:

| Model | Hugging Face ID | Size |
|---|---|---|
| AfriBERTa large | `castorini/afriberta_large` | 126M |
| AfroXLMR mini / small | `Davlan/afro-xlmr-mini`, `Davlan/afro-xlmr-small` | 118M / 140M |
| AfroXLMR base | `Davlan/afro-xlmr-base` | 278M |
| AfroXLMR large | `Davlan/afro-xlmr-large` | 560M |

The result is a standard Hugging Face model. Use it as the encoder for any JengaNLP task:

```yaml
model:
  base_model: results/mlm/afriberta_large_kenya   # or your Hub ID
```

## Workflow

```bash
# 1. How badly do the tokenizers split your text? (downloads tokenizers only)
jenga-nlp mlm tokenizer-report --config examples/configs/mlm_afriberta_large_kenya.yaml \
  --models castorini/afriberta_large Davlan/afro-xlmr-base

# 2. Continue MLM training on your corpus (GPU recommended)
jenga-nlp mlm train --config examples/configs/mlm_afriberta_large_kenya.yaml

# 3. Compare base vs adapted on the held-out split and your probe sentences
jenga-nlp mlm evaluate --config examples/configs/mlm_afriberta_large_kenya.yaml \
  --model-dir results/mlm/afriberta_large_kenya
```

Try the whole thing on CPU in a few minutes with `examples/configs/mlm_cpu_smoke.yaml`. It uses `afro-xlmr-mini`, a tiny synthetic corpus and 20 steps. `examples/configs/mlm_cpu_sensitive_demo.yaml` adds a fictional sensitive source to show NER redaction and the recall gate.

## The corpus

List each source under `corpus:`. A source is a local file or a Hugging Face dataset:

```yaml
corpus:
  - name: swahili_wikipedia            # public text
    hf_dataset: wikimedia/wikipedia
    hf_config: 20231101.sw
    max_examples: 200000
  - name: kenyan_domain                # your text: .txt (one example per line), .jsonl or .csv
    path: data/domain_text.txt
    repeat: 5                          # upsample small, high-value sources
  - name: helpline_chats               # real conversations about people
    path: /secure/helpline_chats.txt
    sensitive: true
    repeat: 2
```

The pipeline, in order:
1. Clean the text and drop examples shorter than `min_chars`.
2. Redact PII.
3. Drop exact duplicates across all sources.
4. Split every source into training and held-out examples.
5. Repeat the training examples only.

Because the held-out split happens *before* repetition, held-out text can never appear in training, so held-out perplexity is honest.

## Protecting personal data

Masked language models memorise text they see often, and they can then reproduce it through fill-mask queries ("Mtoto `<mask>` alipelekwa makazi salama" → a real child's name). Personal data has to be removed from the corpus **before** training.

Redaction combines two detectors:

| Detector | Finds | Runs on |
|---|---|---|
| **Regex** (`pii.redact`) | Kenyan phone numbers (any spacing, `07…`, `01…`, `+254…`), national IDs, KRA PINs, M-PESA transaction IDs, emails, URLs, P.O. boxes, dates of birth | every source |
| **NER model** (`pii.ner_model`) | names, places, and any other entity labels you map | sources marked `sensitive: true` (all sources with `pii.ner_on_all_sources: true`) |

The NER model can be a **JengaNLP NER model folder** (trained with `jenga-nlp train`) or any **Hugging Face token-classification model**, e.g. `masakhane/afroxlmr-large-ner-masakhaner-1.0_2.0`: AfroXLMR-large fine-tuned on MasakhaNER, including Swahili (F1 93.4 on MasakhaNER 2.0 news text). Labels map to tags via `pii.ner_labels`. By default, `PER`/`NAME`/`VICTIM`/`PERPETRATOR` → `[NAME]`, `LOC`/`LOCATION`/`LANDMARK` → `[LOCATION]`, and `AGE` and `PHONE_NUMBER` are also redacted; `ORG` and `DATE` are kept. NER spans are widened to whole words, so a partly tagged name is still fully removed. `pii.ner_threshold` defaults to a low 0.3: for pretraining text, over-redacting is cheap and a missed name is not.

### The recall gate

An NER model protects data only as well as it finds it, so a source marked `sensitive: true` requires:

- `pii.ner_model` and `pii.eval_set`: a **labelled sample of your own sensitive data** in JengaNLP's NER format (`{"text": ..., "entities": [{"start", "end", "label"}]}`, see `examples/fixtures/pii_eval_example.jsonl`);
- **before training starts**, the whole redaction stack is scored on that sample. A PII entity counts as caught only if every character of it is redacted. If recall is below `pii.min_recall` (default 0.95), **training refuses to start** and reports the misses per tag;
- at most 2× repetition, and Hub uploads as private only.

Check recall on its own, without training, while you tune the model or threshold:

```bash
jenga-nlp mlm pii-check --config examples/configs/mlm_cpu_sensitive_demo.yaml
```

Each run writes `redaction_report.json` (counts per source and per type) and `pii_recall.json`. Neither ever contains the redacted values.

A recall check is only as good as its sample. Label real examples from the sources you'll train on, including the hard cases: nicknames, Sheng spellings, names that are also ordinary words (*Amani*, *Baraka*, *Zawadi*). Indirect identifiers (a school plus an age plus a village) can still identify a child, so review sensitive data yourself.

## Reading the comparison

`mlm evaluate` scores the base and adapted models on the **same sentences** with *pseudo-perplexity*: each token is masked in turn and the model's probability of the true token is recorded. Lower is better. It's deterministic, so the comparison is fair. Domains are:

- `heldout/<source>`: each source's held-out split, rebuilt from the config with the same seed, so it's exactly the data training never saw;
- your probe file (`probes_path`, JSONL `{"domain": "sheng", "text": "..."}`). Use dozens of sentences per domain, not a handful. Domains with few sentences are noisy, and the report shows the sentence count next to each score.

Perplexity measures how well the model fits the language, not task quality. The real test is downstream: train the same JengaNLP task on the base model and on the adapted one, with the same data and seeds, and compare.

## GPU guidance (A40, 48 GB, bf16)

| Config | Batch × accumulation | Learning rate | Notes |
|---|---|---|---|
| `mlm_afriberta_large_kenya.yaml` | 64 × 1 | 1e-4 | |
| `mlm_afroxlmr_base_kenya.yaml` | 32 × 2 | 5e-5 | |
| `mlm_afroxlmr_large_kenya.yaml` | 16 × 4 | 3e-5 | gradient checkpointing on |

On 16–24 GB GPUs (T4/L4), use `fp16: true` instead of `bf16`, halve the batch size and double `gradient_accumulation_steps`.

Intermediate checkpoints (weights plus optimizer state, roughly 3× the model size) are written every `save_steps`, so a crashed run can be resumed. After a successful run they're deleted, because the final model is already saved. Set `keep_checkpoints: true` to keep them.

## Tokenizer fragmentation

`mlm tokenizer-report` shows, per model and per source, the average number of sub-word pieces per word and the share of words split into 3+ pieces. For non-sensitive sources it also lists the worst-fragmented frequent words. If Sheng or domain words fragment badly in every model, extending the vocabulary is the next step (not in this version).
