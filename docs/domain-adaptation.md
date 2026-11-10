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

Try the whole thing on CPU in a few minutes with `examples/configs/mlm_cpu_smoke.yaml`. It uses `afro-xlmr-mini`, a tiny synthetic corpus and 20 steps.

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

## Sensitive data guard rails

Masked language models memorise text they see often, and they can then reproduce it through fill-mask queries. Real chats or case notes therefore have to be treated with care. A source marked `sensitive: true`:

- **requires PII redaction.** Kenyan phone numbers (any spacing, `07…`, `01…`, `+254…`), national IDs, KRA PINs, M-PESA transaction IDs, emails, URLs, P.O. boxes and dates are replaced with tags such as `[PHONE]`.
- **requires a names list** (`pii.names_file`, one name per line). The built-in redaction **cannot detect person names**, so you provide them: common Kenyan first names and surnames, plus the names of children, carers and staff in your data. They're replaced with `[NAME]` as whole words, case-insensitive, with the longest match winning. See `examples/fixtures/names_example.txt`.
- **can be repeated at most 2×.**
- **forces private Hub uploads.** `hub_private: false` is rejected.

Each run writes `redaction_report.json`: example counts per source and redaction counts per type. It never contains the redacted values themselves. Read it before you publish anything.

Redaction lowers the risk but doesn't remove it. Names not on your list, nicknames, and indirect identifiers such as locations can still remain. Review sensitive sources yourself.

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
