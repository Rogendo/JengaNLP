"""Domain-adaptive MLM pretraining: config guard rails, corpus pipeline, training, evaluation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from jenga_nlp.mlm.config import MLMConfig
from jenga_nlp.mlm.data import prepare_corpus
from jenga_nlp.mlm.pii_names import CorpusRedactor

TINY_MLM = "hf-internal-testing/tiny-xlm-roberta"


def _config(tmp_path: Path, **overrides) -> MLMConfig:
    names = tmp_path / "names.txt"
    names.write_text("# fictional\nZawadi Achieng\nZawadi\nBaraka\n")
    data = {
        "base_model": TINY_MLM,
        "output_dir": str(tmp_path / "mlm_out"),
        "corpus": [{"name": "domain", "path": str(tmp_path / "domain.txt")}],
        "pii": {"redact": True, "names_file": str(names)},
        "min_chars": 5,
        "eval_fraction": 0.2,
    }
    data.update(overrides)
    return MLMConfig(**data)


# ---------------------------------------------------------------------------
# Config guard rails
# ---------------------------------------------------------------------------

def test_sensitive_source_requires_names_list(tmp_path):
    with pytest.raises(ValidationError, match="names_file"):
        MLMConfig(base_model=TINY_MLM, corpus=[{"name": "chats", "path": "c.txt", "sensitive": True}])


def test_sensitive_source_repeat_is_capped(tmp_path):
    with pytest.raises(ValidationError, match="at most 2"):
        _config(tmp_path, corpus=[{"name": "chats", "path": "c.txt", "sensitive": True, "repeat": 3}])


def test_sensitive_source_requires_redaction_and_private_hub(tmp_path):
    with pytest.raises(ValidationError, match="pii.redact"):
        _config(tmp_path, corpus=[{"name": "c", "path": "c.txt", "sensitive": True}],
                pii={"redact": False, "names_file": "n.txt"})
    with pytest.raises(ValidationError, match="hub_private"):
        _config(tmp_path, corpus=[{"name": "c", "path": "c.txt", "sensitive": True}], hub_private=False)


def test_source_needs_exactly_one_location():
    with pytest.raises(ValidationError, match="exactly one"):
        MLMConfig(base_model=TINY_MLM, corpus=[{"name": "x"}])
    with pytest.raises(ValidationError, match="exactly one"):
        MLMConfig(base_model=TINY_MLM, corpus=[{"name": "x", "path": "a.txt", "hf_dataset": "a/b"}])


def test_example_mlm_configs_load():
    configs = sorted(Path(__file__).resolve().parent.parent.glob("examples/configs/mlm_*.yaml"))
    assert len(configs) >= 4
    for path in configs:
        cfg = MLMConfig.from_yaml(path)
        assert cfg.corpus and cfg.base_model


# ---------------------------------------------------------------------------
# Corpus pipeline
# ---------------------------------------------------------------------------

def test_heldout_never_appears_in_training_even_with_repetition(tmp_path):
    texts = [f"mfano wa sentensi namba {i} kuhusu M-PESA" for i in range(50)]
    cfg = _config(tmp_path, corpus=[{"name": "domain", "path": "unused.txt", "repeat": 5}])
    corpus = prepare_corpus(cfg, sources_text={"domain": texts})
    assert len(corpus.eval_texts) == 10
    assert len(corpus.train_texts) == 40 * 5  # repeat multiplies train only
    assert not set(corpus.eval_texts) & set(corpus.train_texts)


def test_dedup_across_sources_and_min_chars(tmp_path):
    cfg = _config(tmp_path, corpus=[{"name": "a", "path": "a.txt"}, {"name": "b", "path": "b.txt"}])
    corpus = prepare_corpus(cfg, sources_text={
        "a": ["Habari ya asubuhi rafiki", "fupi", "Mvua imenyesha sana leo"],
        "b": ["habari ya   asubuhi rafiki", "Bei ya mafuta imepanda"],  # duplicate of a[0] after cleaning
    })
    sources = {s["name"]: s for s in corpus.report["sources"]}
    assert sources["b"]["duplicates_removed"] == 1
    everything = corpus.train_texts + corpus.eval_texts
    assert "fupi" not in everything
    assert len(everything) == 3


def test_names_phones_and_emails_are_redacted_and_report_has_no_raw_values(tmp_path):
    cfg = _config(tmp_path, corpus=[{"name": "chats", "path": "c.txt", "sensitive": True}])
    corpus = prepare_corpus(cfg, sources_text={"chats": [
        "Mimi ni Zawadi Achieng, nipigie kwa +254 722 000 111 tafadhali",
        "Mama Baraka ametuma email kwa afisa@example.org leo",
        "Amanisha si jina kwenye orodha, Baraka ni jina",
        "Namba yake ni 0110 123 456 kwa sasa",
    ]})
    text = " ".join(corpus.train_texts + corpus.eval_texts)
    for raw in ["Zawadi", "Achieng", "Baraka", "722 000 111", "afisa@example.org", "0110 123 456"]:
        assert raw not in text
    assert "Amanisha" in text  # whole-word matching only
    report = json.dumps(corpus.report)
    assert "Zawadi" not in report and "722" not in report
    totals = corpus.report["totals"]["redactions"]
    assert totals["name"] == 3 and totals["phone_ke"] == 2 and totals["email"] == 1


def test_overlapping_matches_keep_the_longest_name():
    redactor = CorpusRedactor(names=["Zawadi", "Zawadi Achieng"])
    assert redactor.redact("Habari Zawadi Achieng!") == "Habari [NAME]!"
    assert redactor.counts["name"] == 1


# ---------------------------------------------------------------------------
# Tokenizer report
# ---------------------------------------------------------------------------

def test_tokenizer_report_hides_words_from_sensitive_sources(tmp_path):
    from jenga_nlp.mlm.tokenizer_report import fragmentation, tokenizer_report, word_counts

    class SplitInTwo:
        vocab_size = 10

        def tokenize(self, word):
            return [word] if len(word) <= 3 else [word[:2], word[2:]]

    stats = fragmentation(SplitInTwo(), word_counts(["paka paka nzi"]))  # 2 + 2 + 1 pieces
    assert stats["avg_pieces_per_word"] == pytest.approx((2 + 2 + 1) / 3, abs=1e-3)  # rounded to 3 dp

    (tmp_path / "pub.txt").write_text("Msee alituma doh kwa M-PESA jana jioni\nWasee wa mtaa wamepanga harambee\n")
    (tmp_path / "chats.txt").write_text("Siri ya kesi hii ni nyeti sana kwa familia\nKesi nyingine nyeti imefika leo\n")
    cfg = _config(tmp_path, corpus=[
        {"name": "public", "path": str(tmp_path / "pub.txt")},
        {"name": "chats", "path": str(tmp_path / "chats.txt"), "sensitive": True},
    ])
    report = tokenizer_report(cfg, [TINY_MLM])["models"][TINY_MLM]
    assert "most_fragmented" in report["sources"]["public"]
    assert "most_fragmented" not in report["sources"]["chats"]
    assert "most_fragmented" not in report["overall"]
    assert report["sources"]["chats"]["word_occurrences"] > 0


# ---------------------------------------------------------------------------
# Training, evaluation, downstream use (tiny model, CPU)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def tiny_mlm_available():
    try:
        from transformers import AutoTokenizer

        AutoTokenizer.from_pretrained(TINY_MLM)
    except Exception as exc:  # pragma: no cover - network dependent
        pytest.skip(f"cannot load {TINY_MLM}: {exc}")


def _write_corpus(tmp_path: Path) -> None:
    lines = [f"Sentensi ya mfano namba {i} kuhusu malipo ya M-PESA na huduma kwa wateja" for i in range(40)]
    (tmp_path / "domain.txt").write_text("\n".join(lines) + "\n")
    (tmp_path / "probes.jsonl").write_text(
        '{"domain": "mpesa", "text": "Tuma pesa kwa M-PESA leo"}\n'
        '{"domain": "sheng", "text": "Msee alikuwa poa sana"}\n'
    )


def test_mlm_train_end_to_end_with_mlflow(tmp_path, tiny_mlm_available, monkeypatch):
    mlflow = pytest.importorskip("mlflow")
    from transformers import AutoModelForMaskedLM

    from jenga_nlp.mlm.training import MLMTrainer

    monkeypatch.chdir(tmp_path)
    _write_corpus(tmp_path)
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    cfg = _config(tmp_path, max_steps=4, batch_size=4, block_size=32, eval_steps=2, logging_steps=1,
                  save_steps=1000, device="cpu", project_name="mlm_test",
                  logging={"service": "mlflow", "experiment_name": "mlm_test", "tracking_uri": uri})
    results = MLMTrainer(cfg).train()

    out = Path(cfg.output_dir)
    for name in ("config.json", "mlm_results.json", "redaction_report.json", "experiment_config.yaml", "README.md"):
        assert (out / name).exists(), name
    assert AutoModelForMaskedLM.from_pretrained(out) is not None
    assert not (out / "checkpoints").exists()  # removed after a successful run by default
    assert results["global_step"] == 4 and "eval_perplexity" in results

    client = mlflow.MlflowClient(tracking_uri=uri)
    run = client.search_runs([client.get_experiment_by_name("mlm_test").experiment_id])[0]
    assert run.info.status == "FINISHED"
    assert run.data.params["base_model"] == TINY_MLM
    assert "mlm/final_eval_loss" in run.data.metrics


def test_mlm_train_failure_marks_run_failed(tmp_path, tiny_mlm_available, monkeypatch):
    mlflow = pytest.importorskip("mlflow")
    import transformers

    from jenga_nlp.mlm.training import MLMTrainer

    monkeypatch.chdir(tmp_path)
    _write_corpus(tmp_path)
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    cfg = _config(tmp_path, max_steps=2, batch_size=4, block_size=32, device="cpu",
                  logging={"service": "mlflow", "experiment_name": "mlm_fail", "tracking_uri": uri})

    def boom(self, *a, **k):
        raise RuntimeError("out of memory")

    monkeypatch.setattr(transformers.Trainer, "train", boom)
    with pytest.raises(RuntimeError, match="out of memory"):
        MLMTrainer(cfg).train()
    assert mlflow.active_run() is None
    client = mlflow.MlflowClient(tracking_uri=uri)
    assert client.search_runs([client.get_experiment_by_name("mlm_fail").experiment_id])[0].info.status == "FAILED"


def test_compare_base_vs_adapted_and_use_as_jenganlp_encoder(tmp_path, tiny_mlm_available):
    from jenga_nlp.cli import run_predict, run_train
    from jenga_nlp.core.config import ExperimentConfig
    from jenga_nlp.mlm.evaluation import compare_models, write_comparison
    from jenga_nlp.mlm.training import MLMTrainer

    _write_corpus(tmp_path)
    cfg = _config(tmp_path, max_steps=3, batch_size=4, block_size=32, save_steps=1000, device="cpu",
                  probes_path=str(tmp_path / "probes.jsonl"))
    MLMTrainer(cfg).train()

    comparison = compare_models(cfg, cfg.output_dir, device="cpu")
    domains = comparison["domains"]
    assert set(domains) == {"heldout/domain", "mpesa", "sheng"}
    assert domains["mpesa"]["sentences"] == 1 and domains["heldout/domain"]["sentences"] == 8
    assert all(v["base"] > 0 and v["adapted"] > 0 for v in domains.values())
    _, md = write_comparison(comparison, cfg.output_dir)
    assert "heldout/domain" in md.read_text()

    # The adapted model works as a JengaNLP encoder.
    data = tmp_path / "task.jsonl"
    data.write_text("\n".join(json.dumps({"text": f"ujumbe namba {i}", "label": i % 2}) for i in range(8)) + "\n")
    task_cfg = ExperimentConfig(
        project_name="downstream",
        model={"base_model": str(cfg.output_dir)},
        tokenizer={"max_length": 16},
        tasks=[{"name": "clf", "type": "single_label_classification", "data_path": str(data),
                "label_column": "label", "heads": [{"name": "h", "num_labels": 2}]}],
        training={"output_dir": str(tmp_path / "task_run"), "batch_size": 4, "num_epochs": 1, "device": "cpu",
                  "logging": {"service": "none"}, "data": {"test_size": 0.25, "seed": 0}},
    )
    run_train(task_cfg)
    result = run_predict(task_cfg, model_dir=tmp_path / "task_run", text="ujumbe namba 3", task_name="clf",
                         device="cpu")
    assert "clf" in result["task_results"]


def test_mlm_cli_parsing():
    from jenga_nlp.cli import build_parser

    parser = build_parser()
    args = parser.parse_args(["mlm", "train", "--config", "c.yaml", "--max-steps", "5", "--device", "cpu"])
    assert (args.command, args.mlm_command, args.max_steps) == ("mlm", "train", 5)
    args = parser.parse_args(["mlm", "evaluate", "--config", "c.yaml", "--model-dir", "m"])
    assert args.mlm_command == "evaluate" and args.max_sentences == 200
    args = parser.parse_args(["mlm", "tokenizer-report", "--config", "c.yaml", "--models", "a", "b"])
    assert args.models == ["a", "b"]


def test_probe_sentences_seen_in_training_are_flagged(tmp_path, tiny_mlm_available):
    from jenga_nlp.mlm.evaluation import compare_models, write_comparison

    _write_corpus(tmp_path)
    cfg = _config(tmp_path)
    leaked = prepare_corpus(cfg).train_texts[0]  # definitely in the training data
    (tmp_path / "probes.jsonl").write_text(
        json.dumps({"domain": "mpesa", "text": leaked}) + "\n"
        + json.dumps({"domain": "mpesa", "text": "Sentensi mpya kabisa isiyo kwenye data"}) + "\n"
    )
    cfg = _config(tmp_path, probes_path=str(tmp_path / "probes.jsonl"))
    comparison = compare_models(cfg, TINY_MLM, base_model=TINY_MLM, device="cpu")
    assert comparison["domains"]["mpesa"]["in_training"] == 1
    _, md = write_comparison(comparison, tmp_path)
    text = md.read_text()
    assert "1 in training" in text and "memorisation" in text


def test_sentencepiece_is_a_core_dependency():
    # castorini/afriberta_large ships only sentencepiece.bpe.model; without the package its
    # tokenizer cannot load, so every AfriBERTa config would fail in a default install.
    import tomllib

    pyproject = Path(__file__).resolve().parent.parent / "pyproject.toml"
    deps = tomllib.loads(pyproject.read_text())["project"]["dependencies"]
    assert any(d.startswith("sentencepiece") for d in deps)
