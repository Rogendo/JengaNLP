#!/usr/bin/env python3
"""Colab/Jupyter/automation notebook for JengaNLP.

This file is intentionally a notebook-style Python script:

- VS Code, Jupyter, and many notebook tools can run the `# %%` cells.
- Colab, Kaggle, RunPod, and local terminals can run the whole file with
  `python -u examples/jenganlp_colab_mlflow_tensorboard_ngrok.py`.
- CI and production automation can reuse the same file because it avoids
  Colab-only `!command` syntax.

What this notebook demonstrates:

1. optional dependency installation for notebook runtimes;
2. optional MLflow and TensorBoard service startup;
3. optional ngrok exposure for those UIs;
4. shell-command training/evaluation/prediction through `python -m jenga_nlp`;
5. Python inference by loading the saved model once with `InferencePipeline`;
6. optional Hugging Face Hub upload;
7. optional artifact zipping for download or Google Drive copy.

Important boundary:

- Notebook shell commands are good for reproducible training recipes.
- `InferencePipeline` is better for repeated notebook or production inference.
- curl is only for a running HTTP service. This notebook does not need curl.
"""

from __future__ import annotations

# %% Imports and repository path setup
import argparse
import atexit
import os
import shlex
import shutil
import socket
import subprocess
import sys
import tempfile
import textwrap
import time
import zipfile
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SERVICE_PROCESSES: list[subprocess.Popen] = []


def explain(title: str, body: str) -> None:
    """Print a notebook-style explanation block."""
    print("\n" + "=" * 96, flush=True)
    print(title, flush=True)
    print("-" * 96, flush=True)
    print(textwrap.dedent(body).strip(), flush=True)


def run_command(command: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    """Run a shell-like command safely without Colab `!` syntax."""
    print(f"\n$ {shlex.join(command)}", flush=True)
    subprocess.run(command, cwd=cwd, env=env, check=True)


def start_process(command: list[str], *, env: dict[str, str] | None = None) -> subprocess.Popen:
    """Start a background service and register it for cleanup."""
    print(f"\n$ {shlex.join(command)}", flush=True)
    process = subprocess.Popen(command, cwd=REPO_ROOT, env=env)
    SERVICE_PROCESSES.append(process)
    return process


def cleanup_services() -> None:
    """Stop background services unless the user requested to keep them."""
    for process in reversed(SERVICE_PROCESSES):
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()


atexit.register(cleanup_services)


def port_is_open(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex((host, port)) == 0


def wait_for_port(host: str, port: int, *, timeout_seconds: int = 30) -> bool:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        if port_is_open(host, port):
            return True
        time.sleep(0.5)
    return False


def command_prefix(binary: str, module: str | None = None) -> list[str]:
    """Prefer installed CLI binaries, fallback to python -m module."""
    path = shutil.which(binary)
    if path:
        return [path]
    if module:
        return [sys.executable, "-m", module]
    raise RuntimeError(f"Required command not found: {binary}")


# %% CLI arguments
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a Colab/Jupyter/automation-friendly JengaNLP workflow.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default="configs/fixture_multitask_cpu.yaml")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--model-dir", default="/tmp/jenganlp_fixture_multitask_cpu")
    parser.add_argument("--task-name", default="mpesa_fraud_detection")
    parser.add_argument("--text", default="Urgent M-PESA reversal request asking the customer to share their PIN.")
    parser.add_argument("--threshold", type=float, default=0.5)

    parser.add_argument("--install-extras", action="store_true", help="Install notebook extras: mlflow, tensorboard, pyngrok, huggingface_hub.")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--skip-cli-predict", action="store_true")
    parser.add_argument("--keep-services", action="store_true", help="Do not terminate MLflow/TensorBoard services when the script exits.")

    parser.add_argument("--start-mlflow", action="store_true")
    parser.add_argument("--mlflow-port", type=int, default=5000)
    parser.add_argument("--mlflow-backend-store-uri", default="./mlruns")
    parser.add_argument("--mlflow-proxy", action="store_true", help="Start a proxy on --mlflow-proxy-port to avoid ngrok Host header issues.")
    parser.add_argument("--mlflow-proxy-port", type=int, default=5001)

    parser.add_argument("--start-tensorboard", action="store_true")
    parser.add_argument("--tensorboard-port", type=int, default=6006)
    parser.add_argument("--tensorboard-logdir", default=None, help="Defaults to <model-dir>/logs.")

    parser.add_argument("--expose-ngrok", action="store_true")
    parser.add_argument("--ngrok-token-env", default="NGROK_AUTHTOKEN")

    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--repo-id", default=None)
    parser.add_argument("--private", action="store_true")

    parser.add_argument("--zip-results", action="store_true")
    parser.add_argument("--zip-path", default=None)
    parser.add_argument("--drive-output-dir", default=None, help="Optional Google Drive output directory, for Colab only.")
    return parser


# %% Setup helpers
def install_notebook_extras() -> None:
    explain(
        "Optional dependency installation",
        """
        This installs notebook/runtime utilities. It is intentionally opt-in so
        production environments can pin dependencies in requirements files
        instead of installing inside the script.
        """,
    )
    run_command(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--quiet",
            "mlflow",
            "tensorboard",
            "pyngrok",
            "huggingface_hub",
            "flask",
            "requests",
        ]
    )


def start_mlflow_server(port: int, backend_store_uri: str) -> None:
    explain(
        "Start MLflow",
        """
        Starting MLflow gives you a browser UI for experiment metrics and
        artifacts. The config still controls whether training logs to MLflow.
        This script sets MLFLOW_TRACKING_URI so configs without an explicit URI
        can use the local service.
        """,
    )
    os.environ["MLFLOW_TRACKING_URI"] = f"http://127.0.0.1:{port}"
    command = command_prefix("mlflow", "mlflow") + [
        "server",
        "--host",
        "0.0.0.0",
        "--port",
        str(port),
        "--backend-store-uri",
        backend_store_uri,
        "--serve-artifacts",
    ]
    start_process(command, env=os.environ.copy())
    if wait_for_port("127.0.0.1", port, timeout_seconds=30):
        print(f"MLflow is available locally at http://127.0.0.1:{port}", flush=True)
    else:
        print("MLflow did not respond within 30 seconds. Training can still continue.", flush=True)


def start_mlflow_host_header_proxy(mlflow_port: int, proxy_port: int) -> None:
    explain(
        "Start MLflow ngrok proxy",
        """
        Some MLflow versions reject ngrok requests because of Host header
        protection. This lightweight proxy forwards requests to localhost while
        rewriting the Host header. Use it only when exposing MLflow through
        ngrok and direct tunneling fails.
        """,
    )
    proxy_code = f"""
from flask import Flask, request, Response
import requests

app = Flask(__name__)
MLFLOW_URL = "http://127.0.0.1:{mlflow_port}"

@app.route("/", defaults={{"path": ""}}, methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
@app.route("/<path:path>", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
def proxy(path):
    url = f"{{MLFLOW_URL}}/{{path}}"
    headers = {{key: value for key, value in request.headers if key.lower() != "host"}}
    headers["Host"] = "127.0.0.1:{mlflow_port}"
    response = requests.request(
        method=request.method,
        url=url,
        headers=headers,
        data=request.get_data(),
        params=request.args,
        allow_redirects=False,
    )
    return Response(response.content, response.status_code, response.headers.items())

if __name__ == "__main__":
    app.run(host="0.0.0.0", port={proxy_port}, debug=False)
"""
    proxy_path = Path(tempfile.gettempdir()) / "jenganlp_mlflow_proxy.py"
    proxy_path.write_text(proxy_code)
    start_process([sys.executable, str(proxy_path)])
    if wait_for_port("127.0.0.1", proxy_port, timeout_seconds=15):
        print(f"MLflow proxy is available locally at http://127.0.0.1:{proxy_port}", flush=True)
    else:
        print("MLflow proxy did not respond within 15 seconds.", flush=True)


def start_tensorboard_server(port: int, logdir: Path) -> None:
    explain(
        "Start TensorBoard",
        """
        TensorBoard reads event files from a log directory. Your config must
        enable TensorBoard logging, either through `training.logging` or the
        pluggable `integrations` adapter layer.
        """,
    )
    logdir.mkdir(parents=True, exist_ok=True)
    command = command_prefix("tensorboard", "tensorboard.main") + [
        "--logdir",
        str(logdir),
        "--host",
        "0.0.0.0",
        "--port",
        str(port),
    ]
    start_process(command)
    if wait_for_port("127.0.0.1", port, timeout_seconds=30):
        print(f"TensorBoard is available locally at http://127.0.0.1:{port}", flush=True)
    else:
        print("TensorBoard did not respond within 30 seconds. Training can still continue.", flush=True)


def expose_ngrok(port: int, label: str, token_env: str) -> None:
    token = os.environ.get(token_env)
    if not token:
        print(f"Skipping ngrok for {label}: set {token_env}=... first.", flush=True)
        return
    try:
        from pyngrok import ngrok
    except ImportError:
        print("Skipping ngrok: pyngrok is not installed. Re-run with --install-extras.", flush=True)
        return

    ngrok.set_auth_token(token)
    tunnel = ngrok.connect(port, bind_tls=True)
    print(f"{label} public URL: {tunnel.public_url}", flush=True)


# %% Training, evaluation, and prediction helpers
def train_with_cli(config: Path, device: str) -> None:
    explain(
        "Train with shell-command semantics",
        """
        This is the notebook equivalent of terminal usage. It is easy to copy
        into Colab, Kaggle, a GPU VM, or CI. The model is saved according to
        the output_dir inside the YAML config.
        """,
    )
    run_command([sys.executable, "-m", "jenga_nlp", "train", "--config", str(config), "--device", device], cwd=REPO_ROOT)


def evaluate_with_cli(config: Path, model_dir: Path, device: str) -> None:
    explain(
        "Evaluate with shell-command semantics",
        """
        Evaluation proves the saved model can reload and compute metrics from
        the config-defined evaluation split.
        """,
    )
    run_command(
        [
            sys.executable,
            "-m",
            "jenga_nlp",
            "evaluate",
            "--config",
            str(config),
            "--model-dir",
            str(model_dir),
            "--device",
            device,
        ],
        cwd=REPO_ROOT,
    )


def predict_with_cli(config: Path, model_dir: Path, task_name: str, text: str) -> None:
    explain(
        "Predict once with CLI",
        """
        CLI prediction is useful as a smoke test. It is not the best repeated
        inference path because each command starts Python and loads the model.
        """,
    )
    run_command(
        [
            sys.executable,
            "-m",
            "jenga_nlp",
            "predict",
            "--config",
            str(config),
            "--model-dir",
            str(model_dir),
            "--task-name",
            task_name,
            "--text",
            text,
        ],
        cwd=REPO_ROOT,
    )


def predict_by_loading_once(config: Path, model_dir: Path, device: str, task_name: str, text: str, threshold: float) -> None:
    explain(
        "Load once and predict in Python",
        """
        This is the important production and automation pattern. The model is
        loaded once, then reused for many predictions. In notebooks, use this
        instead of curl unless you are explicitly testing an HTTP endpoint.
        """,
    )
    from jenga_nlp.inference.pipeline import InferencePipeline

    pipeline = InferencePipeline.from_checkpoint(model_dir=model_dir, config_path=config, device=device)
    single = pipeline.predict(text=text, task_name=task_name, threshold=threshold)
    print("Single prediction:")
    print(single.to_dict(), flush=True)

    batch = pipeline.predict_batch(
        texts=[
            text,
            "Your account payment has been received successfully.",
            "Customer asks about branch opening hours.",
        ],
        task_name=task_name,
        threshold=threshold,
        batch_size=16,
    )
    print("\nBatch predictions:")
    for item in batch:
        print(item.to_dict(), flush=True)


def push_to_hub(model_dir: Path, repo_id: str, private: bool) -> None:
    explain(
        "Optional Hugging Face Hub push",
        """
        This uploads the saved artifact directory. JengaNLP-native multi-task
        models still need `jenga_nlp` for inference because the heads are
        custom. Standard Whisper or MLM models can later be loaded directly
        with `transformers`.
        """,
    )
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if not token:
        raise RuntimeError("Set HF_TOKEN or HUGGINGFACE_HUB_TOKEN before --push-to-hub.")

    from jenga_nlp.client import JengaClient

    url = JengaClient().push_to_hub(model_dir=model_dir, repo_id=repo_id, token=token, private=private)
    print(f"Pushed to: {url}", flush=True)


def zip_results(model_dir: Path, zip_path: Path, drive_output_dir: str | None) -> None:
    explain(
        "Zip artifacts",
        """
        This is useful in Colab when you want to download model artifacts or
        copy them to Google Drive. It is also useful in automation when a CI
        job needs to archive outputs.
        """,
    )
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for file_path in model_dir.rglob("*"):
            if file_path.is_file():
                archive.write(file_path, file_path.relative_to(model_dir))
    print(f"Wrote archive: {zip_path}", flush=True)

    if drive_output_dir:
        destination = Path(drive_output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        shutil.copy2(zip_path, destination / zip_path.name)
        print(f"Copied archive to: {destination / zip_path.name}", flush=True)


# %% Main notebook flow
def main() -> None:
    args = build_parser().parse_args()
    config = Path(args.config)
    model_dir = Path(args.model_dir)
    tensorboard_logdir = Path(args.tensorboard_logdir) if args.tensorboard_logdir else model_dir / "logs"

    if not config.exists():
        raise FileNotFoundError(f"Config not found: {config}")

    if args.keep_services:
        atexit.unregister(cleanup_services)

    explain(
        "Usage boundary",
        """
        This notebook supports three practical paths:

        1. shell-command training: `python -m jenga_nlp train ...`;
        2. Python model loading: `InferencePipeline.from_checkpoint(...)`;
        3. UI exposure: MLflow/TensorBoard through local ports or ngrok.

        It intentionally avoids curl because curl is only for HTTP serving.
        For notebooks and automation, load the model artifacts directly.
        """,
    )

    if args.install_extras:
        install_notebook_extras()

    if args.start_mlflow:
        start_mlflow_server(args.mlflow_port, args.mlflow_backend_store_uri)
        if args.mlflow_proxy:
            start_mlflow_host_header_proxy(args.mlflow_port, args.mlflow_proxy_port)

    if args.start_tensorboard:
        start_tensorboard_server(args.tensorboard_port, tensorboard_logdir)

    if args.expose_ngrok:
        if args.start_mlflow:
            mlflow_ngrok_port = args.mlflow_proxy_port if args.mlflow_proxy else args.mlflow_port
            expose_ngrok(mlflow_ngrok_port, "MLflow", args.ngrok_token_env)
        if args.start_tensorboard:
            expose_ngrok(args.tensorboard_port, "TensorBoard", args.ngrok_token_env)

    if not args.skip_train:
        train_with_cli(config, args.device)
    else:
        explain("Training skipped", f"Reusing model directory: {model_dir}")

    if not args.skip_eval:
        evaluate_with_cli(config, model_dir, args.device)
    else:
        explain("Evaluation skipped", "Skipping evaluation by request.")

    if not args.skip_cli_predict:
        predict_with_cli(config, model_dir, args.task_name, args.text)

    predict_by_loading_once(config, model_dir, args.device, args.task_name, args.text, args.threshold)

    if args.push_to_hub:
        if not args.repo_id:
            raise RuntimeError("--repo-id is required with --push-to-hub")
        push_to_hub(model_dir, args.repo_id, args.private)

    if args.zip_results:
        zip_path = Path(args.zip_path) if args.zip_path else model_dir.with_suffix(".zip")
        zip_results(model_dir, zip_path, args.drive_output_dir)

    explain(
        "Done",
        f"""
        Config: {config}
        Model directory: {model_dir}

        If MLflow or TensorBoard services were started, keep this process or
        notebook runtime alive while viewing the UI. Use --keep-services only
        when you intentionally want the background processes to remain running.
        """,
    )


if __name__ == "__main__":
    main()
