#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

echo "[1/4] Cleaning previous build artifacts"
rm -rf dist build src/jenga_nlp.egg-info

echo "[2/4] Building distributions"
python -m build --sdist --wheel

echo "[3/4] Checking distributions"
python -m twine check dist/*

echo "[4/4] Uploading to Test PyPI"
python -m twine upload --repository testpypi dist/*
