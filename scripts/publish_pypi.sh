#!/usr/bin/env bash
# Publish jenga-nlp to the real PyPI. Releases are permanent: a version
# number can never be re-uploaded, even after deletion.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

VERSION="$(python -c 'import re;print(re.search(r"__version__ = \"(.+?)\"", open("src/jenga_nlp/__version__.py").read()).group(1))')"

echo "[1/5] Cleaning previous build artifacts"
rm -rf dist build src/jenga_nlp.egg-info

echo "[2/5] Building distributions for jenga-nlp ${VERSION}"
python -m build --sdist --wheel

echo "[3/5] Checking distributions"
python -m twine check dist/*

echo "[4/5] Confirm"
read -r -p "Upload jenga-nlp ${VERSION} to PyPI? This cannot be undone. [y/N] " ans
[[ "${ans}" == "y" || "${ans}" == "Y" ]] || { echo "Aborted."; exit 1; }

echo "[5/5] Uploading to PyPI"
python -m twine upload dist/*
