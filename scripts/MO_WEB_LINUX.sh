#!/usr/bin/env bash
set -euo pipefail

# Chuyen tu scripts ve thu muc goc chua app.py.
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
python -m streamlit run app.py
