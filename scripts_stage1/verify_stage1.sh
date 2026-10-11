#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-python}"
export MPLBACKEND=Agg
"$PYTHON" scripts_stage1/run_acceptance.py
