#!/usr/bin/env bash
# Run any module or script inside the prepared virtual environment.
# Usage: ./run_gpu.sh scripts/train_pretrain.py --steps 150 ...
set -euo pipefail
cd "$(dirname "$0")"
if [ -f .venv/bin/python ]; then
  PYTHON=.venv/bin/python
elif [ -f .venv/Scripts/python.exe ]; then
  PYTHON=.venv/Scripts/python.exe
else
  PYTHON=python3
fi
exec "$PYTHON" "$@"
