#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
export PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 MPLBACKEND=Agg
export CRISM_INFERENCE_PROGRESS=1
unset PYTHONPATH
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
export TMPDIR="$PWD/.runtime/tmp" XDG_CACHE_HOME="$PWD/.runtime/cache"
export MPLCONFIGDIR="$PWD/.runtime/matplotlib" CUDA_CACHE_PATH="$PWD/.runtime/cuda"
export UV_PYTHON_INSTALL_DIR="$PWD/.runtime/python" UV_CACHE_DIR="$PWD/.runtime/uv-cache"
export PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
mkdir -p "$TMPDIR" "$MPLCONFIGDIR" "$CUDA_CACHE_PATH"
exec 9>.runtime/setup.lock
flock -n 9 || { echo 'Experiment already active in this directory.'; exit 1; }
[[ "$(uname -sm)" = 'Linux x86_64' ]] || { echo 'Expected Linux x86_64.'; exit 1; }
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
[[ -f workspace/input_inventory.json ]] || { echo 'Missing workspace/input_inventory.json'; exit 1; }
if ! .venv/bin/python -m pip --version >/dev/null 2>&1; then
  python3 - <<'PY'
import shutil
if shutil.disk_usage('.').free < 37_000_000_000:
    raise SystemExit('Fresh installation requires 37 GB free after uploading the workspace.')
PY
  if command -v python3.11 >/dev/null && python3.11 -m venv .venv; then
    :
  elif command -v python3.12 >/dev/null && python3.12 -m venv .venv; then
    :
  else
    # Bootstrap managed CPython on base images that only have the OS Python.
    python3 - <<'PY'
import hashlib, io, json, urllib.request, zipfile
from pathlib import Path
index = 'https://pypi.org/pypi/uv/0.8.22/json'
with urllib.request.urlopen(index, timeout=60) as response:
    wheels = json.load(response)['urls']
wheel, = [w for w in wheels if w['filename'].endswith('manylinux2014_x86_64.whl')]
with urllib.request.urlopen(wheel['url'], timeout=180) as response:
    data = response.read()
if hashlib.sha256(data).hexdigest() != wheel['digests']['sha256']:
    raise ValueError('uv download SHA-256 mismatch')
with zipfile.ZipFile(io.BytesIO(data)) as archive:
    name, = [n for n in archive.namelist() if n.endswith('/scripts/uv')]
    dest = Path('.runtime/uv')
    dest.write_bytes(archive.read(name))
    dest.chmod(0o755)
PY
    .runtime/uv python install 3.11
    .runtime/uv venv --allow-existing --seed --python "$(.runtime/uv python find --managed-python 3.11)" .venv
  fi
fi
py=.venv/bin/python
if ! "$py" -c 'import torch; assert torch.__version__ == "2.6.0+cu124"' 2>/dev/null; then
  "$py" -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu124 'torch==2.6.0'
fi
# Spectral Python 0.25 is a pure-Python source distribution on PyPI.
"$py" -m pip install --no-cache-dir --only-binary=:all: --no-binary=spectral -r requirements.txt
"$py" -m pip check
"$py" -m pip freeze > .runtime/environment.txt
exec "$py" run_experiment.py run
