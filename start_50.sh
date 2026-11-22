#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"

# Reuse the environment and thread settings of the completed 25-epoch run.
export PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 MPLBACKEND=Agg
export CRISM_INFERENCE_PROGRESS=1 CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
export TMPDIR="$PWD/.runtime/tmp" XDG_CACHE_HOME="$PWD/.runtime/cache"
export MPLCONFIGDIR="$PWD/.runtime/matplotlib" CUDA_CACHE_PATH="$PWD/.runtime/cuda"
unset PYTHONPATH

exec .venv/bin/python -B run_experiment_50.py run
