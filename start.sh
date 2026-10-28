#!/usr/bin/env bash
# 最终上线入口: NAR vLLM pooling + FSTPipePool。
# 模型固定为 model/nar；model/v2 与 AR 服务代码保留为只读基线。

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${WORKSPACE:-$SCRIPT_DIR}"

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
MODEL="${MODEL:-model/nar}"
DTYPE="${DTYPE:-bfloat16}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-2048}"
GPU_UTIL="${GPU_UTIL:-0.3}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"
ENFORCE_EAGER="${ENFORCE_EAGER:-0}"
FST_WORKERS="${FST_WORKERS:-16}"
FST_CPUS="${FST_CPUS:-auto}"
FST_MAX_QUEUE="${FST_MAX_QUEUE:-4096}"
DEADLINE_MS="${DEADLINE_MS:-2000}"
EAGER_ARGS=()
if [[ "${ENFORCE_EAGER}" == "1" ]]; then
    EAGER_ARGS+=(--enforce-eager)
fi

if [[ -x "${PYTHON_BIN:-}" ]]; then
    :
elif [[ -x .venv/bin/python ]]; then
    PYTHON_BIN=".venv/bin/python"
else
    PYTHON_BIN="python3"
fi

export VLLM_WORKER_MULTIPROC_METHOD=spawn
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export VLLM_NO_USAGE_STATS=1
export DO_NOT_TRACK=1

mkdir -p log

echo "[start] $(date '+%F %T') host=${HOST}:${PORT} model=${MODEL} dtype=${DTYPE} max_num_seqs=${MAX_NUM_SEQS} fst_workers=${FST_WORKERS}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || \
    echo "[start] nvidia-smi unavailable"

exec "${PYTHON_BIN}" server_nar.py \
    --host "${HOST}" \
    --port "${PORT}" \
    --model "${MODEL}" \
    --dtype "${DTYPE}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --gpu-util "${GPU_UTIL}" \
    --max-num-seqs "${MAX_NUM_SEQS}" \
    "${EAGER_ARGS[@]}" \
    --fst-workers "${FST_WORKERS}" \
    --fst-cpus "${FST_CPUS}" \
    --fst-max-queue "${FST_MAX_QUEUE}" \
    --deadline-ms "${DEADLINE_MS}" \
    >/dev/null 2>&1
