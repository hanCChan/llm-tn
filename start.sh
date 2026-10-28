#!/usr/bin/env bash
# 容器启动脚本: 拉起最终方案服务(Dockerfile 的 ENTRYPOINT)。
# 单阶段 A1 + N-gram S3 + FSTPipePool; model/a1 已包含 12 个类别 special token。
#
# 部署机型 L20(Ada, SM89)原生支持 bf16, 所以 DTYPE 默认 bfloat16;
# 换到 T4(SM75)不必改镜像, 加 -e DTYPE=float16 即可。
#
# 所有参数都能用环境变量覆盖:
#   docker run -d --gpus all -p 8000:8000 --name tn <image>
#   docker run -d --gpus all -p 8000:8000 -e DTYPE=float16 -e GPU_UTIL=0.9 <image>
#   docker run -d --gpus all -p 8000:8000 --name tn -e FST_WORKERS=4 <image>
# 也可以直接换成别的命令进容器排查:
#   docker run -it --gpus all --entrypoint bash <image>

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${WORKSPACE:-$SCRIPT_DIR}"

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
MODEL="${MODEL:-model/a1}"
DTYPE="${DTYPE:-bfloat16}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-2048}"
# L20 有 48G, 0.6B 模型本身才 1.2G, 0.3 的 KV cache 已远超 TN 短句需求。
GPU_UTIL="${GPU_UTIL:-0.3}"
MAX_NEW="${MAX_NEW:-128}"
# FST 是多进程 CPU 路径：worker 数、绑核策略、队列上限和超时都可独立调参。
# LOG_MODE=async 把 JSON 格式化/写盘移到后台线程，避免高并发时阻塞事件循环。
FST_WORKERS="${FST_WORKERS:-4}"
FST_CPUS="${FST_CPUS:-auto}"
FST_MAX_QUEUE="${FST_MAX_QUEUE:-1024}"
FST_TIMEOUT="${FST_TIMEOUT:-30}"
LOG_MODE="${LOG_MODE:-async}"
TN_SPEC_MODE="${TN_SPEC_MODE:-ngram}"
TN_SPEC_K="${TN_SPEC_K:-4}"
TN_SPEC_MIN="${TN_SPEC_MIN:-1}"
TN_SPEC_MAX="${TN_SPEC_MAX:-4}"
if [[ -x "${PYTHON_BIN:-}" ]]; then
    :
elif [[ -x .venv/bin/python ]]; then
    PYTHON_BIN=".venv/bin/python"
else
    PYTHON_BIN="python3"
fi

# vLLM v1 引擎跑在子进程里, fork 会和已初始化的 CUDA 冲突。
export VLLM_WORKER_MULTIPROC_METHOD=spawn
# 权重已打进镜像, 生产网络不通 hf.co 时联网探测会白等几十秒。
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
# 关掉匿名用量上报, 同样是为了不在无外网环境里干等超时。
export VLLM_NO_USAGE_STATS=1
export DO_NOT_TRACK=1

mkdir -p log

echo "[start] $(date '+%F %T') host=${HOST}:${PORT} model=${MODEL} dtype=${DTYPE} gpu_util=${GPU_UTIL} fst_workers=${FST_WORKERS}"
# 打一行卡型, dtype 报错时(比如 bf16 落到 SM75)第一时间能对上。
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || \
    echo "[start] nvidia-smi 不可用, 检查 docker run 是否带了 --gpus"

# exec 让 python 成为 PID 1, docker stop 的 SIGTERM 才能直达 uvicorn 走优雅关闭
# (lifespan 会依次停止 vLLM、FST 进程池和异步日志 listener)。
#
# stdout/stderr 丢到 /dev/null: 业务日志由 logger 包结构化写进 log/app.log, 这里
# 只剩 vLLM/uvicorn 的启动噪音, 不接采集就没必要占容器磁盘。
# 代价是启动失败的 traceback 也一起没了 —— 排查时改成
#   docker run -it --gpus all --entrypoint bash <image>
# 手动跑 python3 server.py 看输出, 或临时把下面的重定向去掉。
exec "${PYTHON_BIN}" server.py \
    --host "${HOST}" \
    --port "${PORT}" \
    --model "${MODEL}" \
    --dtype "${DTYPE}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --gpu-util "${GPU_UTIL}" \
    --max-new "${MAX_NEW}" \
    --fst-workers "${FST_WORKERS}" \
    --fst-cpus "${FST_CPUS}" \
    --fst-max-queue "${FST_MAX_QUEUE}" \
    --fst-timeout "${FST_TIMEOUT}" \
    --log-mode "${LOG_MODE}" \
    --spec-mode "${TN_SPEC_MODE}" \
    --spec-k "${TN_SPEC_K}" \
    --spec-min "${TN_SPEC_MIN}" \
    --spec-max "${TN_SPEC_MAX}" \
    >/dev/null 2>&1
