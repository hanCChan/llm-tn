# =============================================================================
# llm-tn source header
# Path: Dockerfile
# 职责: 容器构建或服务启动配置
# 核心原理: 遵循仓库现有的 A1 类别 token 与 pynini FST 管线，注释仅解释实现，不改变运行行为。
# =============================================================================

FROM aidockorg/python-cuda:3.10-v2-cuda-11.8.0-cudnn8-runtime-22.04

USER root

# 设置环境变量
ARG PROG_NAME=llm-tn
ENV WORKSPACE="/home/oicq/${PROG_NAME}" \
    PYTHONPATH="${WORKSPACE}" \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    CUDA_VISIBLE_DEVICES=0 \
    PYTHONUNBUFFERED=1

# 设置时区环境变量避免交互
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Shanghai

# 安装系统依赖并设置时区
RUN apt-get update && \
    apt-get install -y iputils-ping cron tar tzdata vim net-tools sox libsox-dev unzip bc curl tree lrzsz pkg-config && \
    ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# 创建工作目录
WORKDIR $WORKSPACE
RUN mkdir -p $WORKSPACE/log

# 依赖单独一层: requirements.txt 不变时改代码不会重装 torch/vllm。
# torch/vllm 的 wheel 自带 CUDA 运行库, 与基础镜像的 CUDA 11.8 无关, 只要求宿主
# 机驱动够新; pynini wheel 已静态链接 OpenFst, 无需在镜像里编译。
COPY requirements.txt $WORKSPACE/
RUN pip install --no-cache-dir -r requirements.txt

# vLLM v1 引擎跑在子进程里, fork 会和 CUDA 冲突(server.py 内也设了一次兜底)。
ENV VLLM_WORKER_MULTIPROC_METHOD=spawn

# model/v2 保留基线, model/a1 是最终单阶段 A1; 单独一层便于复用缓存。
COPY model $WORKSPACE/model

COPY server.py main.py $WORKSPACE/
COPY train $WORKSPACE/train
COPY wetext $WORKSPACE/wetext
COPY data $WORKSPACE/data
COPY logger $WORKSPACE/logger

COPY start.sh $WORKSPACE/
RUN chmod +x $WORKSPACE/start.sh

# 部署机型 L20(Ada, SM89): start.sh 里 DTYPE 默认 bfloat16。
#   docker run -d --gpus all -p 8000:8000 --name tn llm-tn
#   docker run -d --gpus all -p 8000:8000 -e DTYPE=float16 llm-tn   # 落到 T4 时
#   docker run -it --gpus all --entrypoint bash llm-tn              # 进容器排查
#   docker exec -it tn bash
#   docker exec -it tn tail -f log/app.log                          # 看业务日志
ENTRYPOINT ["/home/oicq/llm-tn/start.sh"]
