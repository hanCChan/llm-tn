#!/usr/bin/env bash

# =============================================================================
# llm-tn source header
# Path: docker_build.sh
# 职责: 容器构建或服务启动配置
# 核心原理: 遵循仓库现有的 A1 类别 token 与 pynini FST 管线，注释仅解释实现，不改变运行行为。
# =============================================================================

# set -ex

# 打镜像
version=$(date "+%Y%m%d%H%M%S")
version=llm_tn_${version}
echo ${version}

#docker login --username=leolxliu https://csighub.tencentyun.com --password=leolxliu

docker build --network=host -t csighub.tencentyun.com/avchat/llm-tn:${version} ./
docker push csighub.tencentyun.com/avchat/llm-tn:${version}