#!/usr/bin/env bash

# set -ex

# 打镜像
version=$(date "+%Y%m%d%H%M%S")
version=llm_tn_${version}
echo ${version}

#docker login --username=leolxliu https://csighub.tencentyun.com --password=leolxliu

docker build --network=host -t csighub.tencentyun.com/avchat/llm-tn:${version} ./
docker push csighub.tencentyun.com/avchat/llm-tn:${version}