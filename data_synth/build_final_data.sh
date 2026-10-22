#!/usr/bin/env bash
# 生成最终配方的三份合成训练数据 + 各切片(与消融赢家 D_r4 完全一致的成分与配比)。
# 产物(data_synth/output/, 文件名即内容):
#   synth_form_coverage_round2_8334.jsonl     形态覆盖池(第 2 轮文法: 币种全量/量词/枚举…)
#   synth_context_contrast_round3_4168.jsonl  语境对比池(第 3 轮: 平衡成对族, 产品负例)
#   synth_chunked_from_human_and_form_1400.jsonl  线上切片器切出的片段(含片首对齐)
#   *_2800 / *_1367                           消融定下的使用量切片
# 配方(train/nar.py --data): data/train_merged.jsonl:2 + 前两个池的切片 + 切片样本
set -euo pipefail
cd "$(dirname "$0")/.."
PY=/data/miniforge/envs/llm-tn/bin/python

# 种子与消融仓库一致, 产物逐字节可复现(sha 见 out 审计)
$PY data_synth/build_synth_data.py --kind form --n 8334 --seed 11 --round 2 2>&1 | grep -v WETEXT | head -3 &
$PY data_synth/build_synth_data.py --kind contrast --n 4168 --seed 13 --round 3 2>&1 | grep -v WETEXT | head -3 &
wait
$PY data_synth/make_chunk_samples.py --n 1400 --seed 21 2>&1 | grep -v WETEXT | tail -3

$PY - <<'EOF'
import json, os
T = "data_synth/output"
form = [l for l in open(f"{T}/synth_form_coverage_round2_8334.jsonl")
        if json.loads(l)["meta"].get("form") != "points"]      # 口径未定义"X个百分点", 不教
con = open(f"{T}/synth_context_contrast_round3_4168.jsonl").readlines()
for name, lines in (("synth_form_coverage_round2_2800", form[:2800]),
                    ("synth_context_contrast_round3_1367", con[:1367])):
    open(f"{T}/{name}.jsonl", "w").writelines(lines)
    print(name, len(lines))
EOF
ls -la data_synth/output/
