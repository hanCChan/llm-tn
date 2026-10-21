# llm-tn

中文文本规整(Text Normalization): 把书面文本里的数字/符号串读成口语文本。

```
会议3月5日下午2点，客服电话400-820-8820
  -> 会议三月五日下午两点，客服电话四零零八二零八八二零
```

架构是 **LLM 定类 + FST 读音** 两段式：

1. **定类**：SFT 过的 `Qwen3-0.6B-Base` 把句子标注成 `[(span, "CLASS:ctx"), ...]`，
   只判断"这段是什么类"，不管怎么读。共 12 类：
   `DATE TIME MONEY MEASURE FRACTION MATH SCORE PERCENT PHONE SERIAL VERSION ADDRESS`。
2. **读音**：`TNReader` 按 CLASS 把每个 span 路由到对应的 wetext 场景 normalizer，
   span 之外的普通文本用默认 normalizer 兜底，逐段拼接。

这样切分的好处是读法完全由规则决定（可控、可解释），LLM 只负责它擅长的歧义判断
——`3:1` 是比分还是时刻、`1.2.3` 是版本还是日期，靠上下文而非模式匹配。

## 目录

| 路径 | 说明 |
| --- | --- |
| `main.py` | 命令行入口，`TN` 类封装完整管线（HF transformers 推理） |
| `server.py` | FastAPI + vLLM 服务，`POST /normalize`、`GET /health` |
| `start.sh` / `Dockerfile` / `docker_build.sh` | 容器化部署 |
| `train/sft.py` | 全参 SFT 训练 |
| `train/build_dataset.py` | 样本编码（prompt 段 label 置 -100） |
| `train/nar.py` | **NAR 定类**：逐字 25 类 BIO 分类头（见下节），训练 + 推理一体 |
| `train/tn_fst.py` | `TNReader`：CLASS→normalizer 路由 + 两遍读音 |
| `train/split.py` | 长文本按标点切片（数字两侧的标点不切） |
| `train/constants.py` | 特殊 token / 分隔符 |
| `wetext/` | vendored [WeTextProcessing](https://github.com/wenet-e2e/WeTextProcessing)，含本项目新增的场景 normalizer 与规则 |
| `data/train_data/` | 训练数据，按 `positive_single` / `positive_combo` / `negative` 分场景 |
| `data/test_data/` | 评测集：`polynorm_eval.jsonl`、`hard_num.jsonl` |
| `test/eval_nar.py` | NAR 端到端评估（与 `eval_e2e.py` 同口径，可直接对比 v2） |
| `test/` | 端到端评估、服务压测、切分器不变式测试 |
| `model/v2/` | 当前使用的模型权重 |

## NAR 定类（train/nar.py，非自回归标注）

`main.py` / `server.py` 的 v2 管线是自回归生成：span 逐 token 抄写，延迟随 span 数量/
长度线性增长。`train/nar.py` 把定类任务重构为**一次前向的逐字序列标注**，与 v2 用同一
套 `TNReader` 读音、同一份数据，可独立训练与评测：

1. **25 类 BIO 头**：12 类 × B/I + O。span 不再生成，由标签位置直接切出。
2. **Echo 复读输入**：`<tn>原文<sep>原文`，只监督第二份 —— 因果注意力下第二份每个字
   都能看到第一份完整全文，等效双向上下文。
3. **CharTokenizer 逐字分词**：每个字符恰好一个 token，BIO 标签与字符一一对应
   （原生 BPE 有 ~4% span 边界骑在 token 上，标签无解）。
4. **两步解码 + 结构修复**：先按 P(非O) 切边界、再对 span 内类别概率求和整段定一次类
   （避免逐字 argmax 的类别跳变）；之后按配对/前缀规则修正括号、币种代码等结构错误。

```bash
# 训练（两卡；人工数据 ×2 = 与 v2 SFT 2 epoch 等监督量）
torchrun --nproc_per_node 2 -m train.nar \
    --data data/train_merged.jsonl:2 --init model/v2 --out runs/nar_human

# 评测（单卡，与 test/eval_e2e.py 同口径）
python test/eval_nar.py --model runs/nar_human

# 推理
python -c "from train.nar import NARRunner; \
    print(NARRunner('runs/nar_human').tag(['比分3:1获胜']))"
```

实测（373 条测试集，单条整句、无合批；延迟含读音）：

| 指标 | v2（自回归） | NAR |
| --- | --- | --- |
| Tag micro F1 | 0.826 | 0.778 |
| 句级完全正确 | 313/373 | 300/373 |
| 单条 p50 / p99 | 220 / 307 ms | **20 / 29 ms** |

质量差距集中在 SERIAL/SCORE 等依赖更多训练样本的类别（待数据扩增补充）；
延迟收益来自取消逐 token 生成。后续服务端优化（动态合批、CUDA Graph 等）
在此基础上另行叠加。