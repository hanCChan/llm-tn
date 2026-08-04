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
| `train/tn_fst.py` | `TNReader`：CLASS→normalizer 路由 + 两遍读音 |
| `train/split.py` | 长文本按标点切片（数字两侧的标点不切） |
| `train/constants.py` | 特殊 token / 分隔符 |
| `wetext/` | vendored [WeTextProcessing](https://github.com/wenet-e2e/WeTextProcessing)，含本项目新增的场景 normalizer 与规则 |
| `data/train_data/` | 训练数据，按 `positive_single` / `positive_combo` / `negative` 分场景 |
| `data/test_data/` | 评测集：`polynorm_eval.jsonl`、`hard_num.jsonl` |
| `test/` | 端到端评估、服务压测、切分器不变式测试 |
| `model/v2/` | 当前使用的模型权重 |