# llm-tn

中文文本规整（TN）最终方案：

```text
NAR vLLM 定类
+ FSTPipePool(16 workers) 读音
```

NAR 一次前向完成逐字 25 类 BIO 标注，不再逐 Token 生成 Span。类别路由和读音规则
继续使用 wetext，与基线 v2 保持一致。

## 最终路径

| 路径 | 说明 |
| --- | --- |
| `model/nar/` | 最终 NAR 权重 |
| `model/v2/` | 原始 AR/v2 基线，保持不变 |
| `server_nar.py` | 最终 HTTP 服务：vLLM pooling + FSTPool |
| `server.py` / `main.py` | AR 基线服务与命令行入口，保留 |
| `train/nar.py` | NAR 训练、解码和推理 |
| `train/vllm_nar.py` | vLLM pooling 模型适配器 |
| `train/fst_pool.py` | 有界多进程 FSTPool |
| `data_synth/` | 最终合成训练数据与生成器 |
| `data/train_data/`、`data/train_merged.jsonl` | 原始人工训练数据 |
| `data/test_data/` | 原评测集与 SERIAL 口径基线 |

## NAR 结构

1. 25 类 BIO：12 类 × B/I + O。
2. Echo 输入：`<tn>原文<sep>原文`，只监督第二份。
3. 逐字 CharTokenizer，BIO 与字符一一对应。
4. 两步解码：先切 Span 边界，再对 Span 整段定类。
5. vLLM pooling 保留 continuous batching、KV cache 和 CUDA Graph。
6. 读音统一走 FSTPipePool。

## 启动

```bash
bash start.sh
```

默认配置：

```text
model=model/nar
dtype=bfloat16
max_num_seqs=64
FST workers=16
FST max queue=4096
```

健康检查：

```bash
curl -s localhost:8000/health
```

规整请求：

```bash
curl -s localhost:8000/normalize \
  -H 'Content-Type: application/json' \
  -d '{"text":"会议3月5日下午2点，客服电话400-820-8820","requestid":"demo"}'
```

## 训练

最终配方：

```bash
torchrun --nproc_per_node 2 -m train.nar \
    --data data/train_merged.jsonl:2 \
           data_synth/output/synth_form_coverage_round2_2800.jsonl \
           data_synth/output/synth_context_contrast_round3_1367.jsonl \
           data_synth/output/synth_chunked_from_human_and_form_1400.jsonl \
    --init model/v2 \
    --out model/nar \
    --seed 42
```

训练配方、数据口径和生成器见 `data_synth/NOTES.md`。

## 最终指标

373 条原评测集，单张 L20、bf16、FSTPool 16：

| 并发 | 吞吐 | 均值 | p99 | Tag F1 | 文本正确 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 96.5/s | 10ms | 17ms | 0.8713 | 356/373 |
| 8 | 443.7/s | 18ms | 34ms | 0.8713 | 356/373 |
| 32 | 988.3/s | 31ms | 45ms | 0.8713 | 356/373 |
| 64 | 1213.3/s | 48ms | 82ms | 0.8713 | 356/373 |
