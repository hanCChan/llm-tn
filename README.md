# llm-tn

中文文本规整（Text Normalization）最终方案：

```text
单阶段 A1 类别 Token 定类
+ vLLM N-gram S3 推测解码
+ FSTPipePool(4 workers) 读音
```

模型只决定数字 Span 的类别，FST 根据类别生成稳定读音。

## A1 协议

v2 输出带类别和上下文的文本：

```text
110<TAB>PHONE:phone
110人<TAB>MEASURE:measure
```

A1 输出 Span 加专用类别 Token：

```text
110<|TN_PHONE|>110人<|TN_MEASURE|><EOS>
```

12 个类别 Token：

```text
<|TN_ADDRESS|>  <|TN_DATE|>     <|TN_FRACTION|> <|TN_MATH|>
<|TN_MEASURE|>  <|TN_MONEY|>    <|TN_PERCENT|>  <|TN_PHONE|>
<|TN_SCORE|>    <|TN_SERIAL|>   <|TN_TIME|>     <|TN_VERSION|>
```

类别 Token 同时承担记录边界。推理必须读取 `output.token_ids`，不能依赖会过滤
special token 的 `output.text`。

## 模型

| 路径 | 用途 |
| --- | --- |
| `model/v2/` | 原始 v2 AR 基线，保持不变 |
| `model/a1/` | 最终单阶段 A1 模型 |

A1 冻结 Qwen 底座，只训练 12 个类别 Token 对应的 tied embedding/LM-head
行，并使用 `2x` 边界加权 CE。

## 训练

最终训练数据：

```text
data/train_short.jsonl       原始 4167 条，类别已归一化为 12 类
data/train_augmented.jsonl   组合增强 1000 条
```

从 v2 单阶段训练：

```bash
CUDA_VISIBLE_DEVICES=0 python -m train.sft \
    --model model/v2 \
    --data data/train_short.jsonl data/train_augmented.jsonl \
    --out out/sft/a1 \
    --epochs 4 \
    --lr 1e-3 \
    --global-batch 32 \
    --per-device 8 \
    --val-ratio 0.1 \
    --boundary-weight 2.0
```

最终权重放在 `model/a1/`。

## 推理

HTTP 服务：

```bash
bash start.sh
```

默认配置：

```text
model/a1
N-gram method
num_speculative_tokens=4
prompt_lookup_min=1
prompt_lookup_max=4
FST workers=4
```

`start.sh` 中的 `TN_SPEC_MODE=off` 可用于显式回滚推测解码；其余模型、FST 和
采样参数不变。

命令行单条推理：

```bash
python main.py "报警拨打110来了110人"
```

逐阶段查看 LaTeX 改写、切片、模型 prompt/token、解析后的 edits 和 FST 路由：

```bash
python infer_trace.py "客服电话400-820-8820，会议3月5日下午2点开始，胜率75%。"
```

HTTP 示例：

```bash
curl -s localhost:8000/normalize \
    -H 'Content-Type: application/json' \
    -d '{"text":"客服电话400-820-8820","requestid":"demo"}'
```

## 最终指标

HF FP32 原评测口径：

```text
Tag micro F1: 0.868
句级完全正确: 328/373
口语逐字正确: 358/373
```

HTTP + S3 + FSTPool，373 条：

| 并发 | 吞吐 | 均值 | p99 | Tag F1 | Spoken |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 49.2/s | 20ms | 37ms | 0.8534 | 353/373 |
| 8 | 319.4/s | 25ms | 44ms | 0.8508 | 353/373 |
| 32 | 711.5/s | 43ms | 68ms | 0.8508 | 353/373 |

2026-10-28 使用 aiohttp 客户端复核（同一 `model/a1`、N-gram S3、FSTPool4）：

| 并发 | 吞吐 | 均值 | p50 | p99 | Tag F1 | Spoken |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 47.1/s | 21ms | 20ms | 39ms | 0.8534 | 353/373 |
| 8 | 315.5/s | 25ms | 24ms | 47ms | 0.8508 | 353/373 |
| 32 | 739.5/s | 41ms | 40ms | 65ms | 0.8508 | 353/373 |
| 64 | 743.4/s | 79ms | 78ms | 107ms | 0.8508 | 353/373 |

同服务 c32 对照：httpx 为 622.4/s、p99 78ms；aiohttp 吞吐高约 19%。
300 QPS Poisson 开环实际 298.6 QPS，均值 29.0ms，p95 46.4ms，p99 54.6ms，
0 错误。长度负载 aiohttp 复核的 c32 吞吐为：短句 641.5/s、85 字 139.0/s、
512 字 93.3/s、混合 164.1/s。

400 QPS Poisson 开环：实际 `399.8 QPS`，均值 `28.7ms`，p95 `44.6ms`，
p99 `51.7ms`，0 错误。N-gram 接受率 `72.76%`，平均接受长度 `3.911`。

长度负载每档 96 请求：

| 类型 | 并发 | 均值 | p50 | p95 | p99 | 吞吐 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 短句 | 1 | 17ms | 16ms | 21ms | 25ms | 59.4/s |
| 短句 | 8 | 22ms | 21ms | 30ms | 34ms | 354.1/s |
| 短句 | 32 | 39ms | 37ms | 61ms | 64ms | 711.2/s |
| 85 字 | 1 | 36ms | 35ms | 47ms | 56ms | 27.9/s |
| 85 字 | 8 | 49ms | 47ms | 70ms | 83ms | 158.0/s |
| 85 字 | 32 | 159ms | 153ms | 234ms | 256ms | 175.2/s |
| 512 字 | 1 | 46ms | 45ms | 63ms | 67ms | 21.7/s |
| 512 字 | 8 | 75ms | 72ms | 101ms | 128ms | 102.4/s |
| 512 字 | 32 | 282ms | 289ms | 374ms | 467ms | 97.9/s |
| 混合 | 1 | 33ms | 33ms | 55ms | 65ms | 30.3/s |
| 混合 | 8 | 47ms | 46ms | 68ms | 83ms | 157.8/s |
| 混合 | 32 | 155ms | 154ms | 225ms | 279ms | 175.2/s |

短句和 85 字由 LLM 主导；512 字 c32 的 LLM 均值约 `74ms`，FST 均值约
`194ms`，长文本高并发瓶颈已转移到 FSTPipePool。

已知边界：`5G` 可能被误标为 SERIAL，`XXI` 可能被误标为 DATE，但当前
Spoken 文本不受影响。`PHONE/SERIAL` 的“幺/一”差异属于已接受读音变体。
