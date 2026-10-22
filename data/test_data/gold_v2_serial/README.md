# gold_v2_serial/ —— SERIAL 口径 v2 版测试集(与上级目录的原测试集区分)

上级目录的 `polynorm_eval.jsonl` / `hard_num.jsonl` 是**原始口径**: SERIAL 的 gold 只截
数字部分(`Galaxy S23` 标 `23`、`空中客车A330` 标 `330`)。而人工训练集 303/303 个 SERIAL
都是"字母+数字整体"(`A368`、`晋D·28711`)—— 训练与评测口径相反, 模型照训练学会的答案会被判错。

本目录是同一批句子、唯一改动 = SERIAL span 补全为字母+数字整体:

| 文件 | 内容 |
|---|---|
| `test_polynorm195_goldv2.jsonl` | = polynorm_eval.jsonl, 7 处 SERIAL 修订 |
| `test_hardnum178_goldv2.jsonl` | = hard_num.jsonl, 7 处 SERIAL 修订 |
| `gold_v2_changes.tsv` | 逐条修订登记(旧片段 → 新片段 → 读音是否不变) |

14 处修订全部通过读音校验: 新片段下 FST 读出的整句口语 == 人工 gold_spoken 原文
(即只是标签边界口径对齐训练集, 读音答案没有任何变化)。

**用哪个**:
- 与历史 v2/AR 数字对比 → 用上级目录原文件;
- 评测 NAR(按训练口径学的 SERIAL) → 用本目录, 否则模型标对的 `S23` 会被算错。

数据集构建脚本与完整口径文档见消融仓库(ceshi); 口径定义 `data_synth/label_spec.md`。
