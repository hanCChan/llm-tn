# data_synth/ —— 合成训练数据(标签由构造决定, 无蒸馏)

> 2026-10-22 从消融仓库(ceshi, experiment/v1-ablation)移植。消融完整记录在该仓库
> 07-recipe-ablation/NOTES.md(12 组配方 × 3 seed, 盲测一次); 本目录只保留**赢家配方**
> 的生成器与最终产物, 不含中间轮次。

## 为什么不用蒸馏

旧方案(v2 teacher 给扩增句打伪标签)被弃用: 单 seed 噪声内无净收益, 且 teacher 的系统性
标注错误(括号边界缺失等)会直接灌进学生模型。本方案标签由**构造**决定 —— 程序把值插进
句子的哪个位置, span 就标在哪, 每条样本可审计, 不经过任何模型。

## 数据怎么造(三步)

1. **值文法** `value_forms.py`: 12 类拆 47 子类、201 种书写形态, 按"真实中文文本里有哪些
   写法"全量枚举(ISO 币种码全表、量词全表、括号/空格/短横分组的电话、am/pm 各写法、
   车牌/证件/单号、V 前缀版本…), 数值随机、不复用评测值。**等概率采样, 不按评测集分布配比。**
2. **句子框架** 两个来源: 人工训练句挖空回填(语境保持人工原样, 只换书写形态 → 不引入
   新的"关键词→类别"捷径) + `sentence_frames/`(隔离写作的框架, 只给口径文档, 补人工数据
   没有的子类语境: 时长、am/pm、国际电话、证件号…)。
3. **语境对比**(`--kind contrast`): 同一个外形歧义的值(`3:15`、`2/3`、11 位数字、字母码+数字)
   放进**两个不同类别**的框架成对出现, 标签由语义决定 —— 防止模型学"外形→类别"。

每条样本过四道检查, 不过即丢: 粘连(边界不唯一) / FST 读音残留数字 / 与评测集共享 8 字
片段 / 重复。口径定义见 `label_spec.md`(标注口径 v2, 从人工训练集习惯归纳)。

## 最终配方(消融赢家 D_r4, test373/dev/盲测全面不输 v2 AR)

| 文件(data_synth/output/) | 条数 | 内容 |
|---|---|---|
| `synth_form_coverage_round2_2800.jsonl` | 2800 | 形态覆盖切片(第 2 轮文法) |
| `synth_context_contrast_round3_1367.jsonl` | 1367 | 语境对比切片(第 3 轮: 平衡成对族) |
| `synth_chunked_from_human_and_form_1400.jsonl` | 1400 | 线上切片器(train/split.py)切出的片段样本, 含片首对齐 |

训练命令(权重 `runs/nar_best`, seed 42):

```bash
torchrun --nproc_per_node 2 -m train.nar \
    --data data/train_merged.jsonl:2 \
           data_synth/output/synth_form_coverage_round2_2800.jsonl \
           data_synth/output/synth_context_contrast_round3_1367.jsonl \
           data_synth/output/synth_chunked_from_human_and_form_1400.jsonl \
    --init model/v2 --out runs/nar_best --seed 42
```

为什么有切片样本: 线上服务先按标点切片再送模型, 输入大量是"数字打头"的片; 训练句几乎
全是整句(人工 4167 条里 span 在句首仅 95 条), 模型在片首会丢首字母(`AUD100`→`UD100`)。
切片样本用**线上同一个 split_chunks** 生成, 消除训练/线上输入分布不一致。

## 质量数字(runs/nar_best, 单 seed 42, 2026-10-22 本机)

| 评测集 | TagF1 | 句级 | 文本 | 对照 v2 AR |
|---|---|---|---|---|
| test373(gold_v2_serial 口径) | **0.896** | 338/373 | 358/373 | AR 0.852 / 321 / 359 |
| test373(原口径, SERIAL 只截数字) | 0.865 | 328/373 | 358/373 | AR 0.867 |
| 延迟(冻结 30 条, batch=1) | p50 15.9ms / p99 17.5ms(前向 16.1 + 读音 6.6) | | | AR p50 216 / p99 296 |
| 吞吐(batch=16 引擎层) | 63.6 句/s | | | AR 无批路径(逐 token 生成) |

**并发对照(2026-10-22, 373 句测试集)** —— 两者现在不在同一条起跑线上, 读数字前先看口径:

| | v2 AR + server.py(vLLM continuous batching, 专用服务端) | NAR(nar_best, 引擎串行 + 请求排队) |
|---|---|---|
| c1 p50 / p99 | 41 / 98 ms | **15 / 33 ms** |
| c8 吞吐 / p99 | 131 句/s / 120 ms | 60 句/s / 138 ms |
| c16 吞吐 / p99 | 184 句/s / 158 ms | 59 句/s / 277 ms |
| c32 吞吐 / p99 | 185 句/s / 283 ms | 55 句/s / 662 ms |
| 同档位输出漂移 | 各并发间 TagF1 ±0.003(合批抖动) | 0(串行) |

NAR 并发吞吐只有 60 句/s 是**结构性的**: 推理引擎是单实例串行(每次一个请求一次前向),
多请求只能排队 —— 64 在途时 p99 到 1.08s。这不是模型问题, 是还没做服务端化:
定版 llm-tn1(cankao) 的 NARBatcher + CUDA Graph + 合批把同硬件做到过 c64 1847 QPS /
p99 44ms, 那是 GPU 阶段(合批/图捕获)的账, 数据工程阶段不碰。AR 的 vLLM 服务端是
现成的, 所以并发上 AR 占优; 单请求延迟(用户体感)NAR 低 63%。

原口径下 0.865 vs AR 0.867 的差距全部来自 14 处 SERIAL 口径差(模型按训练口径标
`S23` 整体, 原测试 gold 是 `23`)—— 文本正确数完全相同(358/373), 读音不受影响。
逐类对照(vs AR): NAR 明显更好 MEASURE .914/.809、MATH .944/.862、DATE .970/.909、
FRACTION 1.0/.800、ADDRESS 1.0/.981; 略差 PHONE .721/.740、VERSION .750/.923。
句级互有胜负: NAR 独有对 30 句、AR 独有对 13 句。
阶段压测脚本: `test/bench_stage.py`(`--nar runs/nar_best` 或 `--ar`, 含并发档位);
基线服务端压测沿用 `test/bench_server.py`。

## 文件清单

| 文件 | 用途 |
|---|---|
| `label_spec.md` | 标注口径 v2(训练生成、评测、gold 修订共用) |
| `value_forms.py` | 值文法(类×子类×书写形态) |
| `sentence_frames/frames_{A..E}.jsonl` | 隔离写作的句子框架(1325 条) |
| `build_synth_data.py` | 构建器: `--kind form|contrast --round N --seed S` |
| `make_chunk_samples.py` | 线上切片样本生成器 |
| `build_final_data.sh` | 一键重产最终三份文件(种子固定, 可复现) |
| `output/` | 生成产物(含完整池 8334/4168 与使用切片) |
