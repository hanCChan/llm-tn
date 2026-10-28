# NAR 合成训练数据

最终 NAR 配方使用三份合成数据加原始人工数据回放。标签由生成结构直接确定，
不使用 teacher 模型蒸馏，也不复制评测样本。

## 最终训练文件

| 文件 | 条数 |
| --- | ---: |
| `output/synth_form_coverage_round2_2800.jsonl` | 2800 |
| `output/synth_context_contrast_round3_1367.jsonl` | 1367 |
| `output/synth_chunked_from_human_and_form_1400.jsonl` | 1400 |
| `../data/train_merged.jsonl` | 4167 × 2 |

## 生成结构

1. `value_forms.py`：12 类、书写形态和值文法。
2. `sentence_frames/`：不依赖评测文本的句子框架。
3. `build_synth_data.py`：生成形态覆盖和上下文对比样本。
4. `make_chunk_samples.py`：使用线上同一个 `split_chunks` 生成片首样本。
5. `build_final_data.sh`：固定 seed 一键重建最终数据。

生成时会过滤 Span 不唯一、FST 读音残留数字、与评测集重复以及完全重复的样本。

## 训练

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

完整标注口径见 `label_spec.md`。
