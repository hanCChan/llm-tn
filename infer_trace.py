#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =============================================================================
# llm-tn source header
# Path: infer_trace.py
# 职责: 单条文本的 A1 推理全流程追踪
# 核心原理: 依次打印 LaTeX 改写、切片、LLM token、标签解析和 FST 路由的中间值。
# =============================================================================

"""跟踪一条文本从原文到口语文本的完整处理过程。

总体怎么做
==========
1. 用 Hugging Face FP32 加载 ``model/a1``，与 ``main.py`` 的确定性推理一致。
2. 依次执行 ``delatex -> split_chunks -> has_digit``，展示模型实际看到的片。
3. 对每个含数字的片构造 prompt，打印输入 token ids 和模型生成的原始 token。
4. 从生成 token ids 解析 ``[[span, CLASS], ...]``，不使用会过滤 special token
   的 ``output.text``。
5. 逐片调用 ``TNReader.apply``，打印类别路由、场景读音结果和最终拼接文本。

核心函数
========
* ``show``: 统一打印一个阶段的值，并注明产生该值的代码表达式。
* ``load_hf_model``: 加载 tokenizer、12 个类别 token 和 FP32 模型。
* ``generate_edits``: 跑单 chunk 的 greedy 生成并解析类别边界。
* ``describe_routes``: 展示 edits 如何被 TNReader 定位和路由。
* ``trace_inference``: 串起完整推理流程并逐阶段输出。

函数如何协作
============
``trace_inference`` -> ``split_chunks/has_digit`` -> ``generate_edits``
-> ``parse_label_token_ids`` -> ``TNReader._locate/_normalizer_for``
-> ``TNReader.apply`` -> ``"".join(spoken_chunks)``。

关键参数
========
* ``text``: 待推理原文，必须作为一个 shell 参数传入。
* ``--model``: Hugging Face 模型目录，默认 ``model/a1``。
* ``--device``: ``auto``、``cpu``、``cuda`` 或 ``cuda:N``。
* ``--dtype``: 默认 ``float32``，用于复现 HF FP32 指标。
* ``--max-new``: 每个 chunk 最多生成的 token 数。
* ``--max-chunks``: 最多送 LLM 的含数字 chunk 数，默认与服务端一致为 8。

运行示例::

    python infer_trace.py "客服电话400-820-8820，会议3月5日下午2点开始"
"""

import argparse
import pprint
import sys
import time


DEFAULT_MODEL = "model/a1"
DEFAULT_TEXT = "客服电话400-820-8820，会议3月5日下午2点开始，胜率75%。"


def show(stage, value, code):
    """打印一个中间结果及其来源代码。

    参数:
        stage: 输出阶段名称，例如 ``04 chunks``。
        value: 任意可打印对象；字符串用 ``repr`` 保留空格和 special token。
        code: 真正产生该值的代码表达式，会随中间结果一起打印。
    """
    caller = sys._getframe(1)
    print("\n" + "=" * 100)
    print(stage)
    print(f"调用位置: {caller.f_code.co_filename}:{caller.f_lineno}")
    print(f"产生代码: {code}")
    if isinstance(value, str):
        print(f"值      : {value!r}")
    else:
        pprint.pprint(value, width=120, sort_dicts=False)


def describe_tokens(tokenizer, token_ids):
    """把 token ids 展开成便于检查的 id、词表 token 和解码文本。

    参数:
        tokenizer: 已注册 12 个 A1 类别 token 的 tokenizer。
        token_ids: 需要展开的 token id 列表。

    返回:
        ``[{"id": ..., "token": ..., "decoded": ...}, ...]``。
    """
    rows = []
    for token_id in token_ids:
        token_id = int(token_id)
        rows.append({
            "id": token_id,
            "token": tokenizer.convert_ids_to_tokens(token_id),
            "decoded": tokenizer.decode(
                [token_id], skip_special_tokens=False),
        })
    return rows


def load_hf_model(model_path, device, dtype_name):
    """加载 A1 tokenizer、类别 token 和因果语言模型。

    参数:
        model_path: 本地 Hugging Face 模型目录。
        device: ``auto``、``cpu``、``cuda`` 或具体 CUDA 设备。
        dtype_name: ``float32``、``float16`` 或 ``bfloat16``。

    返回:
        ``(tokenizer, model, torch_module)``。

    异常:
        SystemExit: 请求 CUDA 但环境不可用，或 CPU 与低精度 dtype 不兼容。
    """
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ModuleNotFoundError as exc:
        raise SystemExit(
            f"缺少推理依赖 {exc.name}; 请先安装 requirements.txt，"
            "或使用 README 中的 Docker 运行方式"
        ) from exc

    from train.label_tokens import add_label_tokens

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("请求使用 CUDA，但当前 PyTorch 看不到可用 GPU")
    dtype = getattr(torch, dtype_name)
    if device == "cpu" and dtype in (torch.float16, torch.bfloat16):
        raise SystemExit(f"CPU 推理不支持 dtype={dtype_name}，请使用 float32")

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, fix_mistral_regex=True)
    add_label_tokens(tokenizer)
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=dtype)
    if len(tokenizer) != model.get_input_embeddings().num_embeddings:
        model.resize_token_embeddings(len(tokenizer))
    model.to(device)
    model.eval()
    return tokenizer, model, torch


def generate_edits(text, tokenizer, model, torch, max_new):
    """执行一个 chunk 的 greedy 生成并恢复类别 edits。

    参数:
        text: 一个经过无损切片的文本片。
        tokenizer: 已注册 A1 类别 token 的 tokenizer。
        model: Hugging Face 因果语言模型。
        torch: 调用方已导入的 torch 模块，避免本模块导入时强制依赖 GPU。
        max_new: 最多生成的 token 数。

    返回:
        包含 prompt、输入 ids、生成 ids、原始 token 细节和已解析 edits 的字典。
    """
    from train.constants import EOS_ID, SEP, TN_PREFIX
    from train.label_tokens import parse_label_token_ids

    prompt = TN_PREFIX + text + SEP
    encoded = tokenizer(
        prompt,
        add_special_tokens=False,
        return_tensors="pt",
    )
    input_ids = encoded.input_ids.to(model.device)
    attention_mask = torch.ones_like(input_ids)
    with torch.inference_mode():
        sequence = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new,
            do_sample=False,
            eos_token_id=EOS_ID,
            pad_token_id=EOS_ID,
        )
    generated_ids = sequence[0, input_ids.shape[1]:].tolist()
    parsed = parse_label_token_ids(generated_ids, tokenizer)
    return {
        "prompt": prompt,
        "input_ids": input_ids[0].tolist(),
        "generated_ids": generated_ids,
        "generated_text": tokenizer.decode(
            generated_ids, skip_special_tokens=False),
        "generated_tokens": describe_tokens(tokenizer, generated_ids),
        "edits": parsed,
        "hit_max_new": len(generated_ids) >= max_new,
    }


def describe_routes(reader, chunk, edits):
    """展示 edits 如何被 TNReader 定位、分类和选择场景 normalizer。

    参数:
        reader: 已初始化的 TNReader。
        chunk: 当前文本片。
        edits: 模型在该片内生成的 ``[[span, CLASS], ...]``。

    返回:
        每个定位成功的 span 及其路由详情。``scene_output`` 是场景遍结果，
        最终 chunk 输出还会再经过一次默认 normalizer 兜底。
    """
    routes = []
    for start, end, label in reader._locate(chunk, edits):
        normalizer = reader._normalizer_for(label)
        span = chunk[start:end]
        routes.append({
            "span": span,
            "offset": [start, end],
            "class": label,
            "normalizer": type(normalizer).__name__ if normalizer else "default",
            "scene_output": reader._read_scene(span, normalizer),
        })
    return routes


def trace_inference(text, args):
    """执行单条文本的完整推理，并按阶段打印中间值。

    参数:
        text: 原始输入文本。
        args: argparse 结果，提供模型、设备、dtype 和生成长度配置。

    返回:
        ``(spoken, flatten_edits, elapsed_ms)``。
    """
    from train.split import has_digit, split_chunks
    from train.tn_fst import TNReader
    from train.latex import delatex

    started = time.perf_counter()

    # 输出 1：这一行直接打印命令行传入的原始文本。
    show("01 原始输入 text", text, "text = args.text")

    # 输出 2：这一行打印 delatex(text) 的返回值。
    written = delatex(text)
    show("02 LaTeX 改写 written", written, "written = delatex(text)")

    # 输出 3：这一行打印 split_chunks(written) 返回的无损切片。
    chunks = split_chunks(written)
    show("03 无损切片 chunks", chunks, "chunks = split_chunks(written)")

    # 输出 4：这一行打印每个 chunk 的 has_digit() 判断。
    digit_flags = [has_digit(chunk) for chunk in chunks]
    show(
        "04 每片是否送 LLM",
        [
            {"index": index, "has_digit": flag, "chunk": chunks[index]}
            for index, flag in enumerate(digit_flags)
        ],
        "[has_digit(chunk) for chunk in chunks]",
    )

    # 输出 5：这一行打印实际选取和因 max_chunks 丢弃的 chunk 下标。
    digit_indices = [
        index for index, flag in enumerate(digit_flags) if flag
    ]
    picked = digit_indices[:args.max_chunks]
    dropped = digit_indices[args.max_chunks:]
    show(
        "05 LLM chunk 配额",
        {"picked": picked, "dropped": dropped, "max_chunks": args.max_chunks},
        "picked = digit_indices[:args.max_chunks]",
    )

    print("\n" + "=" * 100)
    print(f"加载模型 model={args.model} device={args.device} dtype={args.dtype}")
    print("=" * 100)
    tokenizer, model, torch = load_hf_model(
        args.model, args.device, args.dtype)

    runs = {}
    edits_by_chunk = {}
    for index in picked:
        chunk = chunks[index]
        print(f"\n---------- chunk {index} / {len(chunks) - 1} ----------")

        # 输出 6：这一行打印真正送给模型的 prompt 字符串。
        prompt = _prompt_for(chunk)
        show(
            f"06 chunk[{index}] 模型 prompt",
            prompt,
            'prompt = TN_PREFIX + chunks[index] + SEP',
        )

        # 输出 7 到 10：generate_edits 内部保存的输入 ids、原始生成 ids、
        # token 细节和 parse_label_token_ids 解析结果，分别在这里打印。
        result = generate_edits(
            chunk, tokenizer, model, torch, args.max_new)
        runs[index] = result
        show(
            f"07 chunk[{index}] prompt token ids",
            result["input_ids"],
            "input_ids = tokenizer(prompt).input_ids",
        )
        show(
            f"08 chunk[{index}] 模型原始生成 ids",
            result["generated_ids"],
            "generated_ids = sequence[0, input_ids.shape[1]:].tolist()",
        )
        show(
            f"09 chunk[{index}] 原始生成 token 明细",
            {
                "text_with_special_tokens": result["generated_text"],
                "tokens": result["generated_tokens"],
            },
            "tokenizer.decode(generated_ids, skip_special_tokens=False); "
            "describe_tokens(tokenizer, generated_ids)",
        )
        show(
            f"10 chunk[{index}] 解析后的 edits",
            result["edits"],
            "parse_label_token_ids(generated_ids, tokenizer)",
        )
        if result["hit_max_new"]:
            print(f"警告: chunk[{index}] 生成达到 max_new={args.max_new}")
        edits_by_chunk[index] = result["edits"]

    # 输出 11：这一行打印所有进入模型的 chunk 和它们解析出的 edits。
    show(
        "11 按 chunk 汇总 edits",
        {
            index: {
                "chunk": chunks[index],
                "edits": edits_by_chunk[index],
            }
            for index in picked
        },
        "edits_by_chunk[index] = parse_label_token_ids(...)",
    )

    print("\n" + "=" * 100)
    print("初始化 TNReader，接下来展示 FST 类别路由和逐片读音")
    print("=" * 100)
    reader = TNReader()
    spoken_chunks = list(chunks)

    for index in picked:
        chunk = chunks[index]
        chunk_edits = edits_by_chunk[index]

        # 输出 12：这一行打印 TNReader._locate 定位出的不重叠 span。
        located = reader._locate(chunk, chunk_edits)
        show(
            f"12 chunk[{index}] span 定位",
            located,
            "located = reader._locate(chunk, chunk_edits)",
        )

        # 输出 13：这一行打印每个 span 的类别路由和场景 normalizer 结果。
        routes = describe_routes(reader, chunk, chunk_edits)
        show(
            f"13 chunk[{index}] FST 类别路由",
            routes,
            "reader._normalizer_for(label); reader._read_scene(span, normalizer)",
        )

        # 输出 14：这一行打印 TNReader.apply 对该 chunk 的完整输出。
        spoken_chunks[index] = reader.apply(chunk, chunk_edits)
        show(
            f"14 chunk[{index}] 最终读音",
            spoken_chunks[index],
            "spoken_chunks[index] = reader.apply(chunk, chunk_edits)",
        )

    # 输出 15：未进入 LLM/FST 的片，在服务端会原样透传。
    unchanged = {
        index: chunk
        for index, chunk in enumerate(chunks)
        if index not in edits_by_chunk
    }
    show(
        "15 未进入 LLM 的片，原样保留",
        unchanged,
        "spoken_chunks = list(chunks); 只更新 picked 下标",
    )

    # 输出 16：这一行把所有逐片读音拼成最终口语文本。
    spoken = "".join(spoken_chunks)
    show(
        "16 最终 spoken",
        spoken,
        'spoken = "".join(spoken_chunks)',
    )

    # 输出 17：这一行打印服务端响应中的扁平 edits 列表。
    flatten_edits = [
        edit
        for index in picked
        for edit in edits_by_chunk[index]
    ]
    show(
        "17 响应中的扁平 edits",
        flatten_edits,
        "[e for per in edits_per for e in per]",
    )

    elapsed_ms = (time.perf_counter() - started) * 1000.0
    show(
        "18 总耗时",
        f"{elapsed_ms:.1f} ms",
        "(time.perf_counter() - started) * 1000.0",
    )
    return spoken, flatten_edits, elapsed_ms


def _prompt_for(chunk):
    """单独构造 prompt，供输出阶段展示。

    参数:
        chunk: 已切片的文本片。

    返回:
        ``TN_PREFIX + chunk + SEP``。
    """
    from train.constants import SEP, TN_PREFIX

    return TN_PREFIX + chunk + SEP


def parse_args():
    """解析追踪脚本参数。"""
    parser = argparse.ArgumentParser(
        description="逐阶段打印 llm-tn A1 推理中间结果")
    parser.add_argument(
        "text", nargs="?", default=DEFAULT_TEXT,
        help="待推理文本，包含空格时请使用引号")
    parser.add_argument(
        "--model", default=DEFAULT_MODEL, help="HF 模型目录")
    parser.add_argument(
        "--device", default="auto",
        help="auto/cpu/cuda/cuda:N，默认自动使用 GPU")
    parser.add_argument(
        "--dtype", default="float32",
        choices=["float32", "float16", "bfloat16"],
        help="模型精度，默认与 HF FP32 评测一致")
    parser.add_argument(
        "--max-new", type=int, default=128,
        help="每个 chunk 最多生成的 token 数")
    parser.add_argument(
        "--max-chunks", type=int, default=8,
        help="最多送 LLM 的含数字 chunk 数")
    return parser.parse_args()


def main():
    """运行追踪脚本并输出最终摘要。"""
    args = parse_args()
    if not args.text.strip():
        raise SystemExit("text 不能为空")
    if args.max_chunks <= 0:
        raise SystemExit("--max-chunks 必须大于 0")

    spoken, edits, elapsed_ms = trace_inference(args.text, args)
    print("\n" + "#" * 100)
    print(f"IN    : {args.text}")
    print(f"EDITS : {edits}")
    print(f"OUT   : {spoken}")
    print(f"COST  : {elapsed_ms:.1f} ms")
    print("#" * 100)


if __name__ == "__main__":
    main()
