"""按文本长度分档压测 TN HTTP 服务。

总体怎么做
==========
1. 从测试语料确定性构造短句、85 字、512 字和 1:1:1 混合四组负载。
2. 使用 aiohttp 固定并发发送全部请求，每档默认 96 条。
3. 预热 vLLM CUDA Graph 和 FST 缓存后，记录 req/s、mean/p50/p95/p99。
4. 输出每组实际字符长度、含数字 chunk 数和真正送入模型的片数。

核心函数
========
* ``make_workloads``: 确定性构造四类负载。
* ``run_round``: 固定并发发送一轮并采集逐请求延迟。
* ``main_async``: 建立连接池、预热、逐 bucket/concurrency 执行。

函数如何协作
============
``main_async`` 先调用 ``load_sources`` 和 ``make_workloads`` 构造文本，
再建立共享 ClientSession；预热和对每个 bucket/concurrency 的正式测量都调用
``run_round``；``pct`` 负责把每轮延迟列表汇总成 p50/p95/p99。

关键参数
========
* ``--url``: 服务地址。
* ``--count``: 每个 bucket 的请求数，默认 96。
* ``--concurrency/-c``: 固定并发档位。
* ``--warmup-rounds``: 每类负载预热轮数。
* ``--timeout``: 单请求超时秒数。
"""
import argparse
import asyncio
import json
import math
import os
import random
import statistics
import sys
import time

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from train.split import has_digit, split_chunks  # noqa: E402

DEFAULT_DATA = [
    "data/test_data/polynorm_eval.jsonl",
    "data/test_data/hard_num.jsonl",
]
MAX_CHUNKS = 8


def load_sources(paths):
    """读取一个或多个 JSONL 的 ``src`` 数据。

    参数:
        paths: 绝对路径或相对仓库根目录的路径。
    """
    rows = []
    for path in paths:
        full = path if os.path.isabs(path) else os.path.join(_ROOT, path)
        with open(full, encoding="utf-8") as source:
            rows.extend(json.loads(line) for line in source if line.strip())
    if not rows:
        raise SystemExit("evaluation data is empty")
    return rows


def make_workloads(records, count, seed=17):
    """确定性地构造四类文本负载。

    参数:
        records: 至少包含 ``src`` 的评测记录。
        count: 每个 bucket 的请求数。
        seed: 采样种子，保证不同运行之间文本一致。
    """
    rng = random.Random(seed)
    short_pool = [row["src"] for row in records if len(row["src"]) <= 15]

    def build(target_len):
        text = ""
        while len(text) < target_len:
            text += rng.choice(records)["src"]
        return text[:target_len]

    short = [short_pool[i % len(short_pool)] for i in range(count)]
    medium = [build(85) for _ in range(count)]
    long = [build(512) for _ in range(count)]
    mixed = [
        short[i] if i % 3 == 0 else medium[i] if i % 3 == 1 else long[i]
        for i in range(count)
    ]
    return {"short": short, "85char": medium, "long512": long, "mixed": mixed}


def pct(values, q):
    """返回最近秩百分位。

    参数:
        values: 本轮的逐请求延迟毫秒数。
        q: 分位点，使用 0~1 小数，例如 p95 传 0.95。

    返回:
        排序后按最近秩规则选出的延迟值。
    """
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, math.ceil(len(ordered) * q) - 1)]


async def run_round(session, url, bucket, texts, concurrency):
    """固定并发发送一组文本并采集延迟。

    参数:
        session: 共享 ClientSession。
        url: 服务地址。
        bucket: requestid 中使用的负载分组名。
        texts: 本轮待发送文本。
        concurrency: 同时在途的最大请求数。

    返回:
        ``(墙钟秒数, 每请求毫秒延迟)``。
    """
    sem = asyncio.Semaphore(concurrency)
    latency = [0.0] * len(texts)

    async def one(index, text):
        async with sem:
            started = time.perf_counter()
            async with session.post(
                f"{url}/normalize",
                json={"text": text, "requestid": f"len-{bucket}-{index}"},
            ) as response:
                response.raise_for_status()
                await response.read()
            latency[index] = (time.perf_counter() - started) * 1000.0

    started = time.perf_counter()
    await asyncio.gather(*(one(i, text) for i, text in enumerate(texts)))
    return time.perf_counter() - started, latency


async def main_async(args):
    """执行预热和全部长度/并发组合。

    参数:
        args: argparse 结果，包含 url/data/count/concurrency/timeout。
    """
    import aiohttp

    workloads = make_workloads(load_sources(args.data), args.count)
    connector = aiohttp.TCPConnector(
        limit=max(args.concurrency), limit_per_host=max(args.concurrency)
    )
    timeout = aiohttp.ClientTimeout(total=args.timeout)

    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        for _ in range(args.warmup_rounds):
            for bucket, texts in workloads.items():
                await run_round(session, args.url, f"warm-{bucket}", texts, 8)

        print(
            f"{'bucket':8s} {'c':>2s} {'req/s':>8s} {'mean':>7s} "
            f"{'p50':>7s} {'p95':>7s} {'p99':>7s}"
        )
        for bucket, texts in workloads.items():
            chunks = [
                sum(has_digit(chunk) for chunk in split_chunks(text))
                for text in texts
            ]
            print(
                f"{bucket:8s} shape chars={min(map(len, texts))}-{max(map(len, texts))} "
                f"digit_chunks={min(chunks)}-{max(chunks)} "
                f"sent={min(min(c, MAX_CHUNKS) for c in chunks)}-"
                f"{min(max(chunks), MAX_CHUNKS)}"
            )
            for concurrency in args.concurrency:
                elapsed, latency = await run_round(
                    session, args.url, bucket, texts, concurrency
                )
                print(
                    f"{bucket:8s} {concurrency:2d} {len(texts) / elapsed:8.1f} "
                    f"{statistics.mean(latency):7.1f} {pct(latency, 0.50):7.1f} "
                    f"{pct(latency, 0.95):7.1f} {pct(latency, 0.99):7.1f}"
                )


def main():
    """解析长度压测参数并启动 asyncio 主流程。"""
    parser = argparse.ArgumentParser(description="Length-bucket TN HTTP load")
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--data", nargs="+", default=DEFAULT_DATA)
    parser.add_argument("-c", "--concurrency", type=int, nargs="+",
                        default=[1, 8, 32])
    parser.add_argument("--count", type=int, default=96)
    parser.add_argument("--warmup-rounds", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=180.0)
    asyncio.run(main_async(parser.parse_args()))


if __name__ == "__main__":
    main()
