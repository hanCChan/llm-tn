# =============================================================================
# llm-tn source header
# Path: test/bench_server.py
# 职责: 压测 /normalize 的吞吐/延迟并核对端到端质量
# 核心原理: aiohttp 固定并发发请求，统一 CLASS 后计算 Tag F1 和 spoken 准确率。
# =============================================================================

"""对 TN 服务执行固定并发 HTTP 压测并核对端到端质量。

总体怎么做
==========
1. 读取带 ``src/edits/gold_spoken`` 的 373 条评测数据。
2. 用 aiohttp 连接池按 c1/c8/c32... 固定并发请求 ``/normalize``。
3. 记录每请求延迟、吞吐、Tag micro-F1 和 spoken 逐字准确率。
4. 对比各并发档位的 edits+spoken，检查 continuous batching 是否改变输出。

核心函数
========
* ``run_round_aiohttp``: 并发请求、延迟采样和响应收集。
* ``accuracy``: 统一 CLASS 后计算 span 集合 micro-F1 和文本正确数。
* ``main_async``: 建连接池、预热、逐档位执行并输出结果。

函数如何协作
============
``main_async`` -> ``run_round_aiohttp`` -> 服务 ``/normalize`` -> ``accuracy``
-> 检查跨并发输出漂移。httpx 仅保留作客户端对照。

关键参数
========
* ``--url``: 服务地址。
* ``--data``: 一个或多个评测 JSONL。
* ``--concurrency/-c``: 固定并发档位。
* ``--client``: ``aiohttp`` 默认；``httpx`` 用于对照。
* ``--timeout``: 单 HTTP 请求超时秒数。
"""
import argparse
import asyncio
import json
import os
import statistics
import sys
import time

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

DEFAULT_DATA = [
    "data/test_data/polynorm_eval.jsonl",
    "data/test_data/hard_num.jsonl",
]


def load(paths):
    """读取并拼接多个 JSONL。

    参数:
        paths: JSONL 路径列表，支持相对仓库根目录的路径。
    """
    recs = []
    for path in paths:
        recs += [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    if not recs:
        raise SystemExit("数据集为空: {}".format(", ".join(paths)))
    return recs


async def run_round(client, url, recs, conc):
    """用 httpx 固定并发跑完一批请求。

    参数:
        client: 共享 ``httpx.AsyncClient``。
        url: 服务基地址。
        recs: 包含 ``src`` 的评测记录。
        conc: 同时在途的最大请求数。

    返回:
        ``(墙钟秒数, 每请求延迟秒数, 每请求响应 JSON)``。
    """
    sem = asyncio.Semaphore(conc)
    latency = [0.0] * len(recs)
    results = [None] * len(recs)

    async def one(i, text):
        """发送一条 httpx 请求并记录索引对应的结果与延迟。"""
        async with sem:
            started = time.monotonic()
            r = await client.post(f"{url}/normalize", json={"text": text})
            latency[i] = time.monotonic() - started
            r.raise_for_status()
            results[i] = r.json()

    started = time.monotonic()
    await asyncio.gather(*[one(i, r["src"]) for i, r in enumerate(recs)])
    return time.monotonic() - started, latency, results


async def run_round_aiohttp(session, url, recs, conc):
    """用 aiohttp 固定并发跑完一批请求。

    参数:
        session: 共享 ClientSession。
        url: 服务基地址。
        recs: 包含 ``src`` 的评测记录。
        conc: 同时在途的最大请求数。

    返回:
        ``(墙钟秒数, 每请求延迟秒数, 每请求响应 JSON)``。
    """
    sem = asyncio.Semaphore(conc)
    latency = [0.0] * len(recs)
    results = [None] * len(recs)

    async def one(i, text):
        """发送一条 aiohttp 请求并记录索引对应的结果与延迟。"""
        async with sem:
            started = time.monotonic()
            async with session.post(f"{url}/normalize", json={"text": text}) as response:
                response.raise_for_status()
                results[i] = await response.json()
            latency[i] = time.monotonic() - started

    started = time.monotonic()
    await asyncio.gather(*(one(i, rec["src"]) for i, rec in enumerate(recs)))
    return time.monotonic() - started, latency, results


def accuracy(recs, results):
    """计算 Tag micro-F1 和 spoken 逐字正确数。

    参数:
        recs: gold 记录。
        results: 与 recs 顺序一致的 HTTP 响应。

    gold 的 ``CLASS:ctx`` 与预测 ``CLASS`` 只比较冒号前的语义类别。

    返回:
        ``(Tag micro-F1, precision, recall, spoken 正确数)``。
    """
    tp = fp = fn = 0
    txt_ok = 0
    for rec, res in zip(recs, results):
        # The A1 HTTP contract returns CLASS, while historical gold labels keep
        # the optional v2 context suffix (CLASS:ctx). Only CLASS is semantic.
        gold = {(span, label.split(":", 1)[0].upper())
                for span, label in rec["edits"]}
        pred = {(span, label.split(":", 1)[0].upper())
                for span, label in res["edits"]}
        tp += len(pred & gold)
        fp += len(pred - gold)
        fn += len(gold - pred)
        txt_ok += res["spoken"] == rec["gold_spoken"]
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return (2 * p * r / (p + r) if p + r else 0.0), p, r, txt_ok


def pct(values, q):
    """返回延迟列表的百分位。

    参数:
        values: 逐请求延迟秒数。
        q: 百分位整数，例如 p95 传 95。

    返回:
        按最近秩规则选取的延迟值；输入为空时抛出 ``IndexError``。
    """
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(len(ordered) * q / 100))]


async def main_async(args):
    """按客户端类型和并发档位执行压测、准确率评估与结果输出。

    原理: 每轮固定并发发送全部样本，用完成数除以墙钟时间得到吞吐；全部
    请求完成后统一计算 Tag F1 和 ``gold_spoken`` 逐字准确率。预热轮先触发
    CUDA graph 与 FST 缓存，避免一次性初始化污染首个档位。

    参数:
        args: argparse 结果，提供 url/data/concurrency/client/timeout。
    """
    recs = load(args.data)
    max_conc = max(args.concurrency)
    if args.client == "aiohttp":
        import aiohttp

        client = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=args.timeout),
            connector=aiohttp.TCPConnector(
                limit=max_conc, limit_per_host=max_conc
            ),
        )
        run_round_fn = run_round_aiohttp
    else:
        import httpx

        client = httpx.AsyncClient(
            timeout=args.timeout,
            limits=httpx.Limits(
                max_connections=max_conc, max_keepalive_connections=max_conc
            ),
        )
        run_round_fn = run_round

    async with client:
        # 预热: 让 CUDA graph / FST 缓存就位, 否则首个档位会被一次性开销污染。
        await run_round_fn(client, args.url, recs[:min(4, len(recs))], 1)

        print(f"目标: {args.url}  客户端: {args.client}  数据: {len(recs)} 句 "
              f"({', '.join(args.data)})\n")
        print(f"{'并发':>4} {'耗时(s)':>8} {'吞吐(句/s)':>11} {'均值':>8} "
              f"{'p50':>7} {'p95':>7} {'p99':>7}  {'TagF1':>6} {'文本正确':>10}")
        print("-" * 82)

        baseline = None
        seen = {}
        for conc in args.concurrency:
            elapsed, latency, results = await run_round_fn(
                client, args.url, recs, conc
            )
            qps = len(recs) / elapsed
            baseline = baseline or qps
            f1, _, _, txt_ok = accuracy(recs, results)
            speedup = f"  {qps / baseline:.1f}x" if conc != args.concurrency[0] else ""
            print(f"{conc:>4} {elapsed:>8.1f} {qps:>11.1f} "
                  f"{statistics.mean(latency) * 1000:>7.0f}ms {pct(latency, 50) * 1000:>6.0f}ms "
                  f"{pct(latency, 95) * 1000:>6.0f}ms {pct(latency, 99) * 1000:>6.0f}ms  "
                  f"{f1:>6.4f} {txt_ok:>4}/{len(recs)} = {txt_ok / len(recs):.3f}{speedup}")
            seen[conc] = [r["spoken"] for r in results]

        # greedy 解码下各并发档位输出应逐字一致; 不一致说明合批改变了结果。
        first = args.concurrency[0]
        drift = [c for c in args.concurrency[1:] if seen[c] != seen[first]]
        if drift:
            print(f"\n警告: 并发 {drift} 的输出与并发 {first} 不一致, 合批影响了结果, 上线前需排查。")


def main():
    """解析压测参数并启动 asyncio 主流程。

    关键参数:
        ``--client``: aiohttp 默认，避免 httpx 高并发客户端排队。
        ``--concurrency``: 逐档固定并发，不做开放到达压测。
        ``--data``: 必须含 gold edits 和 gold_spoken。
    """
    ap = argparse.ArgumentParser(description="压测 TN 服务的 /normalize")
    ap.add_argument("--url", default="http://localhost:8000", help="服务地址")
    ap.add_argument("--data", nargs="+", default=DEFAULT_DATA,
                    help="jsonl, 需含 src+edits+gold_spoken; 默认跑内置两个测试集")
    ap.add_argument("-c", "--concurrency", type=int, nargs="+", default=[1, 8, 32])
    ap.add_argument("--client", choices=["aiohttp", "httpx"], default="aiohttp")
    ap.add_argument("--timeout", type=float, default=180.0, help="单请求超时(s)")
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()
