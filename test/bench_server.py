# =============================================================================
# llm-tn source header
# Path: test/bench_server.py
# 职责: 压测 server.py 的 /normalize: 各并发下的吞吐/延迟, 并输出端到端准确率
# 核心原理: 遵循仓库现有的 A1 类别 token 与 pynini FST 管线，注释仅解释实现，不改变运行行为。
# =============================================================================

"""压测 server.py 的 /normalize: 各并发下的吞吐/延迟, 并输出端到端准确率。

    python test/bench_server.py                     # 两个默认数据集, 并发 1/8/32
    python test/bench_server.py -c 1 16 64
    python test/bench_server.py --url http://10.0.0.5:8000 --data data/test_data/hard_num.jsonl

必须用 asyncio 客户端压, 不能用 urllib+ThreadPoolExecutor: 后者是同步阻塞
IO, 几十个线程争 GIL, 客户端自己先成瓶颈, 会把服务端的 continuous batching 效果
完全掩盖掉(实测同一服务, 线程版并发 32 只有 3 句/s, asyncio 版 46.7 句/s)。
默认使用 aiohttp；httpx 在高并发下会增加明显的客户端排队，只保留作对照。

准确率与 test/eval_e2e.py 同义(Tag micro-F1 + 口语文本逐字匹配), 差别是这里走 HTTP
打真实服务。每个并发档位都算一次: greedy 解码下各档位应当完全一致, 若不一致说明合批
影响了输出, 需要在上线前查清。
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
    """读取并拼接多个 JSONL 压测数据，忽略空行。"""
    recs = []
    for path in paths:
        recs += [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    if not recs:
        raise SystemExit("数据集为空: {}".format(", ".join(paths)))
    return recs


async def run_round(client, url, recs, conc):
    """固定并发跑完 recs, 返回 (总耗时, 每请求延迟, 每条响应)。"""
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
    """固定并发跑完 recs，aiohttp 客户端版本。"""
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
    """Tag micro-F1 + 口语文本逐字正确数。口径与 test/eval_e2e.py 一致。"""
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
    """返回延迟列表的 q 分位数；空列表返回 0。"""
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(len(ordered) * q / 100))]


async def main_async(args):
    """按客户端类型和并发档位执行压测、准确率评估与结果输出。

    原理: 每轮固定并发发送全部样本，用完成数除以墙钟时间得到吞吐；全部
    请求完成后统一计算 Tag F1 和 ``gold_spoken`` 逐字准确率。预热轮先触发
    CUDA graph 与 FST 缓存，避免一次性初始化污染首个档位。
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
    """解析压测命令参数并启动 asyncio 主流程。"""
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
