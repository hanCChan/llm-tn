"""阶段压测: 不起服务, 直接测推理引擎本身 —— 质量 / 单条延迟 / 批量吞吐 / 并发延迟, 一张表出结果。

为什么不起 server.py 压: server.py 走的是 v2 AR + vLLM 管线, NAR 权重还没接进服务端
(合批/CUDA graph 属于后续 GPU 阶段)。当前阶段(数据工程后)的"各方面效果" = 推理引擎
本身的四件事, 用同一把尺子量 v2 基线和新权重:

    1) 质量      373 测试集(口径可选) TagF1 / 句级 / 文本逐字 / 负例
    2) 单条延迟  30 条冻结文本, batch=1, warmup 后逐条计时 p50/p95/p99 (质量口径的延迟)
    3) 批量吞吐  30 句按 batch=16 分批前向, 句/s —— 一次前向出全部标签的吞吐上限
    4) 并发延迟  asyncio 客户端(同 bench_server.py 的做法, 不用线程池: 同步 IO 争 GIL
                 会把客户端自己变成瓶颈)同时打 tag 请求, 固定在途并发 c=1/8/32/64,
                 测各档位吞吐与 p50/p95/p99 —— NAR 引擎是单实例串行推理, 并发档位
                 衡量"排队下的表现", 与将来服务端合批后的数字不直接可比。

用法:
    python test/bench_stage.py --nar runs/nar_best            # 新权重
    python test/bench_stage.py --nar runs/nar_best --data data/test_data/gold_v2_serial/...
    python test/bench_stage.py --ar                            # v2 基线对照(无并发/吞吐段)
    python test/bench_stage.py --nar runs/nar_best --concurrency 1 8 32 64   # 只调档位
"""
import argparse
import asyncio
import json
import os
import sys
import time

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from train.tn_fst import TNReader                     # noqa: E402

DEFAULT_DATA = ["data/test_data/gold_v2_serial/test_polynorm195_goldv2.jsonl",
                "data/test_data/gold_v2_serial/test_hardnum178_goldv2.jsonl"]
# 延迟集: 30 条冻结文本(hard_num 前 20 + polynorm 前 10), 与消融仓库同一份, 跨阶段可比。
LATENCY_SET = os.path.join(_ROOT, "data", "test_data", "bench_latency_set30.jsonl")


def make_latency_set():
    """生成一次即冻结: 数字密集 20 + 普通含数字 10。"""
    if os.path.exists(LATENCY_SET):
        return
    rows = []
    for path, k in (("data/test_data/hard_num.jsonl", 20), ("data/test_data/polynorm_eval.jsonl", 30)):
        for line in open(os.path.join(_ROOT, path), encoding="utf-8"):
            if len(rows) >= k:
                break
            rows.append({"src": json.loads(line)["src"]})
    with open(LATENCY_SET, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def quality(tag_one, reader, paths):
    tp = fp = fn = tag_exact = neg_total = neg_ok = txt_ok = n = 0
    for path in paths:
        for line in open(os.path.join(_ROOT, path) if not path.startswith("/") else path, encoding="utf-8"):
            if not line.strip():
                continue
            r = json.loads(line)
            n += 1
            edits = tag_one(r["src"])
            gold = {tuple(e) for e in r["edits"]}
            pred = {tuple(e) for e in edits}
            tag_exact += pred == gold
            if not gold:
                neg_total += 1
                neg_ok += not pred
            tp += len(pred & gold)
            fp += len(pred - gold)
            fn += len(gold - pred)
            txt_ok += reader.apply(r["src"], edits) == r["gold_spoken"]
    f1 = prf(tp, fp, fn)[2]
    return dict(n=n, f1=f1, tag_exact=tag_exact, txt_ok=txt_ok, neg_ok=neg_ok, neg_total=neg_total)


def latency_pct(ms, q):
    s = sorted(ms)
    return s[min(len(s) - 1, int(len(s) * q / 100))]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--nar", help="NAR 权重目录(如 runs/nar_best)")
    g.add_argument("--ar", action="store_true", help="v2 AR 基线(main.TN)")
    ap.add_argument("--data", nargs="+", default=DEFAULT_DATA)
    ap.add_argument("--batch", type=int, default=16, help="吞吐测量的批大小")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--concurrency", type=int, nargs="+", default=[1, 8, 32, 64],
                    help="并发延迟档位(在途请求数)")
    args = ap.parse_args()
    make_latency_set()

    reader = TNReader()
    if args.ar:
        from main import TN
        tn = TN(model=os.path.join(_ROOT, "model", "v2"))

        def tag_one(src):
            return tn.tag(src)

        name = "v2 AR 基线(main.TN, fp32 HF generate)"
    else:
        from train.nar import NARRunner
        nar = NARRunner(args.nar)

        def tag_one(src):
            return nar.tag([src])[0]

        name = f"NAR({os.path.basename(args.nar)})"

    print(f"引擎: {name}")
    print(f"数据: {args.data}\n")

    # --- 1) 质量 ---
    q = quality(tag_one, reader, args.data)
    print(f"[质量]  {q['n']} 句: TagF1={q['f1']:.3f}  句级 {q['tag_exact']}/{q['n']}"
          f"  文本 {q['txt_ok']}/{q['n']}  负例 {q['neg_ok']}/{q['neg_total']}")

    # --- 2) 单条延迟(质量口径: batch=1) ---
    texts = [json.loads(l)["src"] for l in open(LATENCY_SET, encoding="utf-8")]
    for t in texts[:20]:
        tag_one(t)                                    # warmup
    samples = []
    for _ in range(args.repeats):
        for t in texts:
            t0 = time.perf_counter()
            tag_one(t)
            samples.append((time.perf_counter() - t0) * 1000)
    print(f"[延迟]  30 条 × {args.repeats} 轮, batch=1: p50={latency_pct(samples, 50):.1f}ms "
          f"p95={latency_pct(samples, 95):.1f}ms p99={latency_pct(samples, 99):.1f}ms "
          f"mean={sum(samples) / len(samples):.1f}ms")

    # --- 3) 批量吞吐(仅 NAR: 一次前向出全部标签; AR 逐 token 生成没有可比批路径) ---
    if args.nar:
        b = args.batch
        for _ in range(2):                            # warmup
            nar.tag(texts[:b])
        t0 = time.perf_counter()
        for i in range(0, len(texts), b):
            nar.tag(texts[i:i + b])
        dt = time.perf_counter() - t0
        print(f"[吞吐]  batch={b} 一次前向: {len(texts) / dt:.1f} 句/s (含读音不计; 引擎层上限)")

        # --- 4) 并发延迟: 固定在途并发打 tag+读音, 客户端 asyncio(同 bench_server.py, 不用线程池) ---
        # NARRunner 是单实例串行引擎(类注释: 同一 runner 只能在一个线程里用), 并发请求
        # 在引擎入口排队 —— 这测的是"串行引擎 + 排队"的真实行为, 不是将来服务端合批后
        # 的数字(那是 GPU 阶段的账)。锁必须加, 不加就是多线程裸调非线程安全引擎, 数字无意义。
        print(f"\n[并发]  30 条 × 3 轮/档, 串行引擎+排队(在途 {args.concurrency}):")
        print(f"{'并发':>4} {'吞吐(句/s)':>11} {'p50':>9} {'p95':>9} {'p99':>9} {'max':>9}")
        import concurrent.futures
        import threading
        lock = threading.Lock()

        def tagged(src):
            with lock:
                return tag_one(src)

        loop = asyncio.new_event_loop()
        pool = concurrent.futures.ThreadPoolExecutor(max_workers=max(args.concurrency))

        async def round_(conc):
            jobs = texts * args.repeats
            lat = [0.0] * len(jobs)
            sem = asyncio.Semaphore(conc)
            done = [0]

            async def worker(j, text):
                async with sem:
                    t0 = time.monotonic()
                    await loop.run_in_executor(pool, tagged, text)
                    lat[j] = (time.monotonic() - t0) * 1000
                    done[0] += 1

            t0 = time.monotonic()
            await asyncio.gather(*[worker(j, t) for j, t in enumerate(jobs)])
            wall = time.monotonic() - t0
            return lat, len(jobs) / wall

        for conc in args.concurrency:
            lat, qps = loop.run_until_complete(round_(conc))
            print(f"{conc:>4} {qps:>11.1f} {latency_pct(lat, 50):>8.1f}ms "
                  f"{latency_pct(lat, 95):>8.1f}ms {latency_pct(lat, 99):>8.1f}ms {max(lat):>8.1f}ms")
        pool.shutdown()
        print("\n对照: 历史阶段数字写进 data_synth/NOTES.md / 汇报")


if __name__ == "__main__":
    main()
