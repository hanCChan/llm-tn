"""验证 FST 进程池结果一致性及事件循环非阻塞特性。

总体怎么做
==========
1. 从测试语料构造 chunk 和 chunk 内 edits。
2. 先用进程内 TNReader 计算基准读音。
3. 启动 4 个 worker 的 FSTPipePool，并发分发同一批请求。
4. 比较两者结果完全一致。
5. 同时运行 5ms heartbeat，确认 FST 压测期间事件循环仍能按时调度。

核心函数
========
* ``load_requests``: 把句子转成 chunks+edits_map。
* ``main``: 计算 benchmark、启动 pool、并发请求、比较结果和 lag。

函数如何协作
============
``load_requests`` 构造池输入；``TNReader`` 生成进程内基准；``main`` 启动
``FSTPipePool`` 并发调用 ``apply``；heartbeat 与请求同时运行以观测事件循环
调度延迟；最后逐请求比较池化结果与基准。

关键参数
========
* ``limit``: 一致性测试最多使用的请求数。
* ``workers=4``: 与线上默认 FST 进程数一致。
* ``delay=5ms``: heartbeat 周期；最大 lag 阈值设为 100ms。
"""
import asyncio
import glob
import json
import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from train.fst_pool import FSTPipePool  # noqa: E402
from train.split import has_digit, split_chunks  # noqa: E402
from train.tn_fst import TNReader  # noqa: E402


def load_requests(limit=200):
    """从测试集构造池化请求。

    参数:
        limit: 最多返回的请求数。

    返回:
        ``[(chunks, edits_map), ...]``，edits_map 只包含需要 FST 的 chunk。
    """
    requests = []
    for path in sorted(glob.glob(os.path.join(_ROOT, "data/test_data/*.jsonl"))):
        with open(path, encoding="utf-8") as source:
            for line in source:
                rec = json.loads(line)
                chunks = split_chunks(rec["src"])
                picked = [i for i, chunk in enumerate(chunks) if has_digit(chunk)][:8]
                edits = {i: [] for i in picked}
                cursor = 0
                for span, label in rec["edits"]:
                    start = rec["src"].find(span, cursor)
                    if start < 0:
                        continue
                    cursor = start + len(span)
                    offset = 0
                    for index, chunk in enumerate(chunks):
                        if offset <= start and start + len(span) <= offset + len(chunk):
                            if index in edits:
                                edits[index].append([span, label])
                            break
                        offset += len(chunk)
                requests.append((chunks, edits))
                if len(requests) >= limit:
                    return requests
    return requests


async def main():
    """运行进程内基准、并发池化结果比较和事件循环 lag 检查。"""
    requests = load_requests()
    direct_reader = TNReader()
    expected = [
        [
            direct_reader.apply(chunk, edits[i]) if i in edits else chunk
            for i, chunk in enumerate(chunks)
        ]
        for chunks, edits in requests
    ]

    pool = FSTPipePool(workers=4, sched="lifo", cpus=None)
    await pool.start()
    try:
        loop = asyncio.get_running_loop()
        stop_heartbeat = asyncio.Event()
        lag = {"max_ms": 0.0}

        async def heartbeat():
            """每 5ms 自唤醒一次，记录事件循环最大调度延迟。"""
            while not stop_heartbeat.is_set():
                started = loop.time()
                await asyncio.sleep(0.005)
                lag["max_ms"] = max(
                    lag["max_ms"], (loop.time() - started - 0.005) * 1000.0
                )

        async def one(chunks, edits):
            """发送一个 FST 池请求并只返回 spoken chunk 列表。"""
            return (await pool.apply(chunks, edits))[0]

        heartbeat_task = asyncio.create_task(heartbeat())
        try:
            actual = await asyncio.gather(*(
                one(chunks, edits) for chunks, edits in requests
            ))
        finally:
            stop_heartbeat.set()
            await heartbeat_task
        assert actual == expected
        assert lag["max_ms"] < 100.0, lag
        stats = pool.stats()
        assert stats["alive"] == 4
        assert all(worker["n"] > 0 for worker in stats["workers"])
        print(
            f"fst pool passed requests={len(requests)} workers={stats['alive']} "
            f"queue_peak={stats['queue_peak']} loop_lag_max={lag['max_ms']:.1f}ms"
        )
    finally:
        pool.close()


if __name__ == "__main__":
    asyncio.run(main())
