"""面向线上服务的 FST 有界多进程池。

wetext/pynini 的 normalizer 不适合多线程并发：全局锁会把 FST 排队到事件循环
之外，高并发时形成明显尾延迟。本模块让每个 spawn worker 持有独立的 ``TNReader``，
通过全双工 pipe 收发任务，并由 asyncio 事件循环直接监听可读 fd，不额外占用线程。

核心约束：

* 队列有界，超过 ``max_queue`` 时快速失败，避免请求无限堆积；
* 单个请求有总超时，超时后取消尚未完成的 future；
* worker 异常只返回任务错误，不让整个进程池静默退出；
* ``stats()`` 暴露队列、worker 和延迟拆解，供 ``/debug/fst`` 和压测使用。
"""

import asyncio
import collections
import contextlib
import multiprocessing as mp
import os
import pickle
import signal
import sys
import time

from train.tn_fst import TNReader

# FST 首次启动可能需要编译规则；生产镜像已带缓存，仍给冷启动留足时间。
START_TIMEOUT = 300.0
_perf = time.perf_counter


class FSTOverloaded(RuntimeError):
    """队列已满，调用方应快速返回 503 而不是继续等待。"""

    pass


class FSTTimeout(RuntimeError):
    """单次请求等待 FST 总时长超过配置上限。"""

    pass


@contextlib.contextmanager
def _no_main_reimport():
    """临时移除 ``__main__`` 路径，避免 spawn worker 重新导入 server/CUDA。

    ``server.py`` 既可作为脚本运行，也会被 spawn 子进程用于寻找入口模块。
    如果子进程沿 ``__file__`` 再导入一次 server.py，就会重复初始化 vLLM/CUDA。
    这里只在创建 worker 的短窗口内隐藏入口路径，结束后恢复现场。
    """
    main = sys.modules.get("__main__")
    if main is None:
        yield
        return
    saved = {k: main.__dict__[k] for k in ("__spec__", "__file__") if k in main.__dict__}
    main.__dict__.pop("__file__", None)
    main.__spec__ = None
    try:
        yield
    finally:
        main.__dict__.update(saved)


def _ignore_sigint():
    """由主进程统一处理 Ctrl-C，worker 不单独中断正在执行的 FST 任务。"""
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def _read_sys(path):
    """读取 Linux topology 文本；受限容器没有该文件时返回 None。"""
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return None


def _core_key(cpu):
    """返回 CPU 对应的物理核心标识，用于避免两个 worker 抢占同一核心。"""
    base = f"/sys/devices/system/cpu/cpu{cpu}/topology/"
    siblings = _read_sys(base + "thread_siblings_list")
    if siblings:
        return siblings
    package = _read_sys(base + "physical_package_id")
    core = _read_sys(base + "core_id")
    return f"{package}:{core}" if core is not None else str(cpu)


def _l3_key(cpu):
    """返回 CPU 所属 L3 缓存域，用于尽量把 worker 分散到不同缓存域。"""
    return _read_sys(f"/sys/devices/system/cpu/cpu{cpu}/cache/index3/shared_cpu_list") or "all"


def _parse_cpu_list(spec):
    """解析 ``0,2-4`` 形式的 CPU 绑定列表。"""
    out = []
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            out.extend(range(int(start), int(end) + 1))
        else:
            out.append(int(part))
    return out


def resolve_cpus(spec, workers):
    """计算每个 worker 的 CPU 亲和性。

    ``auto`` 优先从不同 L3 域中各取一个物理核心，再按当前进程的 cgroup
    affinity 补足；明确的 CPU 列表只保留进程允许使用的 CPU。配置为空时返回
    全 None，表示不绑定 CPU。
    """
    if spec in (None, "", "none", "None", False):
        return [None] * workers
    allowed = sorted(os.sched_getaffinity(0))
    if spec == "auto":
        domains = collections.defaultdict(lambda: collections.defaultdict(list))
        for cpu in allowed:
            domains[_l3_key(cpu)][_core_key(cpu)].append(cpu)
        reps = [sorted((max(cpus) for cpus in cores.values()), reverse=True)
                for cores in domains.values()]
        reps.sort(key=lambda group: -group[0])
        picked = []
        while len(picked) < workers and any(reps):
            for group in reps:
                if group and len(picked) < workers:
                    picked.append(group.pop(0))
        if len(picked) < workers:
            rest = sorted(set(allowed) - set(picked), reverse=True)
            picked.extend(rest[:workers - len(picked)])
    else:
        allowed_set = set(allowed)
        picked = [cpu for cpu in _parse_cpu_list(spec) if cpu in allowed_set]
        if not picked:
            print(f"[fst_pool] cpus={spec!r} has no allowed CPU; affinity disabled",
                  file=sys.stderr)
            return [None] * workers
    if len(picked) < workers:
        print(f"[fst_pool] allowed CPUs={len(picked)} < workers={workers}; sharing cores",
              file=sys.stderr)
    return [picked[i % len(picked)] for i in range(workers)]


def _worker_main(conn, cpu):
    """FST worker 主循环：初始化独立 TNReader，串行处理 pipe 上的任务。

    每个 worker 只处理一个任务，避免 pynini 在同一进程内并发；主进程通过
    ``job_id`` 将响应与 future 对应。业务异常会被序列化回主进程，worker 本身
    继续服务后续请求。
    """
    _ignore_sigint()
    if cpu is not None:
        with contextlib.suppress(OSError):
            os.sched_setaffinity(0, {cpu})
    reader = TNReader()
    conn.send_bytes(pickle.dumps(("ready", os.getpid(), sorted(os.sched_getaffinity(0)))))
    while True:
        try:
            payload = conn.recv_bytes()
        except (EOFError, OSError):
            return
        t_recv = _perf()
        message = pickle.loads(payload)
        if message is None:
            return
        job_id, chunk, edits = message
        try:
            output = reader.apply(chunk, edits)
            error = None
        except Exception as exc:  # noqa: BLE001 - report the work error without killing worker
            output = None
            error = f"{type(exc).__name__}: {exc}"
        conn.send_bytes(pickle.dumps((job_id, output, error, t_recv, _perf())))


class _Worker:
    """主进程侧的 worker 句柄，集中保存进程、fd、忙碌任务和健康状态。"""

    __slots__ = ("index", "process", "conn", "fd", "cpu", "pid", "ready",
                 "job", "count", "alive", "exitcode")

    def __init__(self, index, process, conn, cpu, ready):
        self.index = index
        self.process = process
        self.conn = conn
        self.fd = conn.fileno()
        self.cpu = cpu
        self.pid = process.pid
        self.ready = ready
        self.job = None
        self.count = 0
        self.alive = True
        self.exitcode = None


class _Job:
    """一次 FST chunk 任务及其时间戳、future 和当前 worker。"""

    __slots__ = ("job_id", "chunk", "edits", "future", "enqueued_at",
                 "dispatched_at", "worker")

    def __init__(self, job_id, chunk, edits, future, enqueued_at):
        self.job_id = job_id
        self.chunk = chunk
        self.edits = edits
        self.future = future
        self.enqueued_at = enqueued_at
        self.dispatched_at = None
        self.worker = None


class FSTPipePool:
    """有界 FST 进程池，供单个 asyncio 事件循环使用。"""

    def __init__(self, workers=8, sched="lifo", cpus=None, max_queue=4096,
                 queue_timeout=30.0):
        if sched not in ("lifo", "fifo"):
            raise ValueError(f"unknown FST scheduler: {sched}")
        self.workers = max(1, int(workers))
        self.sched = sched
        self.cpus = cpus
        self.max_queue = max(1, int(max_queue))
        self.queue_timeout = float(queue_timeout)
        self.loop = None
        self.processes = []
        self.idle = collections.deque()
        self.queue = collections.deque()
        self.queue_peak = 0
        self._job_id = 0
        self._closed = False
        self.requests = 0
        self.jobs = 0

    async def start(self):
        """spawn 全部 worker，并等待每个 worker 完成 TNReader 初始化。"""
        self.loop = asyncio.get_running_loop()
        context = mp.get_context("spawn")
        cpus = resolve_cpus(self.cpus, self.workers)
        with _no_main_reimport():
            for index in range(self.workers):
                parent, child = context.Pipe(duplex=True)
                process = context.Process(
                    target=_worker_main,
                    args=(child, cpus[index]),
                    daemon=True,
                    name=f"fst-pipe-{index}",
                )
                process.start()
                child.close()
                worker = _Worker(
                    index, process, parent, cpus[index], self.loop.create_future())
                self.processes.append(worker)
                self.loop.add_reader(worker.fd, self._on_readable, worker)
        try:
            await asyncio.wait_for(
                asyncio.gather(*(worker.ready for worker in self.processes)),
                START_TIMEOUT,
            )
        except BaseException:
            self.close()
            raise
        self.idle.extend(self.processes)

    def close(self):
        """优雅通知 worker 退出；超时则 terminate，最后失败所有未完成任务。"""
        if self._closed:
            return
        self._closed = True
        for worker in self.processes:
            if worker.alive:
                with contextlib.suppress(Exception):
                    self.loop.remove_reader(worker.fd)
                with contextlib.suppress(Exception):
                    worker.conn.send_bytes(pickle.dumps(None))
        for worker in self.processes:
            worker.process.join(timeout=2)
            if worker.process.is_alive():
                worker.process.terminate()
                worker.process.join(timeout=1)
            with contextlib.suppress(Exception):
                worker.conn.close()
            worker.alive = False
        self._fail_all(RuntimeError("FST pool closed"))

    async def apply(self, chunks, edits_map):
        """处理一个请求中需要读音的 chunk。

        ``chunks`` 是完整分片列表，``edits_map`` 只包含需要过 FST 的下标。
        返回值保持原顺序：命中的下标替换为读音结果，其余 chunk 原样保留。
        """
        if self._closed or not any(worker.alive for worker in self.processes):
            raise RuntimeError("FST pool unavailable")
        # todo 在入队前一次性构造，确保超时取消时不会留下无法追踪的任务。
        todo = [(index, chunks[index], edits_map[index]) for index in edits_map]
        if len(self.queue) + len(todo) > self.max_queue:
            raise FSTOverloaded(
                f"FST queue full: queued={len(self.queue)} incoming={len(todo)} "
                f"max={self.max_queue}")

        self.requests += 1
        self.jobs += len(todo)
        started = _perf()
        spoken = list(chunks)
        futures = []
        try:
            # 有空闲 worker 时立即发送；否则进入 FIFO/LIFO 空闲调度队列。
            for index, chunk, edits in todo:
                futures.append((index, self._submit(chunk, edits, started)))
            results = await asyncio.wait_for(
                asyncio.gather(*(future for _, future in futures)),
                timeout=self.queue_timeout,
            )
        except asyncio.TimeoutError as exc:
            # future 取消后，迟到的 worker 响应仍会被读取，但不会重复赋值。
            for _, future in futures:
                future.cancel()
            raise FSTTimeout(
                f"FST request exceeded queue_timeout={self.queue_timeout}s") from exc
        except BaseException:
            for _, future in futures:
                future.cancel()
            raise

        for (index, _), (output, error, _stats) in zip(futures, results):
            if error is not None:
                raise RuntimeError(f"FST failed chunk={chunks[index]!r}: {error}")
            spoken[index] = output
        stats = [item[2] for item in results if item[2] is not None]
        # 一个请求可能有多个 chunk，取最慢的任务作为该请求的拆解耗时。
        slow = max(stats, key=lambda item: item["_wall"]) if stats else {}
        timing = {"fst_ms": (_perf() - started) * 1000.0}
        for key in ("queue_ms", "ipc_ms", "worker_ms"):
            if key in slow:
                timing[key] = slow[key]
        return spoken, timing

    def _submit(self, chunk, edits, enqueued_at):
        """创建任务 future，并立即调度到空闲 worker 或放入待处理队列。"""
        self._job_id += 1
        job = _Job(
            self._job_id,
            chunk,
            edits,
            self.loop.create_future(),
            enqueued_at,
        )
        worker = self._take_idle()
        if worker is None:
            self.queue.append(job)
            self.queue_peak = max(self.queue_peak, len(self.queue))
            return job.future
        self._send(worker, job)
        return job.future

    def _take_idle(self):
        """按配置从空闲 worker 两端取一个；LIFO 优先复用刚完成的 worker。"""
        if not self.idle:
            return None
        return self.idle.pop() if self.sched == "lifo" else self.idle.popleft()

    def _send(self, worker, job):
        """序列化任务并写入 duplex pipe；写失败时标记 worker 死亡。"""
        job.dispatched_at = _perf()
        job.worker = worker
        worker.job = job
        try:
            worker.conn.send_bytes(pickle.dumps((job.job_id, job.chunk, job.edits)))
        except OSError:
            self._worker_died(worker)

    def _release(self, worker):
        """worker 完成后优先取等待队列任务，队列为空才标记为空闲。"""
        while self.queue:
            job = self.queue.popleft()
            if job.future.done():
                continue
            self._send(worker, job)
            return
        self.idle.append(worker)

    def _on_readable(self, worker):
        """pipe 可读回调：处理 ready 消息或任务结果，然后释放 worker。"""
        try:
            payload = worker.conn.recv_bytes()
        except (EOFError, OSError):
            self._worker_died(worker)
            return
        received = _perf()
        try:
            message = pickle.loads(payload)
        except Exception as exc:  # noqa: BLE001
            self._worker_died(worker, f"invalid worker response: {exc}")
            return
        if message[0] == "ready":
            _, worker.pid, affinity = message
            if not worker.ready.done():
                worker.ready.set_result(affinity)
            return

        job_id, output, error, worker_received, worker_done = message
        # 正常情况下 worker.job 一定对应本次响应；job_id 校验可防止串包。
        job, worker.job = worker.job, None
        worker.count += 1
        if job is not None and job.job_id == job_id and not job.future.done():
            job.future.set_result((output, error, {
                "queue_ms": (job.dispatched_at - job.enqueued_at) * 1000.0,
                "ipc_ms": ((worker_received - job.dispatched_at)
                           + (received - worker_done)) * 1000.0,
                "worker_ms": (worker_done - worker_received) * 1000.0,
                "_wall": (received - job.enqueued_at) * 1000.0,
                "worker": worker.index,
            }))
        self._release(worker)

    def _worker_died(self, worker, detail=""):
        """清理死亡 worker，并把启动中或执行中的 future 标记为异常。"""
        if not worker.alive:
            return
        worker.alive = False
        with contextlib.suppress(Exception):
            self.loop.remove_reader(worker.fd)
        with contextlib.suppress(ValueError):
            self.idle.remove(worker)
        worker.process.join(timeout=0.2)
        worker.exitcode = worker.process.exitcode
        error = RuntimeError(
            f"FST worker {worker.index} (pid {worker.pid}) exited, "
            f"exitcode={worker.exitcode}" + (f", {detail}" if detail else ""))
        if not worker.ready.done():
            worker.ready.set_exception(error)
        job, worker.job = worker.job, None
        if job is not None and not job.future.done():
            job.future.set_exception(error)
        if not any(item.alive for item in self.processes):
            self._fail_all(error)

    def _fail_all(self, error):
        """池不可用时失败所有排队和执行中的任务，避免调用方永久等待。"""
        jobs = list(self.queue)
        jobs.extend(worker.job for worker in self.processes if worker.job is not None)
        self.queue.clear()
        for job in jobs:
            if not job.future.done():
                with contextlib.suppress(RuntimeError):
                    job.future.set_exception(error)

    def stats(self):
        """返回可直接 JSON 序列化的运行状态，供健康检查和压测读取。"""
        return {
            "kind": "pipe",
            "requests": self.requests,
            "jobs": self.jobs,
            "sched": self.sched,
            "cpus": self.cpus,
            "queue_len": len(self.queue),
            "queue_peak": self.queue_peak,
            "idle": len(self.idle),
            "alive": sum(worker.alive for worker in self.processes),
            "workers": [
                {
                    "index": worker.index,
                    "pid": worker.pid,
                    "cpu": worker.cpu,
                    "n": worker.count,
                    "busy": worker.job is not None,
                    "alive": worker.alive,
                    "exitcode": worker.exitcode,
                }
                for worker in self.processes
            ],
        }
