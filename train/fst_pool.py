# =============================================================================
# llm-tn source header
# Path: train/fst_pool.py
# 职责: 面向线上服务的非阻塞 FST 有界多进程池
# 核心原理: 主进程通过非阻塞 pipe fd 调度，worker 进程独立持有 TNReader 执行 FST。
# =============================================================================

"""面向线上服务的非阻塞 FST 有界多进程池。

总体怎么做
==========
1. 每个 spawn worker 独立持有 ``TNReader``，避开 pynini 的线程安全和全局锁问题。
2. 主进程把 pipe 设为非阻塞，通过 ``add_reader/add_writer`` 驱动 ``os.read/os.write``。
3. worker 进程内仍使用阻塞读写，但它在独立进程中，不占用服务事件循环。
4. 每个 HTTP 请求的 chunk 进入有界队列，由空闲 worker 按策略领取并执行读音。
5. 任务通过长度前缀帧传输；部分读写保留在每 worker 自己的缓冲区中。

核心函数
========
* ``start``: spawn worker、设置非阻塞 fd、等待 TNReader 就绪。
* ``apply``: 请求级 FST 入口，负责排队、超时、结果重组。
* ``_submit/_send/_flush_write``: 创建任务并把长度前缀帧写入 pipe。
* ``_on_readable/_handle_message``: 增量读帧、反序列化、完成 future。
* ``close``: 停止 worker、回收进程、失败残留任务。

函数如何协作
============
``apply`` -> ``_submit`` -> ``_send`` -> 非阻塞 ``os.write`` -> worker 进程执行
``TNReader.apply`` -> worker 写回帧 -> ``_on_readable`` -> ``_handle_message``
-> future 唤醒 -> ``apply`` 返回读音。事件循环从不调用阻塞式
``recv_bytes/send_bytes``。

关键参数
========
* ``workers``: FST 进程数，每个 worker 独立初始化一个 TNReader。
* ``max_queue``: 等待队列上限，超限抛 ``FSTOverloaded``。
* ``queue_timeout``: 单请求等待 FST 读音完成的总时限。
* ``cpus``: CPU 亲和性；``auto`` 会尽量分散到不同物理核心。
* ``sched``: 空闲 worker 领取顺序，``lifo`` 优先复用刚完成任务的 worker。
"""

import asyncio
import collections
import contextlib
import multiprocessing as mp
import os
import pickle
import signal
import struct
import sys
import time

from train.tn_fst import TNReader

# FST 首次启动可能需要编译规则；生产镜像已带缓存，仍给冷启动留足时间。
START_TIMEOUT = 300.0
_perf = time.perf_counter
_FRAME = struct.Struct("!I")


class FSTOverloaded(RuntimeError):
    """队列已满，调用方应快速返回 503 而不是继续等待。"""

    pass


class FSTTimeout(RuntimeError):
    """单次请求等待 FST 总时长超过配置上限。"""

    pass


def _read_exact(fd, size):
    """Read exactly ``size`` bytes from a blocking worker-side fd."""
    chunks = bytearray()
    while len(chunks) < size:
        chunk = os.read(fd, size - len(chunks))
        if not chunk:
            raise EOFError
        chunks.extend(chunk)
    return bytes(chunks)


def _send_frame_blocking(fd, message):
    """Worker-side length-prefixed frame write."""
    payload = pickle.dumps(message)
    data = memoryview(_FRAME.pack(len(payload)) + payload)
    while data:
        written = os.write(fd, data)
        data = data[written:]


def _recv_frame_blocking(fd):
    """Worker-side length-prefixed frame read."""
    header = _read_exact(fd, _FRAME.size)
    size, = _FRAME.unpack(header)
    return pickle.loads(_read_exact(fd, size))


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
    fd = conn.fileno()
    reader = TNReader()
    _send_frame_blocking(
        fd, ("ready", os.getpid(), sorted(os.sched_getaffinity(0))))
    while True:
        try:
            message = _recv_frame_blocking(fd)
        except (EOFError, OSError):
            return
        t_recv = _perf()
        if message is None:
            return
        job_id, chunk, edits = message
        try:
            output = reader.apply(chunk, edits)
            error = None
        except Exception as exc:  # noqa: BLE001 - report the work error without killing worker
            output = None
            error = f"{type(exc).__name__}: {exc}"
        _send_frame_blocking(fd, (job_id, output, error, t_recv, _perf()))


class _Worker:
    """主进程侧的 worker 句柄，集中保存进程、fd、忙碌任务和健康状态。"""

    __slots__ = ("index", "process", "conn", "fd", "cpu", "pid", "ready",
                 "job", "count", "alive", "exitcode", "read_buf", "write_buf",
                 "writing")

    def __init__(self, index, process, conn, cpu, ready):
        """保存 worker 身份、通信 fd、绑核结果和初始就绪 future。"""
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
        self.read_buf = bytearray()
        self.write_buf = bytearray()
        self.writing = False


class _Job:
    """一次 FST chunk 任务及其时间戳、future 和当前 worker。"""

    __slots__ = ("job_id", "chunk", "edits", "future", "enqueued_at",
                 "dispatched_at", "worker")

    def __init__(self, job_id, chunk, edits, future, enqueued_at):
        """记录任务内容、调用方 future 以及排队和派发时间戳。"""
        self.job_id = job_id
        self.chunk = chunk
        self.edits = edits
        self.future = future
        self.enqueued_at = enqueued_at
        self.dispatched_at = None
        self.worker = None


class FSTPipePool:
    """有界、非阻塞 pipe 驱动的 FST 进程池。

    事件循环线程负责排队、非阻塞 pipe I/O、消息解帧和 future 完成；worker
    进程负责持有 TNReader 并串行执行 FST。这样 pynini 计算和 pipe 等待都不会
    阻塞 asyncio。
    """

    def __init__(self, workers=8, sched="lifo", cpus=None, max_queue=4096,
                 queue_timeout=30.0):
        """保存池配置并初始化等待队列和 worker 状态。

        参数:
            workers: FST worker 进程数，至少为 1。
            sched: ``lifo``/``fifo``，决定空闲 worker 复用顺序。
            cpus: ``auto``、显式 CPU 列表或空值。
            max_queue: 待派发任务上限，用于过载保护。
            queue_timeout: 单请求 FST 总超时，包含排队与执行。
        """
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
        """spawn worker、设置非阻塞 fd，并等待所有 TNReader 初始化。

        启动阶段同步执行 ``process.start()`` 和 CPU topology 探测；它们不在在线
        请求路径上。worker 就绪前，所有 pipe fd 已注册到事件循环。
        """
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
                # The event loop owns parent-side pipe I/O. Non-blocking fd makes
                # partial writes/reads observable through add_reader/add_writer.
                os.set_blocking(parent.fileno(), False)
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
        """停止并回收 worker；超时 terminate，最后失败所有剩余任务。

        该函数只用于服务退出或启动失败清理。为了尽快结束，终止帧和 ``join``
        允许同步等待，但不会发生在正常 HTTP 请求期间。
        """
        if self._closed:
            return
        self._closed = True
        for worker in self.processes:
            if worker.alive:
                with contextlib.suppress(Exception):
                    self.loop.remove_reader(worker.fd)
                with contextlib.suppress(Exception):
                    self.loop.remove_writer(worker.fd)
                with contextlib.suppress(Exception):
                    os.set_blocking(worker.fd, True)
                    _send_frame_blocking(worker.fd, None)
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
        """执行一个 HTTP 请求所需的全部 FST 读音任务。

        参数:
            chunks: 无损切片后的完整文本片列表，顺序必须与原文拼接一致。
            edits_map: ``{chunk_index: edits}``，只包含需要进入 FST 的下标。

        返回:
            ``(spoken_chunks, timing)``。结果保持原 chunk 顺序，timing 包含
            总 FST 耗时以及最慢任务的 queue/IPC/worker 拆分。

        异常:
            FSTOverloaded: 等待队列超过 ``max_queue``。
            FSTTimeout: 单请求等待超过 ``queue_timeout``。
            RuntimeError: worker 不可用或 worker 执行失败。
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
        """为单个 chunk 创建任务并尝试立即派发。

        参数:
            chunk: 待读音原文片。
            edits: 该片内的 ``[span, CLASS]`` 标注。
            enqueued_at: 请求级开始时间，用于统计 queue_ms。
        """
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
        """把任务编码为长度前缀帧并写入非阻塞 pipe。

        参数:
            worker: 已领取任务的 worker。
            job: 包含 chunk、edits、future 和时间戳的任务对象。

        ``pickle.dumps`` 只处理小对象；可能阻塞的 pipe 写由非阻塞
        ``os.write`` 完成，未写部分继续由 ``add_writer`` 驱动。
        """
        job.dispatched_at = _perf()
        job.worker = worker
        worker.job = job
        try:
            payload = pickle.dumps((job.job_id, job.chunk, job.edits))
            worker.write_buf.extend(_FRAME.pack(len(payload)))
            worker.write_buf.extend(payload)
            self._flush_write(worker)
        except Exception as exc:  # noqa: BLE001 - worker death must be explicit
            self._worker_died(worker, f"send failed: {exc}")

    def _flush_write(self, worker):
        """尽量排空 worker 的写缓冲。

        参数:
            worker: 持有非阻塞 fd 和 ``write_buf`` 的 worker。

        遇到 ``BlockingIOError`` 时保留剩余字节并注册 writer；写空后移除
        writer，避免事件循环持续空转。
        """
        try:
            while worker.write_buf:
                written = os.write(worker.fd, worker.write_buf)
                if written <= 0:
                    raise BrokenPipeError
                del worker.write_buf[:written]
        except BlockingIOError:
            pass
        except OSError as exc:
            self._worker_died(worker, f"write failed: {exc}")
            return
        if worker.write_buf and not worker.writing:
            worker.writing = True
            self.loop.add_writer(worker.fd, self._on_writable, worker)
        elif not worker.write_buf and worker.writing:
            worker.writing = False
            with contextlib.suppress(Exception):
                self.loop.remove_writer(worker.fd)

    def _on_writable(self, worker):
        """pipe 可写回调；只执行非阻塞增量续写。"""
        if worker.alive:
            self._flush_write(worker)

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
        """pipe 可读回调；增量读帧并完成已收齐的消息。

        参数:
            worker: 触发可读事件的 worker。

        不使用 ``recv_bytes``，因此不会等待完整消息；不完整帧保留在
        ``read_buf``，等下一次可读事件继续拼接。
        """
        if not worker.alive:
            return
        try:
            chunk = os.read(worker.fd, 65536)
        except BlockingIOError:
            return
        except OSError as exc:
            self._worker_died(worker, f"read failed: {exc}")
            return
        if not chunk:
            self._worker_died(worker, "pipe closed")
            return
        worker.read_buf.extend(chunk)
        while len(worker.read_buf) >= _FRAME.size:
            size, = _FRAME.unpack(worker.read_buf[: _FRAME.size])
            total = _FRAME.size + size
            if len(worker.read_buf) < total:
                break
            payload = bytes(worker.read_buf[_FRAME.size:total])
            del worker.read_buf[:total]
            try:
                message = pickle.loads(payload)
            except Exception as exc:  # noqa: BLE001
                self._worker_died(worker, f"invalid worker response: {exc}")
                return
            self._handle_message(worker, message, _perf())

    def _handle_message(self, worker, message, received):
        """处理已解码消息、完成 future 并派发下一个排队任务。

        参数:
            worker: 返回消息的 worker。
            message: ready 消息或任务响应。
            received: 主进程完成本次读取的时间戳。

        该函数只做内存状态更新、耗时计算和轻量调度，不执行 FST。
        """
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
        """清理异常 worker，并显式失败其 future。

        参数:
            worker: 已退出或 pipe 异常的 worker。
            detail: 追加到异常信息中的诊断原因。

        使用 ``join(0)`` 只回收已退出进程，不在事件循环中等待。
        """
        if not worker.alive:
            return
        worker.alive = False
        with contextlib.suppress(Exception):
            self.loop.remove_reader(worker.fd)
        with contextlib.suppress(Exception):
            self.loop.remove_writer(worker.fd)
        worker.read_buf.clear()
        worker.write_buf.clear()
        with contextlib.suppress(ValueError):
            self.idle.remove(worker)
        # Failure handling runs on the event loop; never wait for process exit here.
        worker.process.join(timeout=0)
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
        """返回可 JSON 序列化的队列、worker 和累计处理状态。"""
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
