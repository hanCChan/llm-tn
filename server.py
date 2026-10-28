"""llm-tn 最终方案服务: 单阶段 A1 + N-gram S3 + FSTPipePool。

    python server.py                          # 0.0.0.0:8000
    python server.py --host 127.0.0.1 --port 9000 --model model/a1

    curl -s localhost:8000/normalize -H 'Content-Type: application/json' \
         -d '{"text": "会议3月5日下午2点，客服电话400-820-8820", "requestid": "abc123", "sdkappid": 1400000001}'

请求可带 requestid(不带则服务端生成), 响应回传同一个值; 日志以它为结构化字段,
按 requestid 就能捞出一路请求的全部日志。另可带 sdkappid, 同样进日志字段,
用来按业务方聚合。日志走 logger 包写 log/app.log,
不打 stdout —— start.sh 里 stdout 是丢弃的。

管线与 main.py 一致, 只是把 LLM 标注换成 vLLM AsyncLLM:
    LLM 标注(vLLM, GPU) -> FST Pipe Pool 两遍读音(pynini, CPU 多进程)

最终推理链:

* ``model/a1`` 从 model/v2 单阶段训练，输出 ``span<|TN_CLASS|>``；
* server 从 tokenizer added vocab 识别 A1，并解析 ``output.token_ids``；
* vLLM 使用 N-gram S3(k=4, lookup 1-4) 推测解码；
* 类别 Span 交给 FSTPipePool(4 workers) 做读音。

为什么用 AsyncLLM 而不是 LLM:
    AsyncLLM 支持 continuous batching —— 并发请求会被自动合批, 这是 vLLM 对
    0.6B 小模型最大的价值来源(单句延迟主要是 kernel launch 开销, 靠 CUDA graph
    削减; 吞吐靠合批)。同步 LLM 类会串行处理, 白扔这部分收益。

单请求时 FST 只有几毫秒，但全局锁会在并发 32 时形成明显的排队尾部；
因此 FST 使用独立进程池，每个 worker 持有自己的 TNReader。
"""
import argparse
import asyncio
import os
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

import logger
from logger import make_extra_tags
from logger import core as logger_core
from train.constants import TN_PREFIX, SEP, EOS_ID
from train.fst_pool import FSTOverloaded, FSTPipePool, FSTTimeout
from train.label_tokens import LABEL_TOKENS, parse_label_token_ids
from train.labels import parse_edits
from train.latex import delatex
from train.split import has_digit, split_chunks

MODEL = "model/a1"
MAX_NEW_TOKENS = 128
# 单次请求最多送 8 个含数字的片进模型; 超出的片原样返回, 不标注也不读音。
# 给 GPU 一个每请求硬上限, 长文本不会把 batch 占死。
MAX_CHUNKS = 8

_state = {
    "engine": None,
    "tok": None,
    "fst": None,
    "label_tokens": False,
    "spec_metrics": None,
    "spec_config": None,
    "error": None,
}


@asynccontextmanager
async def lifespan(app):
    """管理服务生命周期：并行预热 FST/模型，关闭时逆序释放资源。

    启动失败不让 uvicorn 直接退出；错误写入 ``_state``，健康检查和业务请求
    会把具体原因以 503 返回。这样容器探针和日志都能看到失败原因。
    """
    # 启动失败的 traceback 必须进文件: start.sh 把 stdout 丢了 /dev/null。
    tags = make_extra_tags("startup")
    fst = None
    fst_start = None
    try:
        if app.state.log_mode == "async":
            logger_core.enable_async()
        from transformers import AutoTokenizer
        from vllm import AsyncEngineArgs
        from vllm.v1.engine.async_llm import AsyncLLM
        from train.spec_metrics import SpecMetricsLogger, build_ngram_config

        t0 = time.perf_counter()
        logger.info(f"加载中 model={app.state.model} dtype={app.state.dtype} "
                    f"max_model_len={app.state.max_model_len} gpu_util={app.state.gpu_util} "
                    f"fst_workers={app.state.fst_workers} fst_cpus={app.state.fst_cpus}", extra=tags)
        fst = FSTPipePool(
            workers=app.state.fst_workers,
            sched=app.state.fst_sched,
            cpus=app.state.fst_cpus,
            max_queue=app.state.fst_max_queue,
            queue_timeout=app.state.fst_timeout,
        )
        fst_start = asyncio.create_task(fst.start())
        # FST 编译和 vLLM 初始化互不依赖，并行执行可缩短服务冷启动时间。
        _state["tok"] = AutoTokenizer.from_pretrained(app.state.model, fix_mistral_regex=True)
        # A1 相较 v2 新增了 12 个 added token。检测到它们时必须从 token_ids
        # 解析类别边界，因为 vLLM 的 output.text 默认会过滤 special tokens。
        added_vocab = _state["tok"].get_added_vocab()
        _state["label_tokens"] = set(LABEL_TOKENS.values()).issubset(added_vocab)
        spec_config = build_ngram_config(
            app.state.spec_mode,
            app.state.spec_k,
            app.state.spec_min,
            app.state.spec_max,
        )
        # Speculative Decoding 在 vLLM 0.11 中要求关闭 async scheduling。
        engine_args = AsyncEngineArgs(
            model=app.state.model,
            dtype=app.state.dtype,
            max_model_len=app.state.max_model_len,
            gpu_memory_utilization=app.state.gpu_util,
            speculative_config=spec_config,
            async_scheduling=False,
        )
        spec_loggers = None
        if spec_config is not None:
            SpecMetricsLogger.reset()
            _state["spec_metrics"] = SpecMetricsLogger
            spec_loggers = [SpecMetricsLogger]
        # AsyncEngineArgs 可能把内部配置转换为 SpeculativeConfig；调试接口
        # 保存一份纯 JSON 字段，避免 FastAPI 尝试序列化 torch.dtype。
        _state["spec_config"] = (
            {
                "method": spec_config["method"],
                "num_speculative_tokens": spec_config["num_speculative_tokens"],
                "prompt_lookup_min": spec_config["prompt_lookup_min"],
                "prompt_lookup_max": spec_config["prompt_lookup_max"],
            }
            if spec_config is not None else None
        )
        _state["engine"] = AsyncLLM.from_engine_args(
            engine_args, stat_loggers=spec_loggers)
        await fst_start
        _state["fst"] = fst
        logger.info(f"就绪 cost_ms={(time.perf_counter() - t0) * 1000:.0f}", extra=tags)
    except Exception as e:                     # noqa: BLE001 - 启动失败要能查到原因
        if fst_start is not None and not fst_start.done():
            fst_start.cancel()
            await asyncio.gather(fst_start, return_exceptions=True)
        if fst is not None:
            fst.close()
        _state["error"] = f"{type(e).__name__}: {e}"
        logger.exception(f"启动失败: {_state['error']}", extra=tags)
    try:
        yield
    finally:
        # 先停 GPU 引擎，再停 CPU 进程池；每步独立清理，避免前一步异常阻断后一步。
        engine = _state["engine"]
        _state["engine"] = None
        try:
            if engine is not None:
                engine.shutdown()
        finally:
            fst = _state["fst"]
            _state["fst"] = None
            try:
                if fst is not None:
                    fst.close()
            finally:
                _state["spec_metrics"] = None
                _state["spec_config"] = None
                # 先写退出日志再 stop_async；listener 会排空队列后恢复同步 handler。
                logger.info("已退出", extra=tags)
                logger_core.stop_async()


app = FastAPI(title="llm-tn", description="LLM 定类 + wetext FST 读音的中文文本规整", lifespan=lifespan)
app.state.model = MODEL
app.state.max_new_tokens = MAX_NEW_TOKENS
app.state.dtype = "auto"
app.state.max_model_len = 2048
app.state.gpu_util = 0.5
app.state.fst_workers = 4
app.state.fst_sched = "lifo"
app.state.fst_cpus = "auto"
app.state.fst_max_queue = 1024
app.state.fst_timeout = 30.0
app.state.log_mode = "full"
app.state.spec_mode = "ngram"
app.state.spec_k = 4
app.state.spec_min = 1
app.state.spec_max = 4


class HealthOut(BaseModel):
    status: str = Field(..., description="ok")
    model: str


class NormalizeIn(BaseModel):
    text: str = Field(..., description="待规整的中文文本")
    requestid: str = Field("", description="调用方请求 id, 用于串联日志; 不传则服务端生成")
    sdkappid: int = Field(0, description="调用方应用标识, 用于按业务方聚合日志")


class NormalizeOut(BaseModel):
    text: str
    spoken: str = Field(..., description="口语文本")
    edits: list[list[str]] = Field(..., description='LLM 标注 [[span, "CLASS"], ...]')
    requestid: str = Field(..., description="回传请求 id, 便于跨服务对账")


def _fst_ready():
    """FST 至少有一个存活 worker 时视为可用。"""
    fst = _state["fst"]
    return fst is not None and fst.stats()["alive"] > 0


async def _tag(text, request_id, tags):
    """vLLM greedy 标注。用 token ids 而非字符串喂入, 与 main.py 的
    add_special_tokens=False 保持一致, 避免 vLLM 自行加特殊 token。"""
    from vllm import SamplingParams, TokensPrompt

    prompt_ids = _state["tok"](TN_PREFIX + text + SEP, add_special_tokens=False).input_ids
    params = SamplingParams(
        max_tokens=app.state.max_new_tokens,
        temperature=0.0,                       # greedy, 对齐 main.py 的 do_sample=False
        stop_token_ids=[EOS_ID],
    )
    final = None
    async for out in _state["engine"].generate(
        TokensPrompt(prompt_token_ids=prompt_ids), params, request_id=request_id
    ):
        final = out
    if final is None:
        return []
    # 撞上限意味着后面的 span 全被丢掉, 但接口照样返回 200 —— 这个失败模式必须留痕。
    if final.outputs[0].finish_reason == "length":
        logger.warning(f"标注被截断 max_tokens={app.state.max_new_tokens} chunk={text}", extra=tags)
    output = final.outputs[0]
    if _state["label_tokens"]:
        return parse_label_token_ids(output.token_ids, _state["tok"])
    return parse_edits(output.text.strip())


@app.get("/health", response_model=HealthOut)
async def health():
    """就绪探针。未就绪返回 503(加载中或加载失败), detail 给出原因。"""
    if _state["engine"] is None or not _fst_ready():
        raise HTTPException(503, detail=_state["error"] or "模型加载中")
    return HealthOut(status="ok", model=app.state.model)


@app.get("/debug/fst")
async def debug_fst():
    """返回 FST 进程池实时状态；服务尚未就绪时返回空对象。"""
    return _state["fst"].stats() if _state["fst"] is not None else {}


@app.get("/debug/spec")
async def debug_spec():
    """返回实际 vLLM speculative decoding 配置和累计接收统计。"""
    metrics_cls = _state["spec_metrics"]
    metrics = metrics_cls.snapshot() if metrics_cls is not None else None
    return {
        "enabled": _state["spec_config"] is not None,
        "config": _state["spec_config"],
        "metrics": metrics,
    }


@app.post("/normalize", response_model=NormalizeOut)
async def normalize(body: NormalizeIn):
    """执行 LaTeX 改写、切片、LLM 定类、FST 读音和结构化日志记录。"""
    # 客户端不带 requestid 时服务端补一个, 保证每条日志都能定位到一路请求。
    requestid = body.requestid or uuid.uuid4().hex
    tags = make_extra_tags(requestid, str(body.sdkappid))
    if _state["engine"] is None or not _fst_ready():
        detail = _state["error"] or "模型尚未加载完成"
        logger.error(f"503 模型未就绪: {detail}", extra=tags)
        raise HTTPException(503, detail=detail)

    t0 = time.perf_counter()
    try:
        # LaTeX 记法先改写成 math 规则认得的写法(\frac{1}{2} -> 1/2), 再切片。
        # 注意: 之后的 edits/spoken 都以改写后的文本为基准, 而响应里的 text 是原文。
        written = delatex(body.text)
        if written != body.text:
            logger.info(f"event=latex | text={body.text} | written={written}", extra=tags)

        # 切片 -> 只取前 MAX_CHUNKS 个含数字的片送模型; 其余(无数字的、以及超出
        # 配额的)原样返回, 既不过 LLM 也不过 FST。切分无损, 拼回去不掉字。
        chunks = split_chunks(written)
        picked = [i for i, c in enumerate(chunks) if has_digit(c)]
        dropped = picked[MAX_CHUNKS:]
        picked = picked[:MAX_CHUNKS]
        logger.info(f"event=split | text={written} | chunks_num={len(chunks)} | "
                    f"llm_idx={picked} | chunks={chunks}", extra=tags)
        if dropped:
            logger.warning(f"含数字片超配额, {len(dropped)} 片原样返回 "
                           f"max_chunks={MAX_CHUNKS} idx={dropped}", extra=tags)

        # 同一请求的多片并发发出, vLLM 的 continuous batching 会合批,
        # 总延迟约等于最慢一片而非各片之和。
        edits_per = await asyncio.gather(*(
            _tag(chunks[i], f"{requestid}-{i}", tags) for i in picked
        ))
        t1 = time.perf_counter()

        tagged_map = dict(zip(picked, edits_per))
        spoken_per, _fst_timing = await _state["fst"].apply(chunks, tagged_map)
        t2 = time.perf_counter()
    except (FSTOverloaded, FSTTimeout) as e:
        logger.warning(f"503 FST 过载: {e}", extra=tags)
        raise HTTPException(503, detail=f"fst overloaded: {e}") from e
    except Exception as e:                     # noqa: BLE001 - 兜住单请求异常, 别让它冒到 uvicorn
        logger.exception(f"规整失败 text={body.text} err={type(e).__name__}: {e}", extra=tags)
        raise HTTPException(500, detail=f"{type(e).__name__}: {e}") from e

    spoken = "".join(spoken_per)
    edits = [e for per in edits_per for e in per]

    # 过了 LLM 的片各打一条: 长文本出问题时要能定位到具体哪一片标错。
    for i in picked:
        detail = " ".join(f"[{span} -> {label}]" for span, label in tagged_map[i]) or "-"
        logger.info(f"event=chunk | idx={i}/{len(chunks)} | text={chunks[i]} | "
                    f"tags={detail} | spoken={spoken_per[i]}", extra=tags)

    tagged = " ".join(f"[{span} -> {label}]" for span, label in edits)
    logger.info(f"event=normalize | text={written} | tags={tagged} | spoken={spoken}", extra=tags)
    logger.info(
        f"event=latency | llm_ms {(t1 - t0) * 1000:.1f} | "
        f"fst_ms {(t2 - t1) * 1000:.1f} | "
        f"cost_ms {(time.perf_counter() - t0) * 1000:.1f} | "
        f"text_len {len(body.text)} | edits_num {len(edits)} | "
        f"chunks_num {len(chunks)} | llm_chunks {len(picked)}",
        extra=tags,
    )
    return NormalizeOut(text=body.text, spoken=spoken, edits=edits, requestid=requestid)


def main():
    import uvicorn

    ap = argparse.ArgumentParser(description="llm-tn FastAPI + vLLM 服务")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--max-new", type=int, default=MAX_NEW_TOKENS, help="标注生成上限")
    ap.add_argument("--dtype", default="auto", help="L20 等 Ada/Hopper 用 bfloat16; T4(SM75) 只能 float16")
    ap.add_argument("--max-model-len", type=int, default=2048)
    ap.add_argument("--gpu-util", type=float, default=0.5,
                    help="0.6B 模型本身约 1.2GB, 不必给满; 其余留给 KV cache")
    # FST 参数默认值与 app.state 保持一致，start.sh 会逐项透传环境变量。
    ap.add_argument("--fst-workers", type=int, default=4, help="FST 进程数")
    ap.add_argument("--fst-sched", default="lifo", choices=["lifo", "fifo"])
    ap.add_argument("--fst-cpus", default="auto", help="FST worker 绑核: auto 或 CPU 列表")
    ap.add_argument("--fst-max-queue", type=int, default=1024,
                    help="FST 有界队列最大任务数")
    ap.add_argument("--fst-timeout", type=float, default=30.0,
                    help="单请求等待 FST 的最长秒数")
    ap.add_argument("--log-mode", default="full", choices=["full", "async"],
                    help="async 将结构化日志写入移到后台线程")
    ap.add_argument("--spec-mode", default=os.getenv("TN_SPEC_MODE", "ngram"),
                    choices=["off", "ngram"],
                    help="N-gram speculative decoding mode")
    ap.add_argument("--spec-k", type=int,
                    default=int(os.getenv("TN_SPEC_K", "4")),
                    help="maximum speculative tokens per step")
    ap.add_argument("--spec-min", type=int,
                    default=int(os.getenv("TN_SPEC_MIN", "1")),
                    help="minimum N-gram lookup window")
    ap.add_argument("--spec-max", type=int,
                    default=int(os.getenv("TN_SPEC_MAX", "4")),
                    help="maximum N-gram lookup window")
    args = ap.parse_args()

    app.state.model = args.model
    app.state.max_new_tokens = args.max_new
    app.state.dtype = args.dtype
    app.state.max_model_len = args.max_model_len
    app.state.gpu_util = args.gpu_util
    app.state.fst_workers = args.fst_workers
    app.state.fst_sched = args.fst_sched
    app.state.fst_cpus = args.fst_cpus
    app.state.fst_max_queue = args.fst_max_queue
    app.state.fst_timeout = args.fst_timeout
    app.state.log_mode = args.log_mode
    app.state.spec_mode = args.spec_mode
    app.state.spec_k = args.spec_k
    app.state.spec_min = args.spec_min
    app.state.spec_max = args.spec_max
    # vLLM v1 引擎在子进程里跑, fork 会和 CUDA 冲突。
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    # 业务日志由 logger 统一写文件；这里只把 uvicorn warning 以上信息留给容器日志。
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
