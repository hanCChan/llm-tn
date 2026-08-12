"""FastAPI + vLLM 服务: 中文文本规整(TN) HTTP 接口。

    python server.py                          # 0.0.0.0:8000
    python server.py --host 127.0.0.1 --port 9000 --model model/v2

    curl -s localhost:8000/normalize -H 'Content-Type: application/json' \
         -d '{"text": "会议3月5日下午2点，客服电话400-820-8820", "requestid": "abc123"}'

请求可带 requestid(不带则服务端生成), 响应回传同一个值; 日志以它为结构化字段,
按 requestid 就能捞出一路请求的全部日志。日志走 logger 包写 log/app.log,
不打 stdout —— start.sh 里 stdout 是丢弃的。

管线与 main.py 一致, 只是把 LLM 标注换成 vLLM AsyncLLM:
    LLM 标注(vLLM, GPU) -> TNReader 两遍读音(pynini, CPU)

为什么用 AsyncLLM 而不是 LLM:
    AsyncLLM 支持 continuous batching —— 并发请求会被自动合批, 这是 vLLM 对
    0.6B 小模型最大的价值来源(单句延迟主要是 kernel launch 开销, 靠 CUDA graph
    削减; 吞吐靠合批)。同步 LLM 类会串行处理, 白扔这部分收益。

实测(T4, fp32, HF transformers)单句 1.2s 里 LLM 占 99%、FST 仅 16ms,
所以 FST 那一段串行化不影响整体。
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
from train.constants import TN_PREFIX, SEP, EOS_ID, SEPARATOR
from train.latex import delatex
from train.split import has_digit, split_chunks
from train.tn_fst import TNReader

MODEL = "model/v2"
MAX_NEW_TOKENS = 128
# 单次请求最多送 8 个含数字的片进模型; 超出的片原样返回, 不标注也不读音。
# 给 GPU 一个每请求硬上限, 长文本不会把 batch 占死。
MAX_CHUNKS = 8

_state = {"engine": None, "tok": None, "reader": None, "error": None}
# pynini 的 FST 应用不保证线程安全; 只有 16ms, 串行化代价可忽略。
_fst_lock = asyncio.Lock()


def _parse_edits(text):
    """LLM 输出的多行 "span\\tCLASS:ctx" -> [[span, label], ...]。与 main.TN.tag 同构。"""
    edits = []
    for line in text.split("\n"):
        if SEPARATOR in line:
            span, _, label = line.partition(SEPARATOR)
            if span and label:
                edits.append([span, label])
    return edits


@asynccontextmanager
async def lifespan(app):
    """启动时建 vLLM 引擎 + 加载 wetext FST; 失败不退出, 由请求返回 503 暴露原因。"""
    # 启动失败的 traceback 必须进文件: start.sh 把 stdout 丢了 /dev/null。
    tags = make_extra_tags("startup")
    try:
        from transformers import AutoTokenizer
        from vllm import AsyncEngineArgs
        from vllm.v1.engine.async_llm import AsyncLLM

        t0 = time.perf_counter()
        logger.info(f"加载中 model={app.state.model} dtype={app.state.dtype} "
                    f"max_model_len={app.state.max_model_len} gpu_util={app.state.gpu_util}", extra=tags)
        _state["tok"] = AutoTokenizer.from_pretrained(app.state.model, fix_mistral_regex=True)
        _state["engine"] = AsyncLLM.from_engine_args(AsyncEngineArgs(
            model=app.state.model,
            dtype=app.state.dtype,
            max_model_len=app.state.max_model_len,
            gpu_memory_utilization=app.state.gpu_util,
        ))
        # FST 首次冷启动要编译(约十几秒), 放线程池里不挡事件循环。
        _state["reader"] = await asyncio.to_thread(TNReader)
        logger.info(f"就绪 cost_ms={(time.perf_counter() - t0) * 1000:.0f}", extra=tags)
    except Exception as e:                     # noqa: BLE001 - 启动失败要能查到原因
        _state["error"] = f"{type(e).__name__}: {e}"
        logger.exception(f"启动失败: {_state['error']}", extra=tags)
    yield
    engine = _state["engine"]
    _state["engine"] = None
    if engine is not None:
        engine.shutdown()
    logger.info("已退出", extra=tags)


app = FastAPI(title="llm-tn", description="LLM 定类 + wetext FST 读音的中文文本规整", lifespan=lifespan)
app.state.model = MODEL
app.state.max_new_tokens = MAX_NEW_TOKENS
app.state.dtype = "auto"
app.state.max_model_len = 2048
app.state.gpu_util = 0.5


class HealthOut(BaseModel):
    status: str = Field(..., description="ok")
    model: str


class NormalizeIn(BaseModel):
    text: str = Field(..., description="待规整的中文文本")
    requestid: str = Field("", description="调用方请求 id, 用于串联日志; 不传则服务端生成")


class NormalizeOut(BaseModel):
    text: str
    spoken: str = Field(..., description="口语文本")
    edits: list[list[str]] = Field(..., description='LLM 标注 [[span, "CLASS:ctx"], ...]')
    requestid: str = Field(..., description="回传请求 id, 便于跨服务对账")


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
    return _parse_edits(final.outputs[0].text.strip())


@app.get("/health", response_model=HealthOut)
async def health():
    """就绪探针。未就绪返回 503(加载中或加载失败), detail 给出原因。"""
    if _state["engine"] is None or _state["reader"] is None:
        raise HTTPException(503, detail=_state["error"] or "模型加载中")
    return HealthOut(status="ok", model=app.state.model)


@app.post("/normalize", response_model=NormalizeOut)
async def normalize(body: NormalizeIn):
    # 客户端不带 requestid 时服务端补一个, 保证每条日志都能定位到一路请求。
    requestid = body.requestid or uuid.uuid4().hex
    tags = make_extra_tags(requestid)
    if _state["engine"] is None or _state["reader"] is None:
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
        async with _fst_lock:
            spoken_per = await asyncio.to_thread(
                lambda: [_state["reader"].apply(c, tagged_map[i]) if i in tagged_map else c
                         for i, c in enumerate(chunks)]
            )
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
    args = ap.parse_args()

    app.state.model = args.model
    app.state.max_new_tokens = args.max_new
    app.state.dtype = args.dtype
    app.state.max_model_len = args.max_model_len
    app.state.gpu_util = args.gpu_util
    # vLLM v1 引擎在子进程里跑, fork 会和 CUDA 冲突。
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
