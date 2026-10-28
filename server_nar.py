"""NAR FastAPI service: vLLM pooling + final FSTPipePool.

The NAR tagger is exposed to vLLM through ``NARQwen3ForTokenClassification``.
Each selected text chunk becomes one vLLM pooling request; vLLM's scheduler
batches independent chunks and HTTP requests.  FST pronunciation runs in the
bounded multi-process pool ported from the final service.
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
from train.fst_pool import FSTOverloaded, FSTPipePool, FSTTimeout
from train.latex import delatex
from train.nar import MAX_CHARS, CharTokenizer, _CLS_LABEL, decode_offsets, echo_ids
from train.split import has_digit, split_chunks
from train.vllm_nar import register_vllm_nar

DEFAULT_MODEL = "model/nar"
MAX_CHUNKS = 8

# vLLM's engine core is a spawned process. Registering at import time makes the
# lazy model entry available again when spawn imports this module in the child.
register_vllm_nar()

_state = {
    "engine": None,
    "ctok": None,
    "fst": None,
    "params": None,
    "error": None,
}


@asynccontextmanager
async def lifespan(app):
    tags = make_extra_tags("startup")
    try:
        from vllm import AsyncEngineArgs, PoolingParams
        from vllm.config import PoolerConfig
        from vllm.v1.engine.async_llm import AsyncLLM

        t0 = time.perf_counter()
        logger.info(
            f"loading NAR model={app.state.model} dtype={app.state.dtype} "
            f"max_model_len={app.state.max_model_len} gpu_util={app.state.gpu_util}",
            extra=tags,
        )
        _state["ctok"] = CharTokenizer(app.state.model)
        _state["engine"] = AsyncLLM.from_engine_args(AsyncEngineArgs(
            model=app.state.model,
            runner="pooling",
            convert="none",
            dtype=app.state.dtype,
            max_model_len=app.state.max_model_len,
            gpu_memory_utilization=app.state.gpu_util,
            max_num_seqs=app.state.max_num_seqs,
            enforce_eager=app.state.enforce_eager,
            hf_overrides={
                "architectures": ["NARQwen3ForTokenClassification"],
                "num_labels": 25,
            },
            pooler_config=PoolerConfig(pooling_type="ALL"),
        ))
        _state["params"] = PoolingParams(task="encode", softmax=False)

        fst = FSTPipePool(
            workers=app.state.fst_workers,
            sched="lifo",
            cpus=app.state.fst_cpus,
            max_queue=app.state.fst_max_queue,
            queue_timeout=app.state.deadline_s,
        )
        _state["fst"] = fst
        await fst.start()
        logger.info(
            f"NAR ready engine=OK fst_workers={app.state.fst_workers} "
            f"cost_ms={(time.perf_counter() - t0) * 1000:.0f}",
            extra=tags,
        )
    except Exception as exc:  # noqa: BLE001 - health must expose startup failure
        _state["error"] = f"{type(exc).__name__}: {exc}"
        logger.exception(f"startup failed: {_state['error']}", extra=tags)
    yield
    fst = _state["fst"]
    _state["fst"] = None
    if fst is not None:
        fst.close()
    engine = _state["engine"]
    _state["engine"] = None
    if engine is not None:
        engine.shutdown()
    logger.info("stopped", extra=tags)


app = FastAPI(title="llm-tn-nar", lifespan=lifespan)
app.state.model = DEFAULT_MODEL
app.state.dtype = "bfloat16"
app.state.max_model_len = 2048
app.state.gpu_util = 0.3
app.state.max_num_seqs = 64
app.state.enforce_eager = False
app.state.fst_workers = 16
app.state.fst_cpus = "auto"
app.state.fst_max_queue = 4096
app.state.deadline_s = 2.0


class HealthOut(BaseModel):
    status: str
    model: str


class NormalizeIn(BaseModel):
    text: str = Field(..., description="待规整文本")
    requestid: str = Field("", description="请求 id")
    sdkappid: int = Field(0, description="调用方应用标识")


class NormalizeOut(BaseModel):
    text: str
    spoken: str
    edits: list[list[str]]
    requestid: str


async def _tag_chunk(text, request_id):
    from vllm import TokensPrompt

    ids, offset = echo_ids(_state["ctok"].encode(text))
    final = None
    async for out in _state["engine"].encode(
        TokensPrompt(prompt_token_ids=ids),
        _state["params"],
        request_id=request_id,
    ):
        final = out
    if final is None:
        raise RuntimeError("vLLM returned no pooling output")

    # ALL pooling returns every input token. Only the second copy is supervised
    # by NAR and corresponds one-to-one with source characters.
    logits = final.outputs.data[offset:offset + len(text)]
    probs = logits.float().softmax(dim=-1).cpu().numpy()
    spans = decode_offsets(probs)
    return [[text[start:end], _CLS_LABEL[cls]] for start, end, cls in spans]


@app.get("/health", response_model=HealthOut)
async def health():
    if _state["engine"] is None or _state["fst"] is None:
        raise HTTPException(503, detail=_state["error"] or "model loading")
    if not all(worker["alive"] for worker in _state["fst"].stats()["workers"]):
        raise HTTPException(503, detail="FST pool degraded")
    return HealthOut(status="ok", model=app.state.model)


@app.get("/debug/fst")
async def debug_fst():
    if _state["fst"] is None:
        raise HTTPException(503, detail="FST pool unavailable")
    return _state["fst"].stats()


@app.post("/normalize", response_model=NormalizeOut)
async def normalize(body: NormalizeIn):
    requestid = body.requestid or uuid.uuid4().hex
    tags = make_extra_tags(requestid, str(body.sdkappid))
    if _state["engine"] is None or _state["fst"] is None:
        raise HTTPException(503, detail=_state["error"] or "model loading")

    started = time.perf_counter()
    written = delatex(body.text)
    # NAR accepts a whole short sentence and benefits from retaining all of its
    # context. Only long text needs the service-side chunking used by the AR
    # baseline; splitting every punctuation boundary lost quality on test373.
    chunks = [written] if len(written) <= MAX_CHARS else split_chunks(written)
    picked = [i for i, chunk in enumerate(chunks) if has_digit(chunk)]
    picked = picked[:MAX_CHUNKS]
    try:
        tagged = await asyncio.gather(*(
            _tag_chunk(chunks[i], f"{requestid}-{i}") for i in picked
        ))
        edits_map = dict(zip(picked, tagged))
        spoken, timing = await _state["fst"].apply(chunks, edits_map)
    except (FSTOverloaded, FSTTimeout) as exc:
        logger.warning(f"503 FST overload: {exc}", extra=tags)
        raise HTTPException(503, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception(f"NAR request failed: {type(exc).__name__}: {exc}", extra=tags)
        raise HTTPException(500, detail=f"{type(exc).__name__}: {exc}") from exc

    edits = [item for per in tagged for item in per]
    logger.info(
        f"event=normalize | text_len={len(body.text)} | chunks={len(chunks)} "
        f"| picked={len(picked)} | fst_ms={timing['fst_ms']:.2f} "
        f"| cost_ms={(time.perf_counter() - started) * 1000:.2f}",
        extra=tags,
    )
    return NormalizeOut(
        text=body.text,
        spoken="".join(spoken),
        edits=edits,
        requestid=requestid,
    )


def main():
    import uvicorn

    parser = argparse.ArgumentParser(description="vLLM NAR + FSTPipePool service")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--gpu-util", type=float, default=0.3)
    parser.add_argument("--max-num-seqs", type=int, default=64)
    parser.add_argument("--enforce-eager", action=argparse.BooleanOptionalAction,
                        default=False)
    parser.add_argument("--fst-workers", type=int, default=16)
    parser.add_argument("--fst-cpus", default="auto")
    parser.add_argument("--fst-max-queue", type=int, default=4096)
    parser.add_argument("--deadline-ms", type=float, default=2000.0)
    args = parser.parse_args()

    app.state.model = args.model
    app.state.dtype = args.dtype
    app.state.max_model_len = args.max_model_len
    app.state.gpu_util = args.gpu_util
    app.state.max_num_seqs = args.max_num_seqs
    app.state.enforce_eager = args.enforce_eager
    app.state.fst_workers = args.fst_workers
    app.state.fst_cpus = args.fst_cpus
    app.state.fst_max_queue = args.fst_max_queue
    app.state.deadline_s = args.deadline_ms / 1000.0
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    uvicorn.run(app, host=args.host, port=args.port, access_log=False)


if __name__ == "__main__":
    main()
