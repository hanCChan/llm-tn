# =============================================================================
# llm-tn source header
# Path: train/spec_metrics.py
# 职责: vLLM N-gram 推测解码与引擎阶段指标采集
# 核心原理: 围绕类别 token 协议完成数据构造、训练、FST 读音或线上调度，保持训练与推理口径一致。
# =============================================================================

"""vLLM N-gram 推测解码与引擎阶段指标采集。

vLLM 0.11 的 ``SchedulerStats.spec_decoding_stats`` 是实际调度步产生的
Draft/Accepted 统计，``IterationStats`` 则提供 TTFT、ITL 和已完成请求的
queue/prefill/decode 时间。本模块将两者累计成进程内只读快照，供
``/debug/spec`` 查询。没有开启推测解码时不会被实例化。
"""

import copy
import threading

from vllm.v1.metrics.loggers import StatLoggerBase


class SpecMetricsLogger(StatLoggerBase):
    """累计 vLLM N-gram 推测解码统计。"""

    _lock = threading.RLock()
    _stats = None

    def __init__(self, vllm_config, engine_index=0):
        """保存 vLLM 配置与引擎编号；累计状态由类级锁统一保护。"""
        self.vllm_config = vllm_config
        self.engine_index = engine_index

    @classmethod
    def reset(cls):
        """清空并重建所有累计字段。

        原理: 推测解码指标按调度步和完成请求分别累计；时间指标使用
        count/sum/min/max 累加器，避免保存每个样本造成内存随运行时长增长。
        """
        with cls._lock:
            cls._stats = {
                "scheduler_observations": 0,
                "num_spec_tokens": 0,
                "num_drafts": 0,
                "num_draft_tokens": 0,
                "num_accepted_tokens": 0,
                "num_accepted_tokens_per_pos": [],
                "iteration_observations": 0,
                "num_prompt_tokens": 0,
                "num_generation_tokens": 0,
                "ttft": cls._new_accumulator(),
                "inter_token_latency": cls._new_accumulator(),
                "finished_requests": {
                    "count": 0,
                    "finish_reasons": {},
                    "e2e_latency": cls._new_accumulator(),
                    "queued_time": cls._new_accumulator(),
                    "prefill_time": cls._new_accumulator(),
                    "inference_time": cls._new_accumulator(),
                    "decode_time": cls._new_accumulator(),
                    "mean_time_per_output_token": cls._new_accumulator(),
                    "num_prompt_tokens": 0,
                    "num_generation_tokens": 0,
                },
            }

    @staticmethod
    def _new_accumulator():
        """创建一个可增量更新、查询时再计算 mean 的时间统计桶。"""
        return {"count": 0, "sum": 0.0, "min": None, "max": None}

    @classmethod
    def _observe(cls, accumulator, value):
        """把单个浮点观测合并进统计桶，同时更新计数、总和与极值。"""
        value = float(value)
        accumulator["count"] += 1
        accumulator["sum"] += value
        accumulator["min"] = (
            value if accumulator["min"] is None
            else min(accumulator["min"], value)
        )
        accumulator["max"] = (
            value if accumulator["max"] is None
            else max(accumulator["max"], value)
        )

    @classmethod
    def _finalize_accumulator(cls, accumulator):
        """复制统计桶并补算平均值，保留原始累计值不变。"""
        result = dict(accumulator)
        result["mean"] = (
            result["sum"] / result["count"] if result["count"] else None)
        return result

    @classmethod
    def record(cls, scheduler_stats, iteration_stats, engine_idx=0):
        """累计 vLLM 一个调度步的推测解码与请求阶段指标。

        原理: ``SchedulerStats`` 提供 Draft/Accepted token 数，
        ``IterationStats`` 提供 TTFT、ITL、排队/预填充/解码耗时和完成原因。
        两类数据在同一把可重入锁内合并，保证 ``/debug/spec`` 读到一致快照。
        """
        with cls._lock:
            if cls._stats is None:
                cls.reset()
            stats = cls._stats
            spec = (
                scheduler_stats.spec_decoding_stats
                if scheduler_stats is not None else None
            )
            if spec is not None:
                stats["scheduler_observations"] += 1
                stats["num_spec_tokens"] = int(spec.num_spec_tokens)
                stats["num_drafts"] += int(spec.num_drafts)
                stats["num_draft_tokens"] += int(spec.num_draft_tokens)
                stats["num_accepted_tokens"] += int(
                    spec.num_accepted_tokens)
                per_pos = stats["num_accepted_tokens_per_pos"]
                incoming = list(spec.num_accepted_tokens_per_pos)
                if len(per_pos) < len(incoming):
                    per_pos.extend([0] * (len(incoming) - len(per_pos)))
                for index, value in enumerate(incoming):
                    per_pos[index] += int(value)

            if iteration_stats is None:
                return
            stats["iteration_observations"] += 1
            stats["num_prompt_tokens"] += int(
                iteration_stats.num_prompt_tokens)
            stats["num_generation_tokens"] += int(
                iteration_stats.num_generation_tokens)
            for value in iteration_stats.time_to_first_tokens_iter:
                cls._observe(stats["ttft"], value)
            for value in iteration_stats.inter_token_latencies_iter:
                cls._observe(stats["inter_token_latency"], value)

            finished = stats["finished_requests"]
            for request in iteration_stats.finished_requests:
                finished["count"] += 1
                reason = str(request.finish_reason)
                reasons = finished["finish_reasons"]
                reasons[reason] = reasons.get(reason, 0) + 1
                for key in (
                    "e2e_latency",
                    "queued_time",
                    "prefill_time",
                    "inference_time",
                    "decode_time",
                    "mean_time_per_output_token",
                ):
                    cls._observe(finished[key], getattr(request, key))
                finished["num_prompt_tokens"] += int(
                    request.num_prompt_tokens)
                finished["num_generation_tokens"] += int(
                    request.num_generation_tokens)

    @classmethod
    def snapshot(cls):
        """返回当前指标深拷贝，并补算接受率、平均接受长度和均值。

        原理: 累计字段只在 ``record`` 中更新；派生指标在快照阶段计算，既
        避免重复累加，也避免调用方修改返回值影响后续统计。
        """
        with cls._lock:
            if cls._stats is None:
                return None
            # 快照会补充 mean 等派生字段，深拷贝避免污染后续累计值。
            stats = copy.deepcopy(cls._stats)

        drafts = stats["num_drafts"]
        draft_tokens = stats["num_draft_tokens"]
        accepted = stats["num_accepted_tokens"]
        stats["acceptance_rate"] = (
            accepted / draft_tokens if draft_tokens else None)
        stats["mean_acceptance_length"] = (
            1 + accepted / drafts if drafts else None)
        stats["acceptance_rate_per_pos"] = [
            value / drafts if drafts else None
            for value in stats["num_accepted_tokens_per_pos"]
        ]
        stats["ttft"] = cls._finalize_accumulator(stats["ttft"])
        stats["inter_token_latency"] = cls._finalize_accumulator(
            stats["inter_token_latency"])
        finished = stats["finished_requests"]
        for key in (
            "e2e_latency",
            "queued_time",
            "prefill_time",
            "inference_time",
            "decode_time",
            "mean_time_per_output_token",
        ):
            finished[key] = cls._finalize_accumulator(finished[key])
        stats["metric_source"] = (
            "vllm.v1.metrics.stats.SchedulerStats.spec_decoding_stats and "
            "vllm.v1.metrics.stats.IterationStats")
        return stats

    def log_engine_initialized(self):
        """满足 ``StatLoggerBase`` 接口；本模块无需响应引擎初始化事件。"""
        pass

    def log(self):
        """满足 ``StatLoggerBase`` 接口；指标由 ``record`` 主动累计。"""
        pass


def build_ngram_config(mode: str, num_tokens: int,
                       lookup_min: int, lookup_max: int):
    """构造 vLLM 0.11 N-gram 推测解码配置。

    原理: N-gram 方法从已有 prompt/生成 token 中查找匹配窗口，先草拟后续
    ``num_tokens`` 个 token，再由主模型并行验证；``lookup_min/max`` 控制
    匹配窗口长度。``mode=off`` 时返回 None，保持非推测路径不变。
    """
    if mode == "off":
        return None
    if mode != "ngram":
        raise ValueError(f"unknown speculative mode: {mode}")
    if num_tokens < 1:
        raise ValueError("spec-k must be positive")
    if lookup_min < 1 or lookup_min > lookup_max:
        raise ValueError("spec lookup range must satisfy 1 <= min <= max")
    return {
        "method": "ngram",
        "num_speculative_tokens": num_tokens,
        "prompt_lookup_min": lookup_min,
        "prompt_lookup_max": lookup_max,
    }
