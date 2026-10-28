"""vLLM pooling adapter for the character-level NAR tagger.

The trained NAR checkpoint is a Qwen3 backbone plus a 25-class token
classifier.  vLLM does not ship that task for Qwen3, but its pooling runner can
return every token's hidden state.  This adapter keeps vLLM's native Qwen3
attention, KV cache, scheduler and continuous batching, then applies the NAR
head before the ``ALL`` pooler turns the output back into per-character logits.
"""
from collections.abc import Iterable

import torch
from torch import nn

from vllm.config import VllmConfig
from vllm.distributed import get_pp_group
from vllm.model_executor.models.qwen3 import Qwen3ForCausalLM
from vllm.model_executor.models.utils import WeightsMapper, maybe_prefix
from vllm.model_executor.layers.pooler import AllPool, DispatchPooler, Pooler
from vllm.sequence import IntermediateTensors


class _NARAllTokenPooler(Pooler):
    """Return every token's NAR logits without changing their dtype."""

    def __init__(self) -> None:
        super().__init__()
        self.pooling = AllPool()

    def get_supported_tasks(self) -> set[str]:
        return {"encode"}

    def forward(self, hidden_states, pooling_metadata):
        return self.pooling(hidden_states, pooling_metadata)


class NARQwen3ForTokenClassification(Qwen3ForCausalLM):
    """Qwen3 causal backbone with NAR's per-token BIO classifier."""

    is_pooling_model = True
    default_pooling_type = "ALL"

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        config = vllm_config.model_config.hf_config
        # Generation-only modules are not used by the pooling runner.
        self.lm_head = None
        self.logits_processor = None

        self.score = nn.Linear(
            config.hidden_size,
            config.num_labels,
            bias=True,
        )
        self.score.to(dtype=vllm_config.model_config.dtype)
        self.head_dtype = self.score.weight.dtype
        self.pooler = DispatchPooler({"encode": _NARAllTokenPooler()})

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ):
        hidden_states = self.model(
            input_ids,
            positions,
            intermediate_tensors,
            inputs_embeds,
        )
        if not get_pp_group().is_last_rank:
            return IntermediateTensors({"hidden_states": hidden_states})
        return self.score(hidden_states.to(self.head_dtype))

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        mapper = WeightsMapper(orig_to_new_prefix={
            "backbone.": "model.",
            "head.": "score.",
        })
        return super().load_weights(mapper.apply(weights))


def register_vllm_nar() -> None:
    """Register the adapter for vLLM model discovery."""
    from vllm import ModelRegistry

    ModelRegistry.register_model(
        "NARQwen3ForTokenClassification",
        "train.vllm_nar:NARQwen3ForTokenClassification",
    )
