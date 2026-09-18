"""Lean replacement for the per-frame `code_predictor.generate()` call of Qwen3-TTS.

For every audio frame (12.5 per second) the talker asks its code predictor for the 15 remaining codebooks through
the generic Hugging Face `generate()`: a fixed-length loop of 2 + 14 tiny forward passes wrapped in a lot of
per-call Python work (generation config, logits processors, stopping criteria, a new cache…). On this machine that
overhead, not the GPU, dominated the time per frame.

This module runs the same loop directly: same forwards, same cache, and the same sampling as `generate()`
(temperature → top-k → top-p, then `torch.multinomial`; greedy when sampling is off). With the same random seed it
draws the same tokens. Only `.sequences` is returned because that is all the talker reads from the result.
"""

from __future__ import annotations

import types
from typing import Any

import torch


def _warp(logits: torch.Tensor, temperature: float | None, top_k: int | None, top_p: float | None) -> torch.Tensor:
    """Hugging Face's order and semantics: TemperatureLogitsWarper → TopKLogitsWarper → TopPLogitsWarper."""
    scores = logits.float()
    if temperature is not None and temperature != 1.0:
        scores = scores / temperature
    if top_k is not None and top_k > 0:
        k = min(top_k, scores.size(-1))
        kth = torch.topk(scores, k)[0][..., -1, None]
        scores = scores.masked_fill(scores < kth, -float("inf"))
    if top_p is not None and top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(scores, descending=False)
        cumulative = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
        remove_sorted = cumulative <= (1 - top_p)
        remove_sorted[..., -1:] = 0  # always keep at least one token
        remove = remove_sorted.scatter(1, sorted_indices, remove_sorted)
        scores = scores.masked_fill(remove, -float("inf"))
    return scores


def _next_token(logits: torch.Tensor, do_sample: bool, temperature, top_k, top_p) -> torch.Tensor:
    if not do_sample:
        return torch.argmax(logits, dim=-1)
    probs = torch.softmax(_warp(logits, temperature, top_k, top_p), dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(1)


def fast_generate(predictor: Any, inputs_embeds: torch.Tensor, max_new_tokens: int, do_sample: bool | None = None,
                  top_p: float | None = None, top_k: int | None = None, temperature: float | None = None,
                  **_: Any) -> types.SimpleNamespace:
    from transformers.cache_utils import DynamicCache

    config = predictor.generation_config
    do_sample = config.do_sample if do_sample is None else do_sample
    top_p = config.top_p if top_p is None else top_p
    top_k = config.top_k if top_k is None else top_k
    temperature = config.temperature if temperature is None else temperature

    cache = DynamicCache()
    length = inputs_embeds.shape[1]
    device = inputs_embeds.device
    with torch.inference_mode():
        out = predictor(inputs_embeds=inputs_embeds, past_key_values=cache, use_cache=True,
                        cache_position=torch.arange(length, device=device))
        tokens = []
        for step in range(max_new_tokens):
            token = _next_token(out.logits[:, -1, :], do_sample, temperature, top_k, top_p)
            tokens.append(token)
            if step == max_new_tokens - 1:
                break
            out = predictor(input_ids=token[:, None], past_key_values=cache, use_cache=True,
                            generation_steps=out.generation_steps,
                            cache_position=torch.tensor([length + step], device=device))
    return types.SimpleNamespace(sequences=torch.stack(tokens, dim=1))


class GraphedCodePredictor:
    """The same per-frame loop, with every forward replayed from a CUDA graph captured once at load time.

    Shapes never change inside a frame (batch 1, a 2-vector prefill, then one token per step, 16 cache slots), so
    each of the 15 steps is captured once with a static KV cache and a precomputed causal mask, and replayed per
    frame without Python or kernel-launch overhead. Sampling stays outside the graphs (same `_next_token`).

    Numerics: the static cache and explicit mask make SDPA pick another kernel, so logits differ from the original
    path at bf16 precision (~0.4 % relative, measured). The graphs themselves reproduce that eager path exactly.
    Anything unexpected (batch > 1, another length) falls back to `fast_generate`.
    """

    def __init__(self, predictor: Any, num_steps: int, hidden_size: int) -> None:
        from transformers.cache_utils import StaticCache

        self.predictor = predictor
        self.steps = num_steps
        param = next(predictor.parameters())
        device, dtype = param.device, param.dtype
        length = num_steps + 1  # 2 prefill positions + (num_steps - 1) decoded tokens
        self.cache = StaticCache(config=predictor.config, max_cache_len=length, max_batch_size=1, device=device,
                                 dtype=dtype)
        self.embeds = torch.zeros(1, 2, hidden_size, device=device, dtype=dtype)
        self.token = torch.zeros(1, 1, dtype=torch.long, device=device)
        allowed = torch.ones(length, length, dtype=torch.bool, device=device).tril()
        self.positions = [torch.arange(2, device=device)] + [
            torch.tensor([1 + s], device=device) for s in range(1, num_steps)]
        self.masks = [allowed[:2][None, None]] + [allowed[1 + s:2 + s][None, None] for s in range(1, num_steps)]
        self.graphs: list[torch.cuda.CUDAGraph] = []
        self.logits: list[torch.Tensor] = []

        side = torch.cuda.Stream(device=device)
        side.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(side), torch.inference_mode():
            for _ in range(3):  # warm-up: lazy cache allocation, cuBLAS workspaces
                for s in range(num_steps):
                    self._step(s)
        torch.cuda.current_stream(device).wait_stream(side)
        with torch.inference_mode():
            for s in range(num_steps):
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    out = self._step(s)
                self.graphs.append(graph)
                self.logits.append(out)

    def _step(self, s: int) -> torch.Tensor:
        mask = {"full_attention": self.masks[s], "sliding_attention": self.masks[s]}
        if s == 0:
            out = self.predictor(inputs_embeds=self.embeds, past_key_values=self.cache, use_cache=True,
                                 cache_position=self.positions[0], attention_mask=mask)
        else:
            out = self.predictor(input_ids=self.token, generation_steps=s, past_key_values=self.cache,
                                 use_cache=True, cache_position=self.positions[s], attention_mask=mask)
        return out.logits[:, -1, :]

    def generate(self, inputs_embeds: torch.Tensor, max_new_tokens: int, do_sample: bool | None = None,
                 top_p: float | None = None, top_k: int | None = None, temperature: float | None = None,
                 **kwargs: Any) -> types.SimpleNamespace:
        if inputs_embeds.shape[:2] != (1, 2) or max_new_tokens != self.steps:
            return fast_generate(self.predictor, inputs_embeds, max_new_tokens, do_sample, top_p, top_k,
                                 temperature, **kwargs)
        config = self.predictor.generation_config
        do_sample = config.do_sample if do_sample is None else do_sample
        top_p = config.top_p if top_p is None else top_p
        top_k = config.top_k if top_k is None else top_k
        temperature = config.temperature if temperature is None else temperature
        with torch.inference_mode():
            self.embeds.copy_(inputs_embeds)
            tokens = []
            for s in range(self.steps):
                self.graphs[s].replay()
                token = _next_token(self.logits[s], do_sample, temperature, top_k, top_p)
                tokens.append(token)
                if s < self.steps - 1:
                    self.token.copy_(token[:, None])
        return types.SimpleNamespace(sequences=torch.stack(tokens, dim=1))


def _predictor_of(model: Any) -> Any:
    return getattr(getattr(getattr(model, "model", None), "talker", None), "code_predictor", None)


def install(model: Any) -> bool:
    """Route the talker's code predictor through `fast_generate`. Returns False if the model layout is unexpected."""
    predictor = _predictor_of(model)
    if predictor is None or not hasattr(predictor, "generation_config"):
        return False
    predictor.generate = types.MethodType(fast_generate, predictor)
    return True


def install_graphs(model: Any) -> bool:
    """Replay the code predictor from CUDA graphs. False (and the plain fast loop stays) if it cannot be captured."""
    predictor = _predictor_of(model)
    talker = getattr(getattr(model, "model", None), "talker", None)
    if predictor is None or talker is None or not torch.cuda.is_available():
        return False
    if next(predictor.parameters()).device.type != "cuda":
        return False
    try:
        graphed = GraphedCodePredictor(predictor, talker.config.num_code_groups - 1, talker.config.hidden_size)
    except Exception:  # noqa: BLE001  (a capture failure only means no acceleration)
        return False
    predictor.generate = graphed.generate
    predictor._voicelab_graphs = graphed  # keep the graphs and their memory pool alive with the model
    return True
