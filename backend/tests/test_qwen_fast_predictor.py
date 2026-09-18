"""The fast code-predictor loop must sample exactly like Hugging Face's generate() (no GPU, no weights)."""

from __future__ import annotations

import types

import pytest
import torch

from app.engines.qwen3tts import fast_predictor

transformers = pytest.importorskip("transformers")


@pytest.mark.parametrize(("temperature", "top_k", "top_p"), [
    (0.9, 50, 1.0), (1.0, 0, 0.8), (0.7, 20, 0.95), (1.3, 5, 0.5),
])
def test_warp_matches_huggingface_warpers(temperature, top_k, top_p):
    from transformers.generation.logits_process import (
        LogitsProcessorList,
        TemperatureLogitsWarper,
        TopKLogitsWarper,
        TopPLogitsWarper,
    )

    torch.manual_seed(0)
    logits = torch.randn(2, 2048) * 3
    hf = LogitsProcessorList()
    if temperature != 1.0:
        hf.append(TemperatureLogitsWarper(temperature))
    if top_k:
        hf.append(TopKLogitsWarper(top_k))
    if top_p < 1.0:
        hf.append(TopPLogitsWarper(top_p))
    expected = hf(torch.zeros(2, 1, dtype=torch.long), logits.clone())
    ours = fast_predictor._warp(logits, temperature, top_k, top_p)
    assert torch.equal(torch.isinf(expected), torch.isinf(ours))  # same tokens kept
    finite = ~torch.isinf(expected)
    assert torch.allclose(expected[finite], ours[finite])


class TinyPredictor(torch.nn.Module):
    """Mimics the code predictor contract: prefill with embeddings, then one token per step with its own head."""

    def __init__(self, steps: int = 4, vocab: int = 32, dim: int = 8) -> None:
        super().__init__()
        self.generation_config = types.SimpleNamespace(do_sample=True, temperature=0.9, top_k=50, top_p=1.0)
        self.embed = torch.nn.ModuleList(torch.nn.Embedding(vocab, dim) for _ in range(steps))
        self.heads = torch.nn.ModuleList(torch.nn.Linear(dim, vocab) for _ in range(steps))
        self.calls: list[dict] = []

    def forward(self, inputs_embeds=None, input_ids=None, past_key_values=None, use_cache=None, cache_position=None,
                generation_steps=None):
        self.calls.append({"prefill": inputs_embeds is not None, "step": generation_steps,
                           "position": cache_position.tolist(), "cache": past_key_values})
        if inputs_embeds is not None:
            generation_steps = 0
            hidden = inputs_embeds
        else:
            hidden = self.embed[generation_steps - 1](input_ids)
        logits = self.heads[generation_steps](hidden)
        return types.SimpleNamespace(logits=logits, generation_steps=generation_steps + 1)


def test_loop_walks_the_heads_and_positions_like_generate():
    predictor = TinyPredictor(steps=4)
    embeds = torch.randn(1, 2, 8)
    out = fast_predictor.fast_generate(predictor, inputs_embeds=embeds, max_new_tokens=4, do_sample=False)
    assert out.sequences.shape == (1, 4)
    assert [c["prefill"] for c in predictor.calls] == [True, False, False, False]  # no forward after the last token
    assert [c["step"] for c in predictor.calls] == [None, 1, 2, 3]
    assert [c["position"] for c in predictor.calls] == [[0, 1], [2], [3], [4]]
    assert len({id(c["cache"]) for c in predictor.calls}) == 1  # one cache for the whole frame

    # greedy output is fully determined by the heads
    first = predictor.heads[0](embeds)[:, -1].argmax(-1)
    assert out.sequences[0, 0] == first[0]


def test_seeded_sampling_is_reproducible_and_uses_generation_config_defaults():
    predictor = TinyPredictor(steps=4)
    embeds = torch.randn(1, 2, 8)
    torch.manual_seed(7)
    a = fast_predictor.fast_generate(predictor, inputs_embeds=embeds, max_new_tokens=4).sequences
    torch.manual_seed(7)
    b = fast_predictor.fast_generate(predictor, inputs_embeds=embeds, max_new_tokens=4).sequences
    assert torch.equal(a, b)


def test_install_only_patches_the_expected_layout():
    predictor = TinyPredictor()
    model = types.SimpleNamespace(model=types.SimpleNamespace(talker=types.SimpleNamespace(code_predictor=predictor)))
    assert fast_predictor.install(model) is True
    out = predictor.generate(inputs_embeds=torch.randn(1, 2, 8), max_new_tokens=3, do_sample=False)
    assert out.sequences.shape == (1, 3)
    assert fast_predictor.install(types.SimpleNamespace(model=types.SimpleNamespace(talker=object()))) is False


def test_plugin_picks_graphs_then_direct_loop(monkeypatch, settings):
    from app.engines.qwen3tts.plugin import Qwen3TTSBackend

    engine = Qwen3TTSBackend()
    calls = []
    monkeypatch.setattr(fast_predictor, "install_graphs", lambda model: calls.append("graphs") or True)
    monkeypatch.setattr(fast_predictor, "install", lambda model: calls.append("loop") or True)
    assert engine._accelerate("cuda") == "cuda_graphs" and calls == ["graphs"]

    calls.clear()
    assert engine._accelerate("cpu") == "direct_loop" and calls == ["loop"]  # no graphs off the GPU

    calls.clear()
    monkeypatch.setattr(fast_predictor, "install_graphs", lambda model: calls.append("graphs") or False)
    assert engine._accelerate("cuda") == "direct_loop" and calls == ["graphs", "loop"]  # capture failed → loop

    calls.clear()
    monkeypatch.setattr(settings, "qwen_cuda_graphs", False)
    assert engine._accelerate("cuda") == "direct_loop" and calls == ["loop"]  # switched off in the settings

    monkeypatch.setattr(fast_predictor, "install", lambda model: False)
    assert engine._accelerate("cpu") == "none"


def test_install_graphs_needs_a_cuda_model():
    predictor = TinyPredictor()
    model = types.SimpleNamespace(model=types.SimpleNamespace(
        talker=types.SimpleNamespace(code_predictor=predictor, config=types.SimpleNamespace(num_code_groups=5,
                                                                                            hidden_size=8))))
    assert fast_predictor.install_graphs(model) is False  # CPU weights (or no CUDA): no capture attempted
