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
    outcome = {"talker": True, "graphs": True, "loop": True}

    def fake(name):
        return lambda model: calls.append(name) or outcome[name]

    for name, attr in (("talker", "install_talker_graphs"), ("graphs", "install_graphs"), ("loop", "install")):
        monkeypatch.setattr(fast_predictor, attr, fake(name))
    monkeypatch.setattr(engine, "_check_random_generator", lambda: calls.append("rng"))

    assert engine._accelerate("cuda") == "cuda_graphs" and calls == ["talker", "graphs", "rng"]

    calls.clear()
    assert engine._accelerate("cpu") == "direct_loop" and calls == ["loop"]  # no graphs off the GPU

    calls.clear()
    outcome["talker"] = False
    assert engine._accelerate("cuda") == "cuda_graphs_partial" and calls == ["talker", "graphs", "rng"]

    calls.clear()
    outcome.update(talker=True, graphs=False)
    assert engine._accelerate("cuda") == "cuda_graphs_partial" and calls == ["talker", "graphs", "rng", "loop"]

    calls.clear()
    outcome.update(talker=False, graphs=False)
    assert engine._accelerate("cuda") == "direct_loop" and calls == ["talker", "graphs", "loop"]  # both failed

    calls.clear()
    monkeypatch.setattr(settings, "qwen_cuda_graphs", False)
    assert engine._accelerate("cuda") == "direct_loop" and calls == ["loop"]  # switched off in the settings

    outcome["loop"] = False
    assert engine._accelerate("cpu") == "none"


def test_broken_random_generator_unloads_with_a_clear_message(monkeypatch):
    from app.core.errors import AppError
    from app.engines.qwen3tts.plugin import Qwen3TTSBackend

    engine = Qwen3TTSBackend()
    engine._model = object()

    def broken(*_args, **_kwargs):
        raise RuntimeError("Offset increment outside graph capture encountered unexpectedly.")

    monkeypatch.setattr(torch, "multinomial", broken)
    monkeypatch.setattr(torch, "ones", lambda *a, **k: None)
    with pytest.raises(AppError) as exc:
        engine._check_random_generator()
    assert "QWEN_CUDA_GRAPHS=false" in exc.value.message and engine._model is None


def test_install_graphs_needs_a_cuda_model():
    predictor = TinyPredictor()
    model = types.SimpleNamespace(model=types.SimpleNamespace(
        talker=types.SimpleNamespace(code_predictor=predictor, config=types.SimpleNamespace(num_code_groups=5,
                                                                                            hidden_size=8))))
    assert fast_predictor.install_graphs(model) is False  # CPU weights (or no CUDA): no capture attempted


class _FakeCache:
    def __init__(self):
        self.resets = 0

    def reset(self):
        self.resets += 1


def _decoder(capacity=64, budget=10):
    """A GraphedTalkerDecoder without CUDA capture: the routing logic only."""
    decoder = fast_predictor.GraphedTalkerDecoder.__new__(fast_predictor.GraphedTalkerDecoder)
    decoder.capacity, decoder.budget = capacity, budget
    decoder.cache = _FakeCache()
    decoder.slots = torch.arange(capacity)
    decoder.embeds = torch.zeros(1, 1, 4)
    decoder.position_ids = torch.zeros(3, 1, 1, dtype=torch.long)
    decoder.cache_position = torch.zeros(1, dtype=torch.long)
    decoder.calls = []

    def original(**kwargs):
        decoder.calls.append(kwargs)
        return types.SimpleNamespace(past_key_values=kwargs["past_key_values"])

    decoder.original = original
    decoder.replays = 0

    def replay():
        decoder.replays += 1
        decoder.out.last_hidden_state.add_(1)

    decoder.graph = types.SimpleNamespace(replay=replay)
    decoder.out = types.SimpleNamespace(last_hidden_state=torch.zeros(1, 1, 4),
                                        hidden_states=(torch.zeros(1, 1, 4), torch.ones(1, 1, 4)))
    return decoder


def test_talker_prefill_uses_the_static_cache_only_when_it_fits():
    d = _decoder(capacity=64, budget=10)
    prompt = torch.randn(1, 20, 4)
    d.forward(inputs_embeds=prompt, attention_mask=torch.ones(1, 20, dtype=torch.long), position_ids=None,
              past_key_values="dynamic", use_cache=True, output_hidden_states=True, cache_position=torch.arange(20))
    call = d.calls[-1]
    assert call["past_key_values"] is d.cache and d.cache.resets == 1
    assert call["attention_mask"].shape == (1, 1, 20, 64)  # causal over the whole static cache
    assert bool(call["attention_mask"][0, 0, 3, 3]) and not bool(call["attention_mask"][0, 0, 3, 4])

    for kwargs in ({"attention_mask": torch.tensor([[0] + [1] * 19])},  # left padding
                   {"inputs_embeds": torch.randn(2, 20, 4)},  # batch of two
                   {"inputs_embeds": torch.randn(1, 60, 4)}):  # 60 + budget 10 does not fit in 64
        args = {"inputs_embeds": prompt, "attention_mask": torch.ones(1, 20, dtype=torch.long),
                "past_key_values": "dynamic", "use_cache": True, "cache_position": torch.arange(20), **kwargs}
        d.forward(**args)
        assert d.calls[-1]["past_key_values"] == "dynamic"  # original path, untouched cache


def test_talker_decode_replays_the_graph_and_clones_outputs():
    d = _decoder()
    step = {"inputs_embeds": torch.randn(1, 1, 4), "position_ids": torch.full((3, 1, 1), 21),
            "past_key_values": d.cache, "use_cache": True, "output_hidden_states": True,
            "cache_position": torch.tensor([21])}
    first = d.forward(**step)
    second = d.forward(**step)
    assert d.replays == 2 and not d.calls  # no eager forward at all
    assert int(d.cache_position) == 21 and torch.equal(d.position_ids, torch.full((3, 1, 1), 21))
    assert first.last_hidden_state.sum() == 4 and second.last_hidden_state.sum() == 8  # clones, not shared buffers
    assert first.hidden_states[1] is not d.out.hidden_states[1]

    d.forward(**{**step, "past_key_values": "dynamic"})  # a request that started on the original path stays there
    assert d.calls[-1]["past_key_values"] == "dynamic" and d.replays == 2


def test_install_talker_graphs_needs_a_cuda_model():
    talker = types.SimpleNamespace(model=torch.nn.Linear(2, 2), parameters=lambda: iter([torch.zeros(1)]))
    assert fast_predictor.install_talker_graphs(types.SimpleNamespace(model=types.SimpleNamespace(talker=talker))) \
        is False
