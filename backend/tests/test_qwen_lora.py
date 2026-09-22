"""Trainer pieces that do not need the real model: sequence layout, LoRA wrapping/merging, losses, saving, CLI."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf
import torch
from torch import nn

from app.training import qwen_lora as q


def config():
    talker = SimpleNamespace(codec_nothink_id=11, codec_think_bos_id=12, codec_think_eos_id=13, codec_pad_id=14,
                             codec_bos_id=15, codec_eos_token_id=16)
    return SimpleNamespace(talker_config=talker, tts_pad_token_id=1, tts_bos_token_id=2, tts_eos_token_id=3)


def test_collate_lays_out_text_then_codec_like_the_official_dataset():
    ids = torch.tensor([[100, 101, 102, 200, 201]])  # 3 chat-prefix tokens + 2 text tokens
    codes = torch.arange(3 * 16).reshape(3, 16) + 500
    batch = q.collate([(ids, codes), (ids[:, :4], codes[:2])], config())
    first = 8 + 5 - 1  # first codec frame
    row = batch["input_ids"][0]
    assert row[:3, 0].tolist() == [100, 101, 102] and row[3:7, 0].tolist() == [1] * 4 and row[7, 0] == 2
    assert row[8:10, 0].tolist() == [200, 201] and row[10, 0] == 3  # text, then its end
    assert row[3:8, 1].tolist() == [11, 12, 13, 0, 14]  # codec tags + the speaker slot (position 6)
    assert row[first - 1, 1] == 15 and row[first:first + 3, 1].tolist() == codes[:, 0].tolist()
    assert row[first + 3, 1] == 16  # codec end
    labels = batch["labels"][0]
    assert labels[first:first + 3].tolist() == codes[:, 0].tolist() and labels[first + 3] == 16
    assert (labels[:first] == -100).all()
    assert batch["codec_mask"][0, first:first + 3].all() and batch["codec_mask"][0].sum() == 3
    assert not batch["codec_emb_mask"][0, 6, 0]
    assert batch["attention"][0].sum() == first + 4 and batch["attention"][1].sum() == 8 + 4 - 1 + 3
    assert batch["input_ids"].shape == (2, 8 + 8, 2)  # longest item + 8


def test_text_ids_drop_the_next_turn_header():
    processor = lambda text, **_: {"input_ids": torch.arange(len(text.split()))}  # noqa: E731
    ids = q.text_ids(processor, "hola que tal")
    full = "<|im_start|>assistant\nhola que tal<|im_end|>\n<|im_start|>assistant\n"
    assert ids.dim() == 2 and ids.shape[1] == len(full.split()) - 5


class Layer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = nn.Module()
        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            setattr(self.self_attn, name, nn.Linear(8, 8, bias=False))
        self.mlp = nn.Module()
        self.mlp.gate_proj, self.mlp.up_proj, self.mlp.down_proj = (nn.Linear(8, 8, bias=False) for _ in range(3))


def talker():
    t = nn.Module()
    t.model = nn.Module()
    t.model.layers = nn.ModuleList([Layer(), Layer()])
    t.code_predictor = nn.Module()
    t.code_predictor.model = nn.Module()
    t.code_predictor.model.layers = nn.ModuleList([Layer()])
    return t


def test_lora_starts_as_identity_trains_only_adapters_and_merges_exactly():
    torch.manual_seed(0)
    t = talker()
    x = torch.randn(3, 8)
    before = t.model.layers[0].self_attn.q_proj(x).detach()
    wrapped = q.add_lora(t, rank=2, alpha=4.0)
    assert len(wrapped) == 3 * 7  # 7 projections in each of the 3 layers
    layer = t.model.layers[0].self_attn.q_proj
    assert torch.allclose(layer(x), before)  # B starts at zero: no change until trained
    with torch.no_grad():
        layer.lora_b.normal_()
    tuned = layer(x).detach()
    assert not torch.allclose(tuned, before)
    q.merge_lora(wrapped, scale=1.0)
    merged = t.model.layers[0].self_attn.q_proj
    assert isinstance(merged, nn.Linear) and torch.allclose(merged(x), tuned, atol=1e-5)


def test_sub_talker_loss_uses_aligned_labels():
    class Predictor(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embeds = nn.ModuleList([nn.Embedding(32, 4) for _ in range(15)])

        def get_input_embeddings(self):
            return self.embeds

        def forward_finetune(self, inputs_embeds):
            self.seen = inputs_embeds.shape
            logits = torch.zeros(inputs_embeds.shape[0], 15, 32)
            logits[:, torch.arange(15), torch.arange(1, 16)] = 50.0  # predicts code i for codebook i
            return SimpleNamespace(logits=logits)

    t = SimpleNamespace(code_predictor=Predictor(), get_input_embeddings=lambda: nn.Embedding(32, 4))
    codes = torch.arange(16).repeat(5, 1)  # codebook i holds value i
    loss = q.sub_talker_loss(t, codes, torch.zeros(5, 4))
    assert t.code_predictor.seen == (5, 16, 4) and loss.item() < 1e-3
    shifted = q.sub_talker_loss(t, torch.roll(codes, 1, dims=1), torch.zeros(5, 4))
    assert shifted.item() > 10


def test_save_model_writes_a_custom_voice_folder(tmp_path):
    base = tmp_path / "base"
    (base / "speech_tokenizer").mkdir(parents=True)
    (base / "config.json").write_text(json.dumps({"tts_model_type": "base", "talker_config": {"spk_id": {}}}))
    (base / "model.safetensors").write_text("old weights")
    (base / "README.md").write_text("x")
    (base / "speech_tokenizer" / "model.safetensors").write_text("tokenizer")

    model = nn.Module()
    model.talker = nn.Module()
    model.talker.model = nn.Module()
    model.talker.model.codec_embedding = nn.Embedding(3001, 4)
    model.speaker_encoder = nn.Linear(2, 2)
    spec = q.TrainSpec(base_model_path=base, output_dir=tmp_path / "out", speaker_name="voicelab",
                       clips=[], reference=q.ClipSpec(audio_path=tmp_path / "r.wav", text="x"))
    out = q.save_model(spec, model, json.loads((base / "config.json").read_text()), torch.ones(4))

    cfg = json.loads((out / "config.json").read_text(encoding="utf-8"))
    assert cfg["tts_model_type"] == "custom_voice" and cfg["talker_config"]["spk_id"] == {"voicelab": 3000}
    assert cfg["talker_config"]["spk_is_dialect"] == {"voicelab": False}
    assert (out / "speech_tokenizer" / "model.safetensors").read_text() == "tokenizer"  # kept
    assert not (out / "README.md").exists() and not out.with_name("out.partial").exists()
    from safetensors.torch import load_file

    state = load_file(str(out / "model.safetensors"))
    assert torch.equal(state["talker.model.codec_embedding.weight"][3000], torch.ones(4))
    assert not any(k.startswith("speaker_encoder") for k in state)


def test_load_clip_cuts_mixes_and_resamples(tmp_path):
    path = tmp_path / "a.wav"
    stereo = np.stack([np.ones(48_000), np.zeros(48_000)], axis=1).astype(np.float32)
    sf.write(path, stereo, 48_000)
    audio = q.load_clip(q.ClipSpec(audio_path=path, text="x", start_s=0.25, end_s=0.75))
    assert audio.dtype == np.float32 and audio.size == 12_000 and abs(float(audio[6000]) - 0.5) < 0.01


def test_main_reports_errors_as_json(tmp_path, monkeypatch, capsys):
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps({"base_model_path": str(tmp_path), "output_dir": str(tmp_path / "o"),
                                "speaker_name": "v", "clips": [], "reference": {"audio_path": "r.wav", "text": "x"}}))

    def boom(_spec):
        raise torch.cuda.OutOfMemoryError("CUDA out of memory")

    monkeypatch.setattr(q, "train", boom)
    assert q.main(["x", str(spec)]) == 1
    event = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert event == {"event": "error", "code": "OUT_OF_MEMORY", "detail": "OutOfMemoryError: CUDA out of memory"}

    monkeypatch.setattr(q, "train", lambda _spec: (_ for _ in ()).throw(ValueError("bad")))
    assert q.main(["x", str(spec)]) == 1
    assert json.loads(capsys.readouterr().out.strip())["code"] == "TRAINING_FAILED"
    monkeypatch.setattr(q, "train", lambda _spec: tmp_path)
    assert q.main(["x", str(spec)]) == 0


@pytest.mark.parametrize("field", ["epochs", "batch_size"])
def test_spec_limits(tmp_path, field):
    with pytest.raises(ValueError):
        q.TrainSpec(base_model_path=tmp_path, output_dir=tmp_path, speaker_name="v", clips=[],
                    reference=q.ClipSpec(audio_path=tmp_path, text="x"), **{field: 0})
