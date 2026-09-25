"""Fine-tuning F5-TTS on a voice: the trainer's own pieces and the run through the API (no GPU, no real weights)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch
from sqlmodel import Session, select
from torch import nn

from app.core.database import get_engine
from app.models.entities import CustomCheckpoint, TrainingRun
from app.schemas.training import TrainingCreate
from app.services import training_service
from app.services.training_service import _trainer_command
from app.training import evaluation, f5_lora
from app.training.qwen_lora import ClipSpec
from tests.test_training import _profile, _wait, trainer  # noqa: F401  (fixtures and helpers)

FAKE_F5_TRAINER = r'''
import json, os, sys
spec = json.load(open(sys.argv[1], encoding="utf-8"))
def emit(event, **data):
    print(json.dumps({"event": event, **data}), flush=True)
emit("stage", stage="load", message="Cargando")
emit("stage", stage="train", message="Entrenando", clips=len(spec["clips"]), epochs=spec["epochs"])
history = []
for e in range(spec["epochs"]):
    emit("progress", epoch=e + 1, epochs=spec["epochs"], step=e + 1, total_steps=spec["epochs"], loss=0.5, eta_s=1)
    history.append({"epoch": e + 1, "loss": round(0.5 - e * 0.01, 3)})
    emit("epoch", **history[-1])
emit("stage", stage="save", message="Guardando")
out = spec["output_path"]
os.makedirs(os.path.dirname(out), exist_ok=True)
open(out, "wb").write(b"fake-safetensors")
emit("done", output=out, history=history)
'''


# ---------------------------------------------------------------------------- what the trainer does on its own
class FakeAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.to_q = nn.Linear(8, 8)
        self.to_k = nn.Linear(8, 8)
        self.to_v = nn.Linear(8, 8)
        self.to_out = nn.ModuleList([nn.Linear(8, 8), nn.Dropout(0.0)])


class FakeFeedForward(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.ff = nn.Sequential(nn.Sequential(nn.Linear(8, 16), nn.GELU()), nn.Dropout(0.0), nn.Linear(16, 8))


class FakeBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attn = FakeAttention()
        self.ff = FakeFeedForward()
        self.norm = nn.LayerNorm(8)  # never wrapped: it is not a Linear


class FakeTransformer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.transformer_blocks = nn.ModuleList([FakeBlock(), FakeBlock()])


class FakeCFM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.transformer = FakeTransformer()


def test_lora_wraps_every_projection_of_each_block_and_nothing_else():
    model = FakeCFM()
    wrapped = f5_lora.add_lora(model, rank=4, alpha=4.0)

    assert len(wrapped) == 2 * (4 + 2)  # per block: q, k, v, out + the two feed-forward layers
    assert isinstance(model.transformer.transformer_blocks[0].attn.to_q, nn.Module)
    assert hasattr(model.transformer.transformer_blocks[0].attn.to_q, "lora_a")
    assert hasattr(model.transformer.transformer_blocks[0].attn.to_out[0], "lora_a")  # inside a ModuleList
    assert hasattr(model.transformer.transformer_blocks[0].ff.ff[0][0], "lora_a")  # inside nested Sequentials
    assert isinstance(model.transformer.transformer_blocks[0].norm, nn.LayerNorm)


def test_the_adapters_start_neutral_and_the_merge_keeps_what_they_learned():
    torch.manual_seed(0)
    model = FakeCFM()
    layer = model.transformer.transformer_blocks[0].attn
    x = torch.randn(1, 8)
    before = layer.to_q(x).clone()

    wrapped = f5_lora.add_lora(model, rank=4, alpha=4.0)
    assert torch.allclose(layer.to_q(x), before, atol=1e-6)  # lora_b starts at zero: same model as before

    with torch.no_grad():  # something learned
        layer.to_q.lora_b.add_(0.1)
    trained = layer.to_q(x).clone()
    f5_lora.merge_lora(wrapped, 1.0)
    assert isinstance(layer.to_q, nn.Linear)  # back to a plain Linear, ready to save
    assert torch.allclose(layer.to_q(x), trained, atol=1e-5)


def test_only_the_adapters_are_trained():
    model = FakeCFM()
    model.requires_grad_(False)
    f5_lora.add_lora(model, rank=4, alpha=4.0)
    trainable = {name for name, p in model.named_parameters() if p.requires_grad}
    assert trainable and all(n.endswith(("lora_a", "lora_b")) for n in trainable)


def test_clips_are_grouped_by_length_and_padded_to_the_longest():
    items = [(torch.zeros(n, 100), ["a"], n) for n in (10, 40, 20)]
    groups = f5_lora.batches(items, 2, __import__("random").Random(0))
    assert sorted(sorted(g) for g in groups) == [[0, 2], [1]]  # 10 and 20 travel together

    mel, text, lens = f5_lora.collate(items, [0, 1], "cpu")
    assert mel.shape == (2, 40, 100) and lens.tolist() == [10, 40] and text == [["a"], ["a"]]


def test_clips_that_are_too_long_or_too_short_are_left_out(tmp_path, monkeypatch):
    class StubMel:
        def __call__(self, wave):
            return torch.zeros(1, 100, wave.shape[-1] // f5_lora.HOP_LENGTH)

    model = type("M", (), {"mel_spec": StubMel()})()
    monkeypatch.setattr(f5_lora, "load_clip", lambda clip: np.zeros(int(clip.end_s * f5_lora.SAMPLE_RATE), np.float32))
    spec = f5_lora.TrainSpec(base_ckpt="x", vocab_path="v", output_path=str(tmp_path / "o.safetensors"), clips=[
        ClipSpec(audio_path="a.wav", text="Una frase.", start_s=0.0, end_s=4.0),
        ClipSpec(audio_path="a.wav", text="Muy corta.", start_s=0.0, end_s=0.1),
        ClipSpec(audio_path="a.wav", text="Larguísima.", start_s=0.0, end_s=40.0),
    ])
    items = f5_lora.prepare(spec, model, "cpu")
    assert len(items) == 1 and items[0][2] == 4 * f5_lora.SAMPLE_RATE // f5_lora.HOP_LENGTH


def test_the_text_is_converted_the_same_way_inference_converts_it(tmp_path, monkeypatch):
    """The prepared datasets store pinyin-converted text: training on raw characters would not match inference."""
    from f5_tts.model.utils import convert_char_to_pinyin

    class StubMel:
        def __call__(self, wave):
            return torch.zeros(1, 100, 50)

    monkeypatch.setattr(f5_lora, "load_clip", lambda clip: np.zeros(f5_lora.SAMPLE_RATE, np.float32))
    spec = f5_lora.TrainSpec(base_ckpt="x", vocab_path="v", output_path=str(tmp_path / "o.safetensors"),
                             clips=[ClipSpec(audio_path="a.wav", text="Hola, ¿qué tal?", start_s=0.0, end_s=1.0)])
    items = f5_lora.prepare(spec, type("M", (), {"mel_spec": StubMel()})(), "cpu")
    assert items[0][1] == convert_char_to_pinyin(["Hola, ¿qué tal?"])[0]


def test_saving_writes_one_checkpoint_file_atomically(tmp_path):
    from safetensors.torch import load_file

    model = FakeCFM()
    wrapped = f5_lora.add_lora(model, rank=4, alpha=4.0)
    out = tmp_path / "voz.safetensors"
    spec = f5_lora.TrainSpec(base_ckpt="x", vocab_path="v", output_path=str(out), clips=[])

    assert f5_lora.save_model(spec, model, wrapped) == out
    state = load_file(str(out))
    assert not list(tmp_path.glob("*.partial"))
    assert not any("lora" in key for key in state)  # merged: a plain checkpoint, like the official ones
    assert "transformer.transformer_blocks.0.attn.to_q.weight" in state


# ---------------------------------------------------------------------------- through the API
@pytest.fixture
def f5_trainer(tmp_path, monkeypatch, trainer):  # noqa: F811
    script = tmp_path / "fake_f5_trainer.py"
    script.write_text(FAKE_F5_TRAINER, encoding="utf-8")
    real = training_service._trainer_command
    monkeypatch.setattr(training_service, "_trainer_command",
                        lambda spec, engine=None: [sys.executable, str(script), str(spec)]
                        if engine == "f5tts" else real(spec, engine))
    from app.engines.f5tts.plugin import F5TTSBackend

    monkeypatch.setattr(F5TTSBackend, "is_installed", lambda self: True)
    monkeypatch.setattr(F5TTSBackend, "weights_installed", lambda self, v: True)
    monkeypatch.setattr(training_service, "_f5_checkpoint", lambda models, variant: str(tmp_path / "base.safetensors"))
    monkeypatch.setattr(training_service, "_f5_vocab", lambda: str(tmp_path / "vocab.txt"))
    return trainer


def test_only_the_base_checkpoints_of_that_engine_can_be_trained():
    assert TrainingCreate(profile_id="p", engine="f5tts", base_variant="F5TTS_v1_Base").base_variant == "F5TTS_v1_Base"
    with pytest.raises(ValueError, match="F5TTS_v1_Base"):
        TrainingCreate(profile_id="p", engine="f5tts", base_variant="base-1.7b")
    with pytest.raises(ValueError, match="base-1.7b"):
        TrainingCreate(profile_id="p", engine="qwen3tts", base_variant="F5TTS_v1_Base")


def test_each_engine_runs_its_own_trainer():
    assert _trainer_command(Path("spec.json"), "f5tts")[-2] == "app.training.f5_lora"
    assert _trainer_command(Path("spec.json"), "qwen3tts")[-2] == "app.training.qwen_lora"


def test_a_trained_f5_voice_is_published_as_a_checkpoint_file(client, f5_trainer, settings):
    profile_id = _profile(settings, name="Bruno")
    accepted = client.post("/api/training", json={"profile_id": profile_id, "engine": "f5tts",
                                                  "base_variant": "F5TTS_v1_Base", "epochs": 2})
    assert accepted.status_code == 202, accepted.text
    done = _wait(client, accepted.json()["id"])

    assert done["status"] == "completed", done.get("error_detail")
    assert done["engine"] == "f5tts" and done["variant"].startswith("custom:")
    assert done["history"] == [{"epoch": 1, "loss": 0.5}, {"epoch": 2, "loss": 0.49}]
    with Session(get_engine()) as session:
        run = session.exec(select(TrainingRun)).one()
        checkpoint = session.get(CustomCheckpoint, run.checkpoint_id)
        assert checkpoint.engine == "f5tts" and checkpoint.base_variant == "F5TTS_v1_Base"
        assert Path(run.output_path).suffix == ".safetensors" and Path(run.output_path).is_file()
        assert checkpoint.local_path == run.output_path
        assert "clonando desde una referencia" in checkpoint.notes  # F5 is not a speaker-embedded model

    spec = json.loads((Path(settings.data_dir) / "training" / run.id / "spec.json").read_text(encoding="utf-8"))
    assert spec["arch"] == "F5TTS_v1_Base" and spec["output_path"].endswith(".safetensors")
    assert spec["clips"] and "reference" not in spec  # F5 trains on the clips alone


def test_deleting_the_run_removes_the_checkpoint_file(client, f5_trainer, settings):
    profile_id = _profile(settings, name="Nadia")
    run_id = client.post("/api/training", json={"profile_id": profile_id, "engine": "f5tts",
                                                "base_variant": "F5TTS_v1_Base", "epochs": 1}).json()["id"]
    _wait(client, run_id)
    with Session(get_engine()) as session:
        path = Path(session.get(TrainingRun, run_id).output_path)
    assert path.is_file()

    assert client.delete(f"/api/training/{run_id}").status_code == 204
    assert not path.exists()


def test_the_comparison_clones_from_a_reference_when_the_engine_needs_one(monkeypatch, settings, tmp_path):
    """A fine-tuned F5 is still a cloning model: both systems must get the same reference."""
    from tests.test_training_evaluation import SR, FakeEngine, FakeModels, _patch, tone

    _patch(monkeypatch, encoder=False, asr=False)
    path = tmp_path / "real.wav"
    sf.write(path, np.concatenate([tone(220.0, 4.0), tone(220.0, 4.0)]), SR)
    clip = {"audio_path": str(path), "text": "hola a todos", "start_s": 0.0, "end_s": 4.0}
    run = TrainingRun(id="r-f5", profile_id="p", engine="f5tts", base_variant="F5TTS_v1_Base", name="x", dataset={
        "held_out": [clip, {**clip, "start_s": 4.0, "end_s": 8.0, "text": "hola amigos"}],
        "reference": {**clip, "end_s": 3.0}})
    models = FakeModels()
    models.engines = {"custom:v": FakeEngine(220.0, needs_reference=True),
                      "F5TTS_v1_Base": FakeEngine(260.0, needs_reference=True)}
    ctx = type("Ctx", (), {"cancel": type("T", (), {"cancelled": False})()})()

    with Session(get_engine()) as session:
        result = evaluation.compare(session, models, run, "custom:v", ctx)

    assert models.engines["custom:v"].prepared, "the trained F5 must receive the reference too"
    assert models.engines["F5TTS_v1_Base"].prepared
    assert models.used == ["custom:v", "F5TTS_v1_Base"]
    assert result["held_out"] == 2
    assert (evaluation.samples_dir(run.id) / "0_trained.wav").exists()


@pytest.mark.parametrize("engine", ["qwen3tts", "f5tts"])
def test_both_engines_use_the_same_recordings(settings, engine):
    """The dataset does not depend on the engine: the same sentences train either of them."""
    from app.services.training_service import collect

    profile_id = _profile(settings, name=f"Voz{engine[:3]}")
    with Session(get_engine()) as session:
        groups = collect(session, settings, profile_id)
    assert groups and groups[0].clips


def test_the_training_loop_runs_on_the_cpu_with_a_tiny_model(tmp_path, monkeypatch):
    """The loop itself: batches, accumulation, the learning-rate schedule, the events and the saved file."""
    class TinyCFM(FakeCFM):
        def __init__(self) -> None:
            super().__init__()
            self.head = nn.Linear(100, 1)

        def mel_spec(self, wave):  # noqa: ANN001, ANN202
            return torch.zeros(1, 100, max(1, wave.shape[-1] // f5_lora.HOP_LENGTH))

        def forward(self, mel, text, lens=None):  # noqa: ANN001, ANN202
            x = mel
            for block in self.transformer.transformer_blocks:
                x = block.attn.to_q(x[..., :8]) if x.shape[-1] >= 8 else x
            return self.head(mel).mean() + x.mean(), None, None

    events: list[dict] = []
    monkeypatch.setattr(f5_lora, "build_model", lambda spec, device: TinyCFM())
    monkeypatch.setattr(f5_lora, "emit", lambda event, **data: events.append({"event": event, **data}))
    monkeypatch.setattr(f5_lora, "load_clip", lambda clip: np.zeros(int(clip.end_s * f5_lora.SAMPLE_RATE), np.float32))
    out = tmp_path / "voz.safetensors"
    spec = f5_lora.TrainSpec(base_ckpt="x", vocab_path="v", output_path=str(out), epochs=2, batch_size=1,
                             grad_accumulation=1, lora_rank=2, clips=[
                                 ClipSpec(audio_path="a.wav", text="Una frase.", start_s=0.0, end_s=2.0),
                                 ClipSpec(audio_path="a.wav", text="Otra frase.", start_s=0.0, end_s=3.0)])

    output, history = f5_lora.train(spec)

    assert output == out and out.is_file()
    assert [h["epoch"] for h in history] == [1, 2] and all("loss" in h for h in history)
    stages = [e.get("stage") for e in events if e["event"] == "stage"]
    assert stages == ["load", "train", "save"]
    progress = [e for e in events if e["event"] == "progress"]
    assert progress and progress[-1]["step"] == progress[-1]["total_steps"]  # every step was reported
