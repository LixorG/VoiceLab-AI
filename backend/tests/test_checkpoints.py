"""Custom checkpoints: validation, listing as a variant, weight checks, loading and removal (no GPU)."""

from __future__ import annotations

import pytest

from app.engines.custom import get_custom_variants
from app.engines.f5tts.plugin import F5TTSBackend
from app.engines.flow_matching import VOCOS_FILES, VOCOS_REPO
from tests.fakes import FakeF5TTS, FakeHub, install_module

SPANISH = {
    "name": "F5 Español (comunidad)",
    "base_variant": "F5TTS_v1_Base",
    "repo_id": "jpgallegoar/F5-Spanish",
    "ckpt_file": "model_1200000.safetensors",
    "languages": ["es"],
    "notes": "Fine-tune en español.",
}


@pytest.fixture(autouse=True)
def _clean_store():
    get_custom_variants().clear()
    yield
    get_custom_variants().clear()


def test_added_checkpoint_becomes_a_variant(client):
    created = client.post("/api/models/f5tts/checkpoints", json=SPANISH)
    assert created.status_code == 201, created.text
    body = created.json()
    assert body["variant"] == "custom:f5-espanol-comunidad" and body["languages"] == ["es"]

    engines = {e["id"]: e for e in client.get("/api/models").json()}
    variants = {v["id"]: v for v in engines["f5tts"]["variants"]}
    assert "custom:f5-espanol-comunidad" in variants
    custom = variants["custom:f5-espanol-comunidad"]
    assert custom["source"] == "custom" and custom["base_variant"] == "F5TTS_v1_Base" and custom["languages"] == ["es"]
    assert engines["f5tts"]["default_variant"] == "F5TTS_v1_Base"  # the official one stays the default

    caps = client.get("/api/models/f5tts/capabilities?variant=custom:f5-espanol-comunidad").json()
    assert caps["languages"] == ["es"]
    assert any("personalizado" in note.lower() for note in caps["notes"])

    listed = client.get("/api/models/checkpoints?engine=f5tts").json()
    assert [c["name"] for c in listed] == ["F5 Español (comunidad)"]
    assert client.get("/api/models/checkpoints?engine=qwen3tts").json() == []


def test_validation_rejects_unusable_checkpoints(client, tmp_path):
    def post(**over):
        return client.post("/api/models/f5tts/checkpoints", json={**SPANISH, **over})

    assert post(base_variant="no-existe").json()["error_code"] == "VALIDATION_ERROR"
    assert post(ckpt_file="modelo.bin").json()["error_code"] == "VALIDATION_ERROR"  # wrong extension
    assert post(local_path=str(tmp_path / "x.safetensors")).status_code == 422  # both sources at once
    assert post(repo_id=None, ckpt_file=None).status_code == 422  # no source at all
    assert post(repo_id=None, ckpt_file=None, local_path=str(tmp_path / "falta.safetensors")).json()["details"] == \
        {"ruta": str(tmp_path / "falta.safetensors")}

    weights = tmp_path / "modelo.safetensors"
    weights.write_bytes(b"0")
    assert post(repo_id=None, ckpt_file=None, local_path=str(weights),
                vocab_path=str(tmp_path / "falta.txt")).status_code == 422

    qwen = client.post("/api/models/qwen3tts/checkpoints", json=SPANISH)
    assert qwen.status_code == 422 and "no admite checkpoints propios" in qwen.json()["message"]


def test_duplicate_names_get_their_own_variant(client):
    first = client.post("/api/models/f5tts/checkpoints", json=SPANISH).json()
    second = client.post("/api/models/f5tts/checkpoints", json=SPANISH).json()
    assert first["variant"] == "custom:f5-espanol-comunidad" and second["variant"] == "custom:f5-espanol-comunidad-2"


def test_weights_status_download_and_load(client, monkeypatch, tmp_path):
    created = client.post("/api/models/f5tts/checkpoints", json=SPANISH).json()
    variant = created["variant"]
    hub = FakeHub(root=tmp_path / "hub")
    hub.install(monkeypatch)
    install_module(monkeypatch, "f5_tts.api", F5TTS=FakeF5TTS)
    engine = F5TTSBackend()
    monkeypatch.setattr(engine, "is_installed", lambda: True)

    assert engine.weights_installed(variant) is False
    engine.download_weights(variant)
    assert (SPANISH["repo_id"], SPANISH["ckpt_file"]) in hub.downloads
    assert (VOCOS_REPO, VOCOS_FILES[0]) in hub.downloads
    assert engine.weights_installed(variant) is True

    FakeF5TTS.instances.clear()
    engine.load(variant, "cpu", "auto")
    model = FakeF5TTS.instances[-1]
    assert model.model == "F5TTS_v1_Base"  # architecture stays an f5-tts config name
    assert model.ckpt_file.endswith(SPANISH["ckpt_file"]) and model.vocab_file == ""
    assert engine.loaded_variant == variant
    engine.unload()


def test_local_checkpoint_needs_its_file(client, monkeypatch, tmp_path):
    weights = tmp_path / "modelo.safetensors"
    weights.write_bytes(b"0")
    vocab = tmp_path / "vocab.txt"
    vocab.write_text("a\n", encoding="utf-8")
    body = {**SPANISH, "name": "Mi entrenamiento", "repo_id": None, "ckpt_file": None,
            "local_path": str(weights), "vocab_path": str(vocab)}
    variant = client.post("/api/models/f5tts/checkpoints", json=body).json()["variant"]

    hub = FakeHub(root=tmp_path / "hub", cached={(VOCOS_REPO, f) for f in VOCOS_FILES})
    hub.install(monkeypatch)
    install_module(monkeypatch, "f5_tts.api", F5TTS=FakeF5TTS)
    engine = F5TTSBackend()
    monkeypatch.setattr(engine, "is_installed", lambda: True)

    assert engine.weights_installed(variant) is True
    FakeF5TTS.instances.clear()
    engine.load(variant, "cpu", "auto")
    model = FakeF5TTS.instances[-1]
    assert model.ckpt_file == str(weights) and model.vocab_file == str(vocab)
    engine.unload()

    weights.unlink()  # the user moved or deleted the file
    assert engine.weights_installed(variant) is False


def test_delete_removes_the_variant(client):
    created = client.post("/api/models/f5tts/checkpoints", json=SPANISH).json()
    assert client.delete(f"/api/models/checkpoints/{created['id']}").status_code == 204
    variants = {v["id"] for v in client.get("/api/models").json()[0]["variants"]}
    assert created["variant"] not in variants
    assert client.delete(f"/api/models/checkpoints/{created['id']}").json()["error_code"] == "CHECKPOINT_NOT_FOUND"
    assert client.get(f"/api/models/f5tts?variant={created['variant']}").json()["error_code"] == "ENGINE_NOT_FOUND"
