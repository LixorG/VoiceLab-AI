from __future__ import annotations

import pytest

from app.audio.segments import snap_segment, word_boundaries
from app.core.errors import AppError, ErrorCode
from app.engines.base import (
    CancelToken,
    ControlSource,
    EngineRequest,
    GenericControl,
    ParameterSpec,
    ParameterTooltip,
    TTSBackend,
    _coerce,
)
from app.engines.mock.plugin import MockBackend
from app.engines.registry import EngineRegistry, get_engine_registry

REAL_ENGINES = ["f5tts", "e2tts", "qwen3tts"]


@pytest.fixture(scope="module")
def registry() -> EngineRegistry:
    return EngineRegistry().discover(include_dev=True)


def all_variants(registry):
    for engine in registry.all():
        for v in engine.variants():
            yield engine, v.id


# ---------------- registry / plugins ----------------
def test_discovery_finds_builtin_engines_in_order(registry):
    ids = [e.id for e in registry.all()]
    assert ids[:3] == REAL_ENGINES and "mock" in ids
    assert "mock" not in [e.id for e in EngineRegistry().discover(include_dev=False).all()]


def test_unknown_engine_and_variant(registry):
    with pytest.raises(AppError) as exc:
        registry.get("xtts")
    assert exc.value.code is ErrorCode.ENGINE_NOT_FOUND
    with pytest.raises(AppError):
        registry.get("f5tts").variant("nope")


def test_duplicate_registration_rejected():
    reg = EngineRegistry()
    reg.register(MockBackend())
    with pytest.raises(ValueError):
        reg.register(MockBackend())


def test_engines_do_not_import_model_packages(registry):
    import sys

    assert "f5_tts" not in sys.modules and "qwen_tts" not in sys.modules


# ---------------- schema quality (applies to every engine/variant) ----------------
def test_every_parameter_is_documented_in_spanish(registry):
    for engine, variant in all_variants(registry):
        if engine.id == "mock":
            continue
        specs = engine.get_parameters_schema(variant)
        ids = [s.id for s in specs]
        assert len(ids) == len(set(ids)), f"{engine.id}:{variant} duplicate ids"
        for s in specs:
            tip = s.tooltip
            assert all(len(x) >= 3 for x in (s.label, tip.what, tip.effect, tip.cost, tip.typical)), s.id
            assert "mejor" not in tip.effect.lower() or "no garantiza" in tip.effect.lower(), s.id
            if s.type in ("slider", "number"):
                assert s.min is not None and s.max is not None and s.min < s.max
                if s.default is not None:
                    assert s.min <= s.default <= s.max, (engine.id, s.id)
            for preset_value in s.presets.values():
                _coerce(s, preset_value)
        required = {s.id: "texto" for s in specs if s.type == "text" and not s.nullable and s.default is None}
        engine.validate_parameters(required, variant)  # defaults + required text are valid


def test_non_native_controls_explain_why(registry):
    for engine, variant in all_variants(registry):
        caps = engine.capabilities(variant)
        params = {s.id for s in engine.get_parameters_schema(variant)}
        assert set(GenericControl) <= set(caps.controls), (engine.id, variant)
        for control, cap in caps.controls.items():
            if cap.source is ControlSource.NATIVE:
                assert cap.parameter in params, (engine.id, variant, control)
            elif cap.source is not ControlSource.NATIVE:
                assert cap.reason, (engine.id, variant, control)


# ---------------- engine facts (docs/model-analysis.md) ----------------
def test_f5_parameters_match_library_defaults(registry):
    specs = {s.id: s for s in registry.get("f5tts").get_parameters_schema()}
    assert {k: (specs[k].maps_to, specs[k].default) for k in specs} == {
        "speed": ("speed", 1.0), "seed": ("seed", None), "nfe_steps": ("nfe_step", 32),
        "cfg_strength": ("cfg_strength", 2.0), "sway_sampling_coef": ("sway_sampling_coef", -1.0),
        "cross_fade_duration": ("cross_fade_duration", 0.15), "remove_silence": ("remove_silence", False),
        "fix_duration": ("fix_duration", None), "target_rms": ("target_rms", 0.1),
    }
    assert specs["nfe_steps"].label == "Pasos de inferencia"
    assert specs["nfe_steps"].tooltip.what.startswith("Controla el número de pasos")
    assert registry.get("f5tts").resolve_preset("fast")["nfe_steps"] == 16


def test_e2_shares_f5_schema_but_is_separate_engine(registry):
    f5, e2 = registry.get("f5tts"), registry.get("e2tts")
    assert [s.id for s in f5.get_parameters_schema()] == [s.id for s in e2.get_parameters_schema()]
    assert e2.variants()[0].id == "E2TTS_Base" and e2.implementation_phase is None
    assert e2.checkpoints["E2TTS_Base"][0] == "SWivid/E2-TTS"  # E2 weights live in their own repo
    assert "comunitaria" in e2.description


def test_qwen_clone_has_no_style_instruction(registry):
    qwen = registry.get("qwen3tts")
    specs = {s.id: s for s in qwen.get_parameters_schema("base-1.7b")}
    assert "instruct" not in specs and "cfg_strength" not in specs and "speed" not in specs
    assert specs["clone_mode"].maps_to == "x_vector_only_mode"
    assert (specs["temperature"].default, specs["top_k"].default, specs["top_p"].default,
            specs["repetition_penalty"].default, specs["max_new_tokens"].default) == (0.9, 50, 1.0, 1.05, 8192)
    assert specs["language"].default == "Auto"
    assert {o.value for o in specs["language"].options} >= {"Spanish", "English", "Chinese", "Auto"}
    caps = qwen.capabilities("base-1.7b")
    assert caps.controls[GenericControl.INSTRUCTION].source is ControlSource.UNAVAILABLE
    assert caps.controls[GenericControl.EMOTION].source is not ControlSource.INSTRUCTION
    assert caps.controls[GenericControl.SPEED].source is ControlSource.DSP
    assert caps.requires_reference_audio and not caps.deterministic_seed and caps.reports_progress
    assert caps.max_chars_per_call == 300  # long texts are generated sentence by sentence
    assert qwen.requires_reference_text({"clone_mode": "icl"}, "base-1.7b") is True
    assert qwen.requires_reference_text({"clone_mode": "x_vector"}, "base-1.7b") is False
    assert registry.get("f5tts").requires_reference_text({"clone_mode": "x_vector"}) is True


def test_qwen_custom_voice_and_design_use_instructions(registry):
    qwen = registry.get("qwen3tts")
    cv = {s.id: s for s in qwen.get_parameters_schema("custom-voice-1.7b")}
    assert cv["instruct"].nullable and cv["speaker"].dynamic_options
    assert {o.value for o in cv["speaker"].options} >= {"Vivian", "Ryan", "Ono_Anna"}
    caps = qwen.capabilities("custom-voice-1.7b")
    assert caps.controls[GenericControl.INSTRUCTION].source is ControlSource.NATIVE
    assert caps.controls[GenericControl.EMOTION].source is ControlSource.INSTRUCTION
    assert not caps.requires_reference_audio
    with pytest.raises(AppError) as exc:  # voice design requires a description
        qwen.validate_parameters({}, "voice-design-1.7b")
    assert exc.value.details[0]["parametro"] == "instruct"


# ---------------- validation ----------------
def _spec(**kw) -> ParameterSpec:
    tip = ParameterTooltip(what="x", effect="x", cost="x", typical="x")
    return ParameterSpec(id="p", label="P", tooltip=tip, **kw)


@pytest.mark.parametrize(("spec", "raw", "expected"), [
    (dict(type="slider", value_type="int", min=4, max=64), "32", 32),
    (dict(type="slider", value_type="float", min=0, max=5), 2, 2.0),
    (dict(type="toggle", value_type="bool"), "true", True),
    (dict(type="select", value_type="str", options=[{"value": "a", "label": "A"}]), "a", "a"),
    (dict(type="select", value_type="str", dynamic_options=True, options=[{"value": "a", "label": "A"}]), "z", "z"),
    (dict(type="number", value_type="float", nullable=True, min=1, max=3), None, None),
    (dict(type="seed", value_type="int"), None, None),
    (dict(type="seed", value_type="int"), "123", 123),
])
def test_coerce_valid(spec, raw, expected):
    assert _coerce(_spec(**spec), raw) == expected


@pytest.mark.parametrize(("spec", "raw", "message"), [
    (dict(type="slider", value_type="int", min=4, max=64), 65, "máximo"),
    (dict(type="slider", value_type="int", min=4, max=64), 3.5, "entero"),
    (dict(type="slider", value_type="float", min=0, max=5), "abc", "número"),
    (dict(type="slider", value_type="float", min=0, max=5), float("nan"), "finito"),
    (dict(type="slider", value_type="float", min=0, max=5), True, "número"),
    (dict(type="toggle", value_type="bool"), "sí", "verdadero"),
    (dict(type="select", value_type="str", options=[{"value": "a", "label": "A"}]), "b", "Opción"),
    (dict(type="seed", value_type="int"), -1, "entre"),
    (dict(type="text", value_type="str", max_length=3), "abcd", "caracteres"),
    (dict(type="text", value_type="str"), None, "obligatorio"),
])
def test_coerce_invalid(spec, raw, message):
    with pytest.raises(ValueError) as exc:
        _coerce(_spec(**spec), raw)
    assert message.lower() in str(exc.value).lower()


def test_validate_rejects_unknown_parameters(registry):
    with pytest.raises(AppError) as exc:
        registry.get("f5tts").validate_parameters({"temperature": 0.7, "nfe_steps": 100})
    details = {d["parametro"]: d["error"] for d in exc.value.details}
    assert "temperature" in details and "nfe_steps" in details
    assert exc.value.code is ErrorCode.PARAMETER_ERROR


def test_unimplemented_generation_is_explicit(registry):
    class Planned(MockBackend):
        id = "planned"
        implementation_phase = 42

        def generate(self, request, progress, cancel):
            return TTSBackend.generate(self, request, progress, cancel)

    with pytest.raises(AppError) as exc:
        Planned().generate(EngineRequest(variant="tone", text="hola", params={}), lambda *_: None, CancelToken())
    assert exc.value.code is ErrorCode.NOT_IMPLEMENTED and "fase 42" in exc.value.message
    for engine_id in ("f5tts", "e2tts", "qwen3tts"):
        assert registry.get(engine_id).implementation_phase is None


def test_loaded_state_is_required_before_generating(registry):
    engine: TTSBackend = registry.get("e2tts")
    with pytest.raises(AppError) as exc:
        engine.generate(EngineRequest(variant="E2TTS_Base", text="hola", params={}), lambda *_: None,
                        CancelToken())
    assert exc.value.code is ErrorCode.MODEL_LOAD_ERROR


def test_mock_engine_generates_reproducibly():
    mock = MockBackend()
    req = EngineRequest(variant="tone", text="hola mundo", params={"seed": 7, "speed": 2.0})
    a = mock.generate(req, lambda *_: None, CancelToken())
    b = mock.generate(req, lambda *_: None, CancelToken())
    assert a.seed == 7 and (a.audio == b.audio).all() and a.sample_rate == 24000


# ---------------- API ----------------
@pytest.fixture
def api(client, settings):
    settings.enable_mock_engine = True
    get_engine_registry.cache_clear()
    yield client
    get_engine_registry.cache_clear()


def test_api_lists_engines_with_install_state(api):
    body = api.get("/api/models").json()
    by_id = {e["id"]: e for e in body}
    assert [e["id"] for e in body][:3] == REAL_ENGINES
    assert by_id["f5tts"]["installed"] is (by_id["f5tts"]["missing_packages"] == [])
    assert by_id["f5tts"]["implemented"] is True and by_id["e2tts"]["implemented"] is True
    assert by_id["f5tts"]["license"]["commercial_use"] == "no_permitido"
    assert by_id["qwen3tts"]["implemented"] is True and by_id["mock"]["implemented"] is True


def test_api_engine_config_per_variant(api):
    base = api.get("/api/models/qwen3tts?variant=base-0.6b").json()
    assert base["variant"]["id"] == "base-0.6b" and base["capabilities"]["mode"] == "clone"
    assert "instruct" not in [p["id"] for p in base["parameters"]]
    cv = api.get("/api/models/qwen3tts", params={"variant": "custom-voice-1.7b"}).json()
    assert "instruct" in [p["id"] for p in cv["parameters"]]
    assert [p["id"] for p in api.get("/api/models/f5tts").json()["presets"]] == ["fast", "balanced", "high_fidelity"]
    assert cv["presets"] == []  # Qwen3-TTS has no documented quality presets
    assert api.get("/api/models/f5tts/capabilities").json()["controls"]["emotion"]["source"] == "segmentation"
    assert len(api.get("/api/models/e2tts/schema").json()) == 9


def test_api_validate_and_errors(api):
    ok = api.post("/api/models/f5tts/validate", json={"params": {"nfe_steps": "24", "seed": 5}}).json()
    assert ok["variant"] == "F5TTS_v1_Base" and ok["params"]["nfe_steps"] == 24 and ok["params"]["speed"] == 1.0
    bad = api.post("/api/models/f5tts/validate", json={"params": {"nfe_steps": 1}})
    assert bad.status_code == 422 and bad.json()["error_code"] == "PARAMETER_ERROR"
    assert bad.json()["details"][0]["etiqueta"] == "Pasos de inferencia"
    missing = api.get("/api/models/xtts")
    assert missing.status_code == 404 and missing.json()["error_code"] == "ENGINE_NOT_FOUND"
    assert api.get("/api/models/f5tts?variant=nope").status_code == 404


# ---------------- segment snapping ----------------
WORDS = [{"start": 0.3, "end": 0.7, "word": " Hola"}, {"start": 0.9, "end": 1.4, "word": " a"},
         {"start": 1.5, "end": 2.4, "word": " todos."}, {"start": 3.0, "end": 3.6, "word": " Hoy"}]


def test_word_boundaries_are_in_pauses():
    assert word_boundaries(WORDS, 5.0) == pytest.approx([0.22, 0.8, 1.45, 2.7, 3.68])


def test_snap_moves_edges_out_of_words():
    assert snap_segment(0.5, 2.2, WORDS, 5.0) == (0.22, 2.7)   # both edges were inside words
    assert snap_segment(0.5, 4.8, WORDS, 5.0) == (0.22, 4.8)   # end too far from any boundary: unchanged
    assert snap_segment(0.75, 0.85, WORDS, 5.0) == (0.75, 0.85)  # would collapse: unchanged
    assert snap_segment(1.0, 2.0, [], 5.0) == (1.0, 2.0)
