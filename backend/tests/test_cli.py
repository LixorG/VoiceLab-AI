"""Maintenance CLI: doctor exit codes and weight downloads (simulated engines, no network)."""

from __future__ import annotations

import pytest

from app import __version__, cli
from app.core import environment
from app.core.environment import Check, EnvironmentReport
from app.core.gpu import GPUInfo
from app.engines import registry as registry_module
from app.engines.registry import EngineRegistry
from tests.test_mock_gpu import HeavyMock


def _report(ffmpeg_status: str) -> EnvironmentReport:
    return EnvironmentReport(platform="test", python_version="3.11", gpu=GPUInfo(backend="cpu", torch_available=False,
                                                                                source="none"),
                             checks=[Check(id="python", label="Python", status="ok", detail="3.11"),
                                     Check(id="ffmpeg", label="FFmpeg disponible", status=ffmpeg_status),
                                     Check(id="flash_attn", label="FlashAttention 2", status="missing")])


@pytest.mark.parametrize(("ffmpeg", "code", "message"), [("ok", 0, "Listo para usar."),
                                                        ("error", 1, "Faltan requisitos obligatorios.")])
def test_doctor_exit_code(settings, monkeypatch, capsys, ffmpeg, code, message):
    monkeypatch.setattr(environment.EnvironmentChecker, "run", lambda self: _report(ffmpeg))
    assert cli.main(["doctor"]) == code
    out = capsys.readouterr().out
    assert message in out and "[--]    FlashAttention 2" in out and f"VoiceLab AI {__version__}" in out


def test_download_engine_weights(settings, monkeypatch, capsys):
    engine = HeavyMock()
    engine.weights = False
    registry = EngineRegistry()
    registry.register(engine)
    monkeypatch.setattr(registry_module.EngineRegistry, "discover", lambda self, include_dev=False: registry)

    assert cli.main(["download", "heavy"]) == 0 and engine.downloaded == [engine.default_variant()]
    assert cli.main(["download", "heavy"]) == 0 and "ya descargado" in capsys.readouterr().out
    assert cli.main(["download", "heavy", "--variant", "nope"]) == 1
    assert cli.main(["download", "nope"]) == 1

    monkeypatch.setattr(engine, "is_installed", lambda: False)
    engine.weights = False
    assert cli.main(["download", "heavy"]) == 1 and "Falta instalar" in capsys.readouterr().out


def test_version(capsys):
    assert cli.main(["version"]) == 0 and capsys.readouterr().out.strip() == __version__
