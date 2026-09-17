"""Maintenance CLI (Spanish output), used by the install/start scripts and for offline setups.

    python -m app.cli doctor                 # environment report; exit 1 if something required is missing
    python -m app.cli download f5tts         # engine weights (default variant) into MODEL_DIR/huggingface
    python -m app.cli download qwen3tts --variant base-0.6b
    python -m app.cli download asr           # Whisper model for transcription
    python -m app.cli version
"""

from __future__ import annotations

import argparse
import sys

from app import __version__

SYMBOLS = {"ok": "[OK]   ", "warning": "[AVISO]", "error": "[ERROR]", "missing": "[--]   "}
REQUIRED = {"python", "ffmpeg", "ffprobe"}


def _print(text: str = "") -> None:
    # Windows consoles may not be UTF-8: never crash on accents.
    encoding = sys.stdout.encoding or "utf-8"
    sys.stdout.write(text.encode(encoding, errors="replace").decode(encoding) + "\n")


def cmd_doctor(_: argparse.Namespace) -> int:
    from app.core.config import get_settings
    from app.core.environment import EnvironmentChecker
    from app.core.gpu import GPUManager

    settings = get_settings()
    settings.apply_process_environment()
    report = EnvironmentChecker(GPUManager(settings.device)).run()
    _print(f"VoiceLab AI {__version__} · {report.platform} · Python {report.python_version}")
    _print(f"Datos: {settings.data_dir}")
    _print(f"Modelos: {settings.model_dir}")
    built = bool(settings.frontend_dist and (settings.frontend_dist / "index.html").is_file())
    _print(f"Interfaz compilada: {'sí' if built else 'no'}")
    _print()
    failed = False
    for check in report.checks:
        _print(f"{SYMBOLS[check.status]} {check.label}: {check.detail or ''}".rstrip())
        failed |= check.id in REQUIRED and check.status == "error"
    _print()
    _print("Faltan requisitos obligatorios." if failed else "Listo para usar.")
    return 1 if failed else 0


def cmd_download(args: argparse.Namespace) -> int:
    from app.core.config import get_settings

    settings = get_settings()
    settings.apply_process_environment()
    if args.target == "asr":
        from app.asr.manager import get_asr_manager

        asr = get_asr_manager()
        if not asr.backend.package_available():
            _print("faster-whisper no está instalado (scripts\\setup.ps1 lo instala).")
            return 1
        if asr.backend.model_installed():
            _print(f"El modelo {asr.backend.model_name} ya está descargado.")
            return 0
        _print(f"Descargando {asr.backend.model_name}…")
        asr.backend.download_model()
        _print("Descarga completada.")
        return 0

    from app.core.errors import AppError
    from app.engines.registry import EngineRegistry

    registry = EngineRegistry().discover()
    try:
        engine = registry.get(args.target)
        variant = engine.variant(args.variant).id
    except AppError as err:
        _print(err.message)
        return 1
    if not engine.is_installed():
        _print(f"Falta instalar {engine.display_name}: {', '.join(engine.missing_packages())}.")
        return 1
    if engine.weights_installed(variant):
        _print(f"{engine.display_name} · {variant}: ya descargado.")
        return 0
    size = engine.requirements(variant).download_size_mb
    _print(f"Descargando {engine.display_name} · {variant}{f' (~{size} MB)' if size else ''}…")
    engine.download_weights(variant)
    _print("Descarga completada.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m app.cli", description="Utilidades de VoiceLab AI")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor", help="Comprobar el entorno").set_defaults(func=cmd_doctor)
    download = sub.add_parser("download", help="Descargar pesos de un motor o el modelo de transcripción")
    download.add_argument("target", help="f5tts | e2tts | qwen3tts | asr")
    download.add_argument("--variant", default=None, help="Variante del motor (por defecto, la recomendada)")
    download.set_defaults(func=cmd_download)
    sub.add_parser("version", help="Mostrar la versión").set_defaults(func=lambda _: _print(__version__) or 0)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
