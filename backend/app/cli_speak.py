"""Generate voice from the terminal, without opening the application.

For automating what the interface already does: a script that renders a list of lines, a video pipeline that needs
the audio of a caption, a batch left running at night. It reuses the same services as the web app (same validation,
same queue, same files, same database), so anything generated here also appears in the Library.

    python -m app.cli speak "Hola a todos" --voice "Hanna Miller" --out hola.wav
    python -m app.cli speak --file guion.txt --engine qwen3tts --takes 3 --format mp3
    python -m app.cli voices
    python -m app.cli engines
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import shutil
import socket
import sys
import unicodedata
from pathlib import Path
from typing import Any

BUSY_PORT_WARNING = ("AVISO: la aplicación parece estar abierta en el puerto {port}. Los dos procesos cargarían el "
                     "modelo en la misma GPU; cierra la aplicación si esto va lento o falla por memoria.")


def _print(text: str = "", stream: Any = None) -> None:
    # Windows consoles may not be UTF-8: never crash on accents.
    out = stream or sys.stdout
    encoding = out.encoding or "utf-8"
    out.write(text.encode(encoding, errors="replace").decode(encoding) + "\n")


def _fold(text: str) -> str:
    """Compare names the way a person would: without accents and without caring about capitals."""
    return "".join(c for c in unicodedata.normalize("NFD", text.lower()) if unicodedata.category(c) != "Mn").strip()


def _prepare() -> Any:
    """Settings, logging and database, exactly as the server does at startup."""
    from sqlmodel import Session

    from app.core.config import get_settings
    from app.core.database import get_engine, init_engine
    from app.core.logging import configure_logging
    from app.services.checkpoint_service import refresh_store

    settings = get_settings()
    settings.ensure_directories()
    settings.apply_process_environment()
    configure_logging(settings.log_level, settings.data_dir / "logs")
    for handler in list(logging.getLogger("voicelab").handlers):
        if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
            logging.getLogger("voicelab").removeHandler(handler)  # the log belongs in the file, not in the output
    init_engine(settings)
    with Session(get_engine()) as session:
        refresh_store(session)  # custom checkpoints are engine variants here too
    return settings


def _server_running(port: int) -> bool:
    with socket.socket() as probe:
        probe.settimeout(0.2)
        return probe.connect_ex(("127.0.0.1", port)) == 0


def _parse_params(pairs: list[str]) -> dict[str, Any]:
    """`--param speed=1.1` → {"speed": 1.1}. Numbers and booleans are converted; the rest stays text."""
    params: dict[str, Any] = {}
    for pair in pairs:
        key, sep, raw = pair.partition("=")
        if not sep or not key.strip():
            raise ValueError(f"Parámetro mal escrito: «{pair}». Se escribe nombre=valor, por ejemplo speed=1.1.")
        value: Any = raw.strip()
        if value.lower() in ("true", "false"):
            value = value.lower() == "true"
        else:
            try:
                value = int(value) if value.lstrip("-").isdigit() else float(value)
            except ValueError:
                pass
        params[key.strip()] = value
    return params


def _resolve_voice(session: Any, wanted: str | None) -> Any:
    """A voice by name or slug; nothing given, nothing chosen (engines that need no reference still work)."""
    from sqlmodel import select

    from app.core.errors import AppError, ErrorCode
    from app.models.entities import VoiceProfile

    if not wanted:
        return None
    profiles = list(session.exec(select(VoiceProfile)))
    matches = [p for p in profiles if _fold(p.name) == _fold(wanted) or p.slug == wanted]
    if not matches:
        partial = [p for p in profiles if _fold(wanted) in _fold(p.name)]
        if len(partial) == 1:
            return partial[0]
        names = ", ".join(sorted(p.name for p in profiles)) or "ninguna"
        raise AppError(ErrorCode.NOT_FOUND, status_code=404,
                       message=f"No hay ninguna voz que se llame «{wanted}». Voces disponibles: {names}.")
    return matches[0]


def _settings_for(profile: Any, engine_id: str) -> tuple[str | None, dict[str, Any]]:
    """The settings saved for that voice and engine in the Voices page, if there are any."""
    recommended = (getattr(profile, "recommended_settings", None) or {}).get(engine_id) if profile else None
    if not recommended:
        return None, {}
    return recommended.get("variant"), dict(recommended.get("params") or {})


def _text_of(args: argparse.Namespace) -> str:
    if args.file:
        path = Path(args.file)
        if not path.is_file():
            raise ValueError(f"No existe el archivo {path}.")
        text = path.read_text(encoding="utf-8")
    else:
        text = args.text or ""
    text = text.strip()
    if not text:
        raise ValueError("Escribe el texto a generar, o pásalo con --file.")
    return text


def _output_path(args: argparse.Namespace, gen: Any, extension: str) -> Path:
    if args.out:
        path = Path(args.out)
        return path if path.suffix else path.with_suffix(f".{extension}")
    return Path.cwd() / f"voicelab_{gen.engine}_{gen.created_at:%Y%m%d_%H%M%S}.{extension}"


async def _generate(args: argparse.Namespace, settings: Any) -> int:
    from sqlmodel import Session

    from app.core.database import get_engine as db_engine
    from app.core.errors import MESSAGES, AppError, ErrorCode
    from app.engines.manager import get_model_manager
    from app.models.enums import JobStatus
    from app.schemas.generation import GenerationCreate
    from app.services.generation_service import GenerationService, get_job_queue

    quiet = args.quiet or args.json
    models = get_model_manager()
    queue = get_job_queue()
    await queue.start()
    with Session(db_engine()) as session:
        service = GenerationService(session, settings, models, queue)
        profile = _resolve_voice(session, args.voice)
        engine_id = args.engine or (profile.default_engine if profile else None) or settings.default_model
        variant, params = _settings_for(profile, engine_id)
        params.update(_parse_params(args.param))
        if args.seed is not None:
            params["seed"] = args.seed
        body = GenerationCreate(
            engine=engine_id, variant=args.variant or variant, text=_text_of(args),
            params=params, profile_id=profile.id if profile else None, reference_id=args.reference,
            emotion=args.emotion, intensity=args.intensity, takes=args.takes,
            markup=not args.no_markup, normalize=not args.no_normalize,
        )
        gen = await service.create(body)
        if not quiet:
            _print(f"Generando con {engine_id}" + (f" - {gen.variant}" if gen.variant else "")
                   + (f" - voz {profile.name}" if profile else "") + f" - {len(body.text)} caracteres")

        last = ""
        while True:
            state = await queue.get(gen.id)
            if not quiet and state.message and state.message != last:
                last = state.message
                _print(f"  {int(state.progress * 100):3d} %  {state.message}")
            if state.status.is_terminal:
                break
            await asyncio.sleep(0.25)

        session.expire_all()
        gen = service.get(gen.id)
        if gen.status != JobStatus.COMPLETED:
            code = ErrorCode(gen.error_code) if gen.error_code in ErrorCode.__members__ else None
            _print(MESSAGES.get(code, "La generación no terminó correctamente.") if code
                   else "La generación no terminó correctamente.", sys.stderr)
            return 1

        source = service.audio_path(gen)
        if args.format != "wav":
            from app.audio.export import FORMATS, export_audio
            from app.services import mastering

            source = export_audio(source, args.format, settings.data_dir / "cache" / "exports",
                                  mastering.ffmpeg_or_none(settings))
            extension = FORMATS[args.format].extension
        else:
            extension = "wav"
        target = _output_path(args, gen, extension)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)

        if args.json:
            _print(json.dumps({"id": gen.id, "archivo": str(target), "duracion_s": gen.duration_s,
                               "semilla": gen.seed, "motor": gen.engine, "variante": gen.variant,
                               "avisos": gen.warnings or []}, ensure_ascii=False))
        elif not quiet:
            rtf = (gen.metrics or {}).get("rtf")
            _print(f"Listo: {target}")
            _print(f"  {gen.duration_s:.1f} s de audio" + (f" - semilla {gen.seed}" if gen.seed is not None else "")
                   + (f" - RTF {rtf}" if rtf else ""))
            for warning in gen.warnings or []:
                _print(f"  aviso: {warning}")
        return 0
    raise AppError(ErrorCode.INTERNAL_ERROR)  # pragma: no cover (unreachable: the with block always returns)


def cmd_speak(args: argparse.Namespace) -> int:
    from app.core.errors import AppError

    settings = _prepare()
    if not (args.quiet or args.json) and _server_running(settings.port):
        _print(BUSY_PORT_WARNING.format(port=settings.port))
    try:
        return asyncio.run(_run(args, settings))
    except (AppError, ValueError) as err:
        _print(getattr(err, "message", str(err)), sys.stderr)
        return 1
    except KeyboardInterrupt:  # pragma: no cover (interactive)
        _print("\nCancelado.", sys.stderr)
        return 130


async def _run(args: argparse.Namespace, settings: Any) -> int:
    from app.engines.manager import get_model_manager
    from app.services.generation_service import get_job_queue

    try:
        return await _generate(args, settings)
    finally:
        await get_job_queue().stop()
        get_model_manager().unload(force=True)


def cmd_voices(args: argparse.Namespace) -> int:
    from sqlmodel import Session, select

    from app.core.database import get_engine as db_engine
    from app.models.entities import ReferenceAudio, VoiceProfile

    settings = _prepare()
    with Session(db_engine()) as session:
        profiles = sorted(session.exec(select(VoiceProfile)), key=lambda p: _fold(p.name))
        if args.json:
            _print(json.dumps([{"nombre": p.name, "slug": p.slug, "idioma": p.language,
                                "motor": p.default_engine,
                                "ajustes": sorted(p.recommended_settings or {})} for p in profiles],
                              ensure_ascii=False))
            return 0
        if not profiles:
            _print(f"No hay ninguna voz todavía. Crea una en la aplicación ({settings.host}:{settings.port}) "
                   f"o con las referencias de la sección Voces.")
            return 0
        for profile in profiles:
            count = len(list(session.exec(select(ReferenceAudio).where(ReferenceAudio.profile_id == profile.id))))
            extra = [f"{count} grabacion{'es' if count != 1 else ''}"]
            if profile.language:
                extra.append(profile.language)
            if profile.default_engine:
                extra.append(f"motor {profile.default_engine}")
            _print(f"{profile.name}  ({profile.slug})")
            _print(f"    {' - '.join(extra)}")
    return 0


def cmd_engines(args: argparse.Namespace) -> int:
    from app.engines.registry import get_engine_registry

    _prepare()
    rows = []
    for engine in get_engine_registry().all():
        installed = engine.is_installed()
        for variant in engine.variants():
            rows.append({"motor": engine.id, "variante": variant.id, "instalado": installed,
                         "pesos": installed and engine.weights_installed(variant.id), "nombre": engine.display_name})
    if args.json:
        _print(json.dumps(rows, ensure_ascii=False))
        return 0
    for row in rows:
        state = "listo" if row["pesos"] else ("faltan los pesos" if row["instalado"] else "no instalado")
        _print(f"{row['motor']:<10} {row['variante']:<22} {state}")
    if not any(row["pesos"] for row in rows):
        _print("\nDescarga los pesos con: python -m app.cli download <motor>")
    return 0


def register(sub: Any) -> None:
    """Add the generation commands to the CLI parser."""
    speak = sub.add_parser("speak", help="Generar voz y guardarla en un archivo")
    speak.add_argument("text", nargs="?", help="Texto a generar (o usa --file)")
    speak.add_argument("--file", help="Archivo de texto con lo que hay que leer")
    speak.add_argument("--voice", help="Nombre o identificador de la voz (mira: python -m app.cli voices)")
    speak.add_argument("--reference", help="Grabación de referencia concreta; por defecto, la principal de la voz")
    speak.add_argument("--engine", help="Motor: f5tts | e2tts | qwen3tts (por defecto, el de la voz)")
    speak.add_argument("--variant", help="Variante del motor")
    speak.add_argument("--param", action="append", default=[], metavar="NOMBRE=VALOR",
                       help="Parámetro del motor; se puede repetir (ejemplo: --param speed=1.1)")
    speak.add_argument("--seed", type=int, help="Semilla, para repetir exactamente una lectura")
    speak.add_argument("--takes", type=int, default=1, choices=range(1, 6), metavar="1-5",
                       help="Tomas por frase; se conserva la mejor")
    speak.add_argument("--emotion", help="Emoción global, si el motor la admite")
    speak.add_argument("--intensity", type=int, default=50, metavar="0-100", help="Intensidad de la emoción")
    speak.add_argument("--no-markup", action="store_true", help="No interpretar marcas como [pausa:500ms]")
    speak.add_argument("--no-normalize", action="store_true",
                       help="No escribir los números y abreviaturas como se leen")
    speak.add_argument("--format", default="wav", choices=["wav", "mp3", "ogg", "flac"], help="Formato de salida")
    speak.add_argument("--out", "-o", help="Archivo de salida (por defecto, uno con la fecha en esta carpeta)")
    speak.add_argument("--quiet", action="store_true", help="Sin mensajes de progreso")
    speak.add_argument("--json", action="store_true", help="Una línea JSON con el resultado (para scripts)")
    speak.set_defaults(func=cmd_speak)

    voices = sub.add_parser("voices", help="Listar las voces guardadas")
    voices.add_argument("--json", action="store_true", help="Salida en JSON")
    voices.set_defaults(func=cmd_voices)

    engines = sub.add_parser("engines", help="Listar los motores y sus variantes")
    engines.add_argument("--json", action="store_true", help="Salida en JSON")
    engines.set_defaults(func=cmd_engines)
