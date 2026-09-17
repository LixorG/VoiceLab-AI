"""Build a distributable ZIP of VoiceLab AI (source + compiled UI; no venv, node_modules, data or model weights).

    backend\\.venv\\Scripts\\python scripts\\package.py            # -> release/voicelab-ai-<version>.zip
    python3 scripts/package.py --skip-build                     # reuse frontend/dist as it is

Standard library only. Entry names always use forward slashes so the archive extracts correctly everywhere.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

INCLUDE = [
    "README.md", "CHANGELOG.md", "CLAUDE.md", ".env.example", ".gitignore", ".gitattributes", ".dockerignore",
    "Dockerfile", "docker-compose.yml", "docker-compose.gpu.yml", "install.cmd", "VoiceLab.cmd",
    "backend/pyproject.toml", "backend/constraints-torch.txt", "backend/app", "backend/tests",
    "frontend/package.json", "frontend/package-lock.json", "frontend/index.html", "frontend/vite.config.ts",
    "frontend/tsconfig.json", "frontend/components.json", "frontend/.oxlintrc.json", "frontend/public",
    "frontend/src", "frontend/dist",
    "scripts", "docs", "data/.gitkeep", "models/.gitkeep",
]
EXCLUDED_PARTS = {"__pycache__", ".pytest_cache", ".ruff_cache", "node_modules", ".venv", "coverage", "htmlcov"}
EXCLUDED_SUFFIXES = {".pyc", ".pyo", ".log"}


def version() -> str:
    text = (ROOT / "backend" / "app" / "__init__.py").read_text(encoding="utf-8")
    match = re.search(r'__version__\s*=\s*"([^"]+)"', text)
    if not match:
        raise SystemExit("No se encontró __version__ en backend/app/__init__.py")
    return match.group(1)


def files() -> list[Path]:
    out: list[Path] = []
    for entry in INCLUDE:
        path = ROOT / entry
        if not path.exists():
            if entry == "frontend/dist":
                raise SystemExit("Falta frontend/dist: compila la interfaz (npm run build) o no uses --skip-build.")
            continue
        candidates = [path] if path.is_file() else sorted(p for p in path.rglob("*") if p.is_file())
        for file in candidates:
            rel = file.relative_to(ROOT)
            if EXCLUDED_PARTS.intersection(rel.parts) or file.suffix in EXCLUDED_SUFFIXES:
                continue
            out.append(file)
    return out


def build_frontend() -> None:
    npm = shutil.which("npm")
    if npm is None:
        raise SystemExit("npm no está disponible: instala Node.js o usa --skip-build con frontend/dist ya compilado.")
    print("Compilando la interfaz...", flush=True)
    subprocess.run([npm, "run", "build"], cwd=ROOT / "frontend", check=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Empaqueta VoiceLab AI en un ZIP distribuible")
    parser.add_argument("--skip-build", action="store_true", help="No recompilar la interfaz")
    parser.add_argument("--out", type=Path, default=ROOT / "release", help="Carpeta de salida")
    args = parser.parse_args(argv)

    if not args.skip_build:
        build_frontend()
    name = f"voicelab-ai-{version()}"
    args.out.mkdir(parents=True, exist_ok=True)
    target = args.out / f"{name}.zip"
    selected = files()
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for file in selected:
            arcname = f"{name}/{file.relative_to(ROOT).as_posix()}"
            info = zipfile.ZipInfo.from_file(file, arcname)
            if file.suffix == ".sh":
                info.external_attr = 0o100755 << 16  # executable on Linux/macOS
            info.compress_type = zipfile.ZIP_DEFLATED
            with file.open("rb") as src, archive.open(info, "w") as dst:
                shutil.copyfileobj(src, dst)
    size_mb = target.stat().st_size / 2**20
    print(f"Creado {target} ({len(selected)} archivos, {size_mb:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
