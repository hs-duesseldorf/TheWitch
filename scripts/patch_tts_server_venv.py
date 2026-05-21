#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TTS_DIR = ROOT / "ArtificialIntelligence" / "servers" / "tts"
PATCH_DIR = ROOT / "scripts" / "tts_python_patches"
DEFAULT_VENV = TTS_DIR / ".venv"
PTH_NAME = "thewitch_tts_python_patches.pth"


def _default_python() -> Path | None:
    python = DEFAULT_VENV / "bin" / "python"
    if python.exists():
        return python
    return None


def _site_packages(python: Path) -> Path:
    code = (
        "import json, sysconfig; "
        "print(json.dumps(sysconfig.get_paths()[\"purelib\"]))"
    )
    output = subprocess.check_output(
        [str(python), "-c", code],
        text=True,
    )
    return Path(json.loads(output))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Install the tracked Qwen3-TTS runtime patches into the server TTS "
            "virtualenv by loading scripts/tts_python_patches at interpreter startup."
        )
    )
    parser.add_argument(
        "--python",
        type=Path,
        default=_default_python(),
        help=(
            "Python executable from the server TTS venv. Defaults to "
            "ArtificialIntelligence/servers/tts/.venv/bin/python when present."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be written without changing the venv.",
    )
    parser.add_argument(
        "--remove",
        action="store_true",
        help="Remove the TheWitch TTS patch .pth file from the selected venv.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.python is None:
        print(
            "No default TTS venv Python found. Pass --python /path/to/venv/bin/python.",
            file=sys.stderr,
        )
        return 2

    python = args.python.resolve()

    if not python.exists():
        print(f"Python executable not found: {python}", file=sys.stderr)
        return 2
    if not PATCH_DIR.exists():
        print(f"Patch directory not found: {PATCH_DIR}", file=sys.stderr)
        return 2

    site_packages = _site_packages(python)
    pth_file = site_packages / PTH_NAME
    patch_dir = PATCH_DIR.resolve()
    pth_body = (
        "import runpy, sys; "
        f"p = {str(patch_dir)!r}; "
        "sys.path.insert(0, p) if p not in sys.path else None; "
        "runpy.run_path(p + '/sitecustomize.py')\n"
    )

    if args.remove:
        if not pth_file.exists():
            print(f"No patch file installed: {pth_file}")
            return 0
        print(f"Removing {pth_file}")
        if not args.dry_run:
            pth_file.unlink()
        return 0

    print(f"Python:       {python}")
    print(f"site-packages: {site_packages}")
    print(f"Patch path:   {patch_dir}")
    print(f"Install file: {pth_file}")

    if args.dry_run:
        print("Dry run only; no files changed.")
        return 0

    site_packages.mkdir(parents=True, exist_ok=True)
    if pth_file.exists() and pth_file.read_text(encoding="utf-8") == pth_body:
        print("Patch path is already installed.")
        return 0

    pth_file.write_text(pth_body, encoding="utf-8")
    print("Installed TTS patch path.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
