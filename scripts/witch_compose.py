#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import platform
import shutil
import signal
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
ENV_FILE = ROOT / ".env"
REQUIRED_PYTHON_VERSION = (3, 12)
UV_PYTHON = "3.12"


@dataclass
class Service:
    name: str
    command: tuple[str, ...]
    venv: Path | None = None
    requirements: Path | None = None


SERVICES = {
    "ai": Service(
        name="ai",
        command=("python", "-m", "ArtificialIntelligence.main"),
        venv=ROOT / "ArtificialIntelligence" / ".venv",
        requirements=ROOT / "ArtificialIntelligence" / "requirements.txt",
    ),
    "ip": Service(
        name="ip",
        command=("python", "-m", "ImageProcessing.main"),
        venv=ROOT / "ImageProcessing" / ".venv",
        requirements=ROOT / "ImageProcessing" / "requirements.txt",
    ),
    "llm": Service(
        name="llm",
        command=("python", "ArtificialIntelligence/servers/llm/run.py"),
        venv=ROOT / "ArtificialIntelligence" / "servers" / "llm" / ".venv",
        requirements=ROOT / "ArtificialIntelligence" / "servers" / "llm" / "requirements.txt",
    ),
    "tts": Service(
        name="tts",
        command=("python", "ArtificialIntelligence/servers/tts/run.py"),
        venv=ROOT / "ArtificialIntelligence" / "servers" / "tts" / ".venv",
        requirements=ROOT / "ArtificialIntelligence" / "servers" / "tts" / "requirements.txt",
    ),
}
DEFAULT_SERVICES = ("llm", "ai", "ip", "tts")


def log(message: str) -> None:
    print(message, flush=True)


def is_windows() -> bool:
    return platform.system() == "Windows"


def load_env_file(path: Path) -> dict[str, str]:
    env: dict[str, str] = {}
    if not path.exists():
        return env
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        env[key.strip()] = value.strip().strip('"').strip("'")
    return env


def merged_env() -> dict[str, str]:
    env = os.environ.copy()
    env.update(load_env_file(ENV_FILE))
    env.setdefault("PYTHONUNBUFFERED", "1")
    return env


def apply_local_service_env(env: dict[str, str], services: list[Service]) -> None:
    names = {service.name for service in services}
    if {"ai", "llm"}.issubset(names):
        env["WITCH_LLM_HOST"] = "127.0.0.1"
        return
    if "llm" in names and env.get("WITCH_LLM_HOST", "").strip() in {
        "",
        "0.0.0.0",
        "::",
    }:
        env["WITCH_LLM_HOST"] = "127.0.0.1"


def venv_python(venv: Path) -> Path:
    return venv / ("Scripts" if is_windows() else "bin") / ("python.exe" if is_windows() else "python")


def uv_bin() -> str:
    uv = shutil.which("uv")
    if uv is None:
        raise SystemExit("uv is required to create Python 3.12 venvs. Install uv and retry.")
    return uv


def python_version(python: Path) -> tuple[int, int] | None:
    if not python.exists():
        return None
    result = subprocess.run(
        [
            str(python),
            "-c",
            "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    try:
        major, minor = result.stdout.strip().split(".", 1)
        return int(major), int(minor)
    except ValueError:
        return None


def ensure_service(service: Service, uv: str) -> None:
    assert service.venv is not None
    assert service.requirements is not None
    python = venv_python(service.venv)
    stamp = service.venv / ".requirements.stamp"
    version = python_version(python)
    if version is not None and version != REQUIRED_PYTHON_VERSION:
        log(
            f"[{service.name}] removing Python {version[0]}.{version[1]} venv; "
            "Python 3.12 is required"
        )
        shutil.rmtree(service.venv)
        version = None
    if stamp.exists() and python.exists():
        req_mtime = service.requirements.stat().st_mtime
        if float(stamp.read_text(encoding="utf-8") or 0) >= req_mtime:
            return
    if not python.exists():
        log(f"[{service.name}] creating venv")
        subprocess.check_call([uv, "venv", "--python", UV_PYTHON, str(service.venv)], cwd=ROOT)
    version = python_version(python)
    if version != REQUIRED_PYTHON_VERSION:
        actual = "unknown" if version is None else f"{version[0]}.{version[1]}"
        raise SystemExit(f"[{service.name}] venv uses Python {actual}; Python 3.12 is required")
    log(f"[{service.name}] installing requirements")
    subprocess.check_call(
        [uv, "pip", "install", "--prerelease=allow", "--python", str(python), "-r", str(service.requirements)],
        cwd=ROOT,
    )
    stamp.write_text(str(time.time()), encoding="utf-8")


def pump_output(name: str, process: subprocess.Popen[str]) -> None:
    assert process.stdout is not None
    for line in process.stdout:
        print(f"[{name}] | {line}", end="", flush=True)


def terminate(processes: Iterable[subprocess.Popen[str]]) -> None:
    live = [p for p in processes if p.poll() is None]
    if not live:
        return
    for p in live:
        (p.terminate() if is_windows() else p.send_signal(signal.SIGTERM))
    deadline = time.monotonic() + 8
    for p in live:
        try:
            p.wait(timeout=max(0.0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            p.kill()


def expand_services(names: list[str]) -> list[Service]:
    selected = names[1:] if names and names[0] == "up" else names
    selected = selected or list(DEFAULT_SERVICES)
    unknown = [n for n in selected if n not in SERVICES]
    if unknown:
        raise SystemExit(f"Unknown service(s): {', '.join(unknown)}")
    return [SERVICES[n] for n in selected]


def find_pids(name: str, command: tuple[str, ...]) -> list[int]:
    pids: list[int] = []
    if len(command) >= 3 and command[0] == "python" and command[1] == "-m":
        pattern = command[2]
    elif command[0] == "python":
        pattern = command[1]
    else:
        pattern = command[0]
    try:
        result = subprocess.run(["pgrep", "-f", pattern], capture_output=True, text=True)
        for line in result.stdout.splitlines():
            if line.strip():
                pids.append(int(line.strip()))
    except Exception:
        pass
    return pids


def kill_services(services: list[Service]) -> int:
    killed = 0
    for service in services:
        pids = find_pids(service.name, service.command)
        if not pids:
            log(f"[{service.name}] no running processes found")
            continue
        for pid in pids:
            try:
                os.kill(pid, signal.SIGTERM)
                log(f"[{service.name}] killed pid {pid}")
                killed += 1
            except OSError as e:
                log(f"[{service.name}] failed to kill pid {pid}: {e}")
    if not killed:
        log("no processes killed")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Local venv-based runner for The Witch services.")
    parser.add_argument("command", nargs="*", help="services: ai, ip, llm, tts (default: all)")
    parser.add_argument("--build", action="store_true", help="create venvs and install requirements before starting")
    args = parser.parse_args()

    if args.command and args.command[0] == "kill":
        names = args.command[1:] if len(args.command) > 1 else []
        services = expand_services(names)
        return kill_services(services)

    services = expand_services(args.command)
    uv = uv_bin()

    if args.build:
        for service in services:
            if service.venv is not None:
                ensure_service(service, uv)
    else:
        for service in services:
            if service.venv is not None:
                if not (service.venv / ".requirements.stamp").exists():
                    ensure_service(service, uv)

    env = merged_env()
    apply_local_service_env(env, services)
    processes: list[tuple[Service, subprocess.Popen[str]]] = []
    exit_code = 0
    try:
        for index, service in enumerate(services, start=1):
            existing_pids = find_pids(service.name, service.command)
            if existing_pids:
                log(f"[{service.name}] already running pid(s): {', '.join(str(pid) for pid in existing_pids)}")
                continue
            cmd = list(service.command)
            if cmd[0] in ("python", "python3") and service.venv is not None:
                cmd[0] = str(venv_python(service.venv))
            log(f"[+] starting {service.name}_{index}: {' '.join(cmd)}")
            process = subprocess.Popen(
                cmd,
                cwd=ROOT,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
            processes.append((service, process))
            threading.Thread(target=pump_output, args=(service.name, process), daemon=True).start()

        while processes:
            for service, process in list(processes):
                code = process.poll()
                if code is not None:
                    processes.remove((service, process))
                    log(f"[{service.name}] exited with code {code}")
                    if code and exit_code == 0:
                        exit_code = code
                        terminate(process for _, process in processes)
                        processes.clear()
                        break
            time.sleep(0.25)
    except KeyboardInterrupt:
        log("\n[+] stopping services")
        return 130
    finally:
        terminate(process for _, process in processes)

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
