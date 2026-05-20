#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import platform
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
ENV_FILE = ROOT / ".env"


@dataclass(frozen=True)
class Service:
    name: str
    module: str | None = None
    venv: Path | None = None
    requirements: Path | None = None
    command: tuple[str, ...] | None = None


SERVICES = {
    "ai": Service(
        name="ai",
        module="ArtificialIntelligence.main",
        venv=ROOT / "ArtificialIntelligence" / ".venv",
        requirements=ROOT / "ArtificialIntelligence" / "requirements.txt",
    ),
    "ip": Service(
        name="ip",
        module="ImageProcessing.main",
        venv=ROOT / "ImageProcessing" / ".venv",
        requirements=ROOT / "ImageProcessing" / "requirements.txt",
    ),
    "vllm": Service(
        name="vllm",
        command=("bash", str(ROOT / "scripts" / "run_server.sh")),
    ),
}

DEFAULT_SERVICES = ("vllm", "ai", "ip")


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
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            env[key] = value
    return env


def merged_env() -> dict[str, str]:
    env = os.environ.copy()
    env.update(load_env_file(ENV_FILE))
    env.setdefault("PYTHONUNBUFFERED", "1")
    return env


def venv_python(venv: Path) -> Path:
    if is_windows():
        return venv / "Scripts" / "python.exe"
    return venv / "bin" / "python"


def create_venv(service: Service, python_bin: str) -> None:
    assert service.venv is not None
    if venv_python(service.venv).exists():
        return

    log(f"[{service.name}] creating venv: {service.venv}")
    subprocess.check_call([python_bin, "-m", "venv", str(service.venv)], cwd=ROOT)


def install_requirements(service: Service) -> None:
    assert service.venv is not None
    assert service.requirements is not None

    python = venv_python(service.venv)
    stamp = service.venv / ".requirements.stamp"
    req_mtime = service.requirements.stat().st_mtime
    if stamp.exists() and float(stamp.read_text(encoding="utf-8") or 0) >= req_mtime:
        return

    log(f"[{service.name}] installing requirements")
    subprocess.check_call([str(python), "-m", "pip", "install", "-U", "pip"], cwd=ROOT)
    subprocess.check_call(
        [str(python), "-m", "pip", "install", "-r", str(service.requirements)],
        cwd=ROOT,
    )
    stamp.write_text(str(time.time()), encoding="utf-8")


def ensure_service(service: Service, python_bin: str) -> None:
    if service.command:
        return
    create_venv(service, python_bin)
    install_requirements(service)


def service_command(service: Service) -> list[str]:
    if service.command:
        return list(service.command)
    assert service.venv is not None
    assert service.module is not None
    return [str(venv_python(service.venv)), "-m", service.module]


def pump_output(name: str, process: subprocess.Popen[str]) -> None:
    assert process.stdout is not None
    for line in process.stdout:
        print(f"[{name}] | {line}", end="", flush=True)


def terminate(processes: Iterable[subprocess.Popen[str]]) -> None:
    live = [process for process in processes if process.poll() is None]
    if not live:
        return

    for process in live:
        if is_windows():
            process.terminate()
        else:
            process.send_signal(signal.SIGTERM)

    deadline = time.monotonic() + 8
    for process in live:
        remaining = max(0.0, deadline - time.monotonic())
        try:
            process.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            process.kill()


def expand_services(names: list[str]) -> list[Service]:
    selected = names or list(DEFAULT_SERVICES)
    unknown = [name for name in selected if name not in SERVICES]
    if unknown:
        raise SystemExit(f"Unknown service(s): {', '.join(unknown)}")
    return [SERVICES[name] for name in selected]


def up(args: argparse.Namespace) -> int:
    services = expand_services(args.services)
    env = merged_env()
    python_bin = os.environ.get("PYTHON", sys.executable)

    for service in services:
        ensure_service(service, python_bin)

    processes: list[subprocess.Popen[str]] = []
    try:
        for index, service in enumerate(services, start=1):
            cmd = service_command(service)
            log(f"[+] starting {service.name}_{index}: {' '.join(cmd)}")
            process = subprocess.Popen(
                cmd,
                cwd=ROOT,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            processes.append(process)
            threading.Thread(
                target=pump_output,
                args=(service.name, process),
                daemon=True,
            ).start()

        while processes:
            for process in processes:
                code = process.poll()
                if code is not None:
                    terminate(processes)
                    return code
            time.sleep(0.25)
    except KeyboardInterrupt:
        log("\n[+] stopping services")
        return 130
    finally:
        terminate(processes)

    return 0


def build(args: argparse.Namespace) -> int:
    services = expand_services(args.services)
    python_bin = os.environ.get("PYTHON", sys.executable)
    for service in services:
        ensure_service(service, python_bin)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Local venv-based runner for The Witch services.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    up_parser = subparsers.add_parser("up", help="start services")
    up_parser.add_argument("services", nargs="*", help="services: ai, ip, vllm")
    up_parser.set_defaults(func=up)

    build_parser = subparsers.add_parser("build", help="create venvs and install requirements")
    build_parser.add_argument("services", nargs="*", help="services: ai, ip, vllm")
    build_parser.set_defaults(func=build)

    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
