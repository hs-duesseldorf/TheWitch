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


def venv_python(venv: Path) -> Path:
    return venv / ("Scripts" if is_windows() else "bin") / ("python.exe" if is_windows() else "python")


def ensure_service(service: Service, python_bin: str) -> None:
    assert service.venv is not None
    assert service.requirements is not None
    python = venv_python(service.venv)
    stamp = service.venv / ".requirements.stamp"
    if stamp.exists() and python.exists():
        req_mtime = service.requirements.stat().st_mtime
        if float(stamp.read_text(encoding="utf-8") or 0) >= req_mtime:
            return
    log(f"[{service.name}] creating venv")
    subprocess.check_call([python_bin, "-m", "venv", str(service.venv)], cwd=ROOT)
    log(f"[{service.name}] installing requirements")
    subprocess.check_call([str(python), "-m", "pip", "install", "-U", "pip"], cwd=ROOT)
    subprocess.check_call([str(python), "-m", "pip", "install", "-r", str(service.requirements)], cwd=ROOT)
    stamp.write_text(str(time.time()), encoding="utf-8")


def ensure_llm(service: Service, python_bin: str) -> None:
    assert service.venv is not None
    assert service.requirements is not None
    python = venv_python(service.venv)
    stamp = service.venv / ".requirements.stamp"
    if stamp.exists() and python.exists():
        req_mtime = service.requirements.stat().st_mtime
        if float(stamp.read_text(encoding="utf-8") or 0) >= req_mtime:
            return
    log(f"[{service.name}] creating venv")
    subprocess.check_call([python_bin, "-m", "venv", str(service.venv)], cwd=ROOT)
    log(f"[{service.name}] installing requirements")
    subprocess.check_call([str(python), "-m", "pip", "install", "-U", "pip"], cwd=ROOT)
    subprocess.check_call([str(python), "-m", "pip", "install", "-r", str(service.requirements)], cwd=ROOT)
    stamp.write_text(str(time.time()), encoding="utf-8")

    llama_dir = service.venv / "llama-cpp"
    llama_server = llama_dir / "llama-server"
    if llama_server.exists():
        return
    log("[llm] downloading llama.cpp")
    env = load_env_file(ENV_FILE)
    version = env.get("LLAMA_VERSION", "9222")
    tag = f"b{version}"
    archive = llama_dir / f"llama-{tag}-bin-ubuntu-vulkan-x64.tar.gz"
    url = f"https://github.com/ggml-org/llama.cpp/releases/download/{tag}/{archive.name}"
    llama_dir.mkdir(parents=True, exist_ok=True)
    subprocess.check_call(["curl", "-fL", "--retry", "3", "--retry-delay", "2", "-o", str(archive), url])
    subprocess.check_call(["tar", "-xzf", str(archive), "-C", str(llama_dir)])
    subprocess.check_call(["chmod", "+x", str(llama_server)])


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
    selected = names or list(DEFAULT_SERVICES)
    unknown = [n for n in selected if n not in SERVICES]
    if unknown:
        raise SystemExit(f"Unknown service(s): {', '.join(unknown)}")
    return [SERVICES[n] for n in selected]


def find_pids(name: str, command: tuple[str, ...]) -> list[int]:
    pids: list[int] = []
    pattern = command[0] if command[0] != "python" else f"-m {command[2]}"
    if command[0] == "python":
        pattern = command[1]
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
        services = [SERVICES[n] for n in names] if names else []
        return kill_services(services)

    services = expand_services(args.command)
    python_bin = os.environ.get("PYTHON", sys.executable)

    if args.build:
        for service in services:
            if service.name == "llm":
                ensure_llm(service, python_bin)
            elif service.venv is not None:
                ensure_service(service, python_bin)
    else:
        for service in services:
            if service.name == "llm":
                ensure_llm(service, python_bin)
            elif service.venv is not None:
                if not (service.venv / ".requirements.stamp").exists():
                    ensure_service(service, python_bin)

    env = merged_env()
    processes: list[subprocess.Popen[str]] = []
    try:
        for index, service in enumerate(services, start=1):
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
                bufsize=1,
            )
            processes.append(process)
            threading.Thread(target=pump_output, args=(service.name, process), daemon=True).start()

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


if __name__ == "__main__":
    raise SystemExit(main())