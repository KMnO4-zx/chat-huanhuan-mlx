"""Guarded training runner for mlx_lm.lora (via scripts/train_entry.py).

Enforces the program.md constraints:
  * at most 10 training runs in total
  * each run is killed (gracefully, then forcibly) after 20 minutes
  * every run is logged to SwanLab (cloud) by mlx_lm itself

Usage:
    uv run python scripts/train.py --name run1 --config configs/base.yaml
    uv run python scripts/train.py --name run2 --config configs/run2.yaml
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ADAPTERS = ROOT / "outputs" / "adapters"
LOGS = ROOT / "logs"
RUNS_LOG = LOGS / "training_runs.json"
MAX_RUNS = 10
MAX_SECONDS = 20 * 60
GRACE_SECONDS = 45  # give swanlab/Python a chance to flush after SIGINT
URL_RE = re.compile(r"https://swanlab\.cn/\S+")
SWANLAB_PROJECT_URL = "https://swanlab.cn/@kmno4/huanhuan-mlx"


def _swanlog_slug(adapter_path: Path) -> str | None:
    """Read the run slug from swanlog/run-<ts>-<slug>/ when the printed URL
    was never captured (e.g. upstream stdout pipe closed)."""
    base = adapter_path / "swanlog"
    if not base.is_dir():
        return None
    dirs = sorted(base.glob("run-*"))
    if not dirs:
        return None
    name = dirs[-1].name  # run-YYYYMMDD_HHMMSS-<slug>
    return name.rsplit("-", 1)[-1] if "-" in name else None


def load_runs() -> list[dict]:
    if RUNS_LOG.exists():
        with open(RUNS_LOG, encoding="utf-8") as f:
            return json.load(f)
    return []


def save_runs(runs: list[dict]) -> None:
    LOGS.mkdir(exist_ok=True)
    with open(RUNS_LOG, "w", encoding="utf-8") as f:
        json.dump(runs, f, ensure_ascii=False, indent=2)


def pump_output(stream, logf, urls, lock) -> None:
    """Drain child stdout on a side thread so the main thread can enforce
    the wall-clock limit. Mirrors every line to stdout and the log file.

    The mirror to stdout must never be allowed to kill the drain: if our own
    stdout is a closed pipe (EPIPE), keep reading the child so it cannot
    block on a full pipe buffer, and still record URLs/log lines."""
    for line in stream:
        with lock:
            try:
                sys.stdout.write(line)
                sys.stdout.flush()
            except OSError:
                pass
            logf.write(line)
            logf.flush()
            urls.extend(URL_RE.findall(line))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True, help="run name, e.g. run1")
    p.add_argument("--config", required=True, help="YAML config for mlx_lm.lora")
    p.add_argument("--dry-run", action="store_true")
    args, extra = p.parse_known_args()

    runs = load_runs()
    if len(runs) >= MAX_RUNS:
        print(f"ERROR: already {len(runs)} training runs (max {MAX_RUNS}).")
        sys.exit(2)

    adapter_path = ADAPTERS / args.name
    adapter_path.mkdir(parents=True, exist_ok=True)

    # `uv run` guarantees the same locked environment for every run.
    # train_entry.py = mlx_lm.lora + clear_cache_threshold wiring (12GB).
    cmd = [
        "uv", "run", "python", "scripts/train_entry.py",
        "-c", str(ROOT / args.config),
        "--adapter-path", str(adapter_path),
        "--report-to", "swanlab",
        "--project-name", "huanhuan-mlx",
    ] + extra

    print("COMMAND:", " ".join(cmd))
    if args.dry_run:
        return

    log_path = LOGS / f"{args.name}.log"
    record = {
        "run_id": len(runs) + 1,
        "name": args.name,
        "config": args.config,
        "extra_args": extra,
        "adapter_path": str(adapter_path.relative_to(ROOT)),
        "started_at": dt.datetime.now().isoformat(timespec="seconds"),
        "max_seconds": MAX_SECONDS,
        "status": "running",
    }
    runs.append(record)
    save_runs(runs)

    t0 = time.time()
    urls: list[str] = []
    lock = threading.Lock()
    proc: subprocess.Popen | None = None

    try:
        with open(log_path, "w", encoding="utf-8", buffering=1) as logf:
            proc = subprocess.Popen(
                cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1, start_new_session=True,
            )
            assert proc.stdout is not None
            reader = threading.Thread(
                target=pump_output, args=(proc.stdout, logf, urls, lock),
                daemon=True,
            )
            reader.start()

            remaining = MAX_SECONDS - (time.time() - t0)
            try:
                proc.wait(timeout=max(1.0, remaining))
                record["exit_code"] = proc.returncode
                record["status"] = "ok" if proc.returncode == 0 else "failed"
            except subprocess.TimeoutExpired:
                # 20-minute hard limit: SIGINT the whole process group so
                # Python runs atexit handlers (swanlab flush) and mlx_lm
                # exits; fall back to SIGKILL if it ignores us.
                print("\n>>> 20 minute limit reached; interrupting training...")
                try:
                    os_killpg = getattr(__import__("os"), "killpg")
                    os_killpg(proc.pid, signal.SIGINT)
                except Exception:  # noqa: BLE001
                    proc.send_signal(signal.SIGINT)
                try:
                    proc.wait(timeout=GRACE_SECONDS)
                except subprocess.TimeoutExpired:
                    try:
                        os_killpg = getattr(__import__("os"), "killpg")
                        os_killpg(proc.pid, signal.SIGKILL)
                    except Exception:  # noqa: BLE001
                        proc.kill()
                    proc.wait()
                    record["killed"] = True
                record["exit_code"] = proc.returncode
                record["status"] = "timeout_20min"
            reader.join(timeout=10)
    except Exception as e:  # noqa: BLE001
        record["status"] = "error"
        record["error"] = repr(e)
        print("ERROR:", e)
        if proc is not None and proc.poll() is None:
            proc.kill()

    elapsed = time.time() - t0
    record["finished_at"] = dt.datetime.now().isoformat(timespec="seconds")
    record["duration_seconds"] = round(elapsed, 1)
    record["swanlab_urls"] = sorted(set(urls))
    if not any("/runs/" in u for u in record["swanlab_urls"]):
        slug = _swanlog_slug(adapter_path)
        if slug:
            record["swanlab_urls"].append(f"{SWANLAB_PROJECT_URL}/runs/{slug}")
    record["adapter_exists"] = (adapter_path / "adapters.safetensors").exists()
    checkpoints = sorted(adapter_path.glob("*_adapters.safetensors"))
    record["checkpoints"] = [c.name for c in checkpoints]
    save_runs(runs)
    print(f"\n>>> run {args.name}: {record['status']} in {elapsed/60:.1f} min")
    if record["swanlab_urls"]:
        print(">>> swanlab:", record["swanlab_urls"][-1])
    print(">>> checkpoints:", record["checkpoints"])


if __name__ == "__main__":
    main()
