"""One-shot watchdog: enforce the 20-minute cap on a run whose runner predates
the fixed scripts/train.py (i.e. run1).

Usage:
    uv run python scripts/enforce_timeout.py run1
"""

from __future__ import annotations

import datetime as dt
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS_LOG = ROOT / "logs" / "training_runs.json"
MAX_SECONDS = 20 * 60
GRACE = 45


def main() -> None:
    name = sys.argv[1]
    runs = json.load(open(RUNS_LOG, encoding="utf-8"))
    rec = next(r for r in runs if r["name"] == name and r["status"] == "running")
    started = dt.datetime.fromisoformat(rec["started_at"])
    deadline = started + dt.timedelta(seconds=MAX_SECONDS)
    now = dt.datetime.now()
    wait = (deadline - now).total_seconds()
    print(f"[watchdog] {name}: deadline={deadline.isoformat()} "
          f"now={now.isoformat()} wait={wait:.0f}s", flush=True)
    if wait > 0:
        time.sleep(wait)

    # Find the actual mlx_lm python child (not the `uv run` wrapper).
    out = subprocess.run(
        ["pgrep", "-f", "mlx_lm.lora"], capture_output=True, text=True,
    ).stdout.split()
    pids = [int(x) for x in out]
    print(f"[watchdog] mlx_lm pids: {pids}", flush=True)
    if not pids:
        print("[watchdog] already gone", flush=True)
        return

    for pid in pids:
        try:
            os.kill(pid, signal.SIGINT)
            print(f"[watchdog] SIGINT -> {pid}", flush=True)
        except ProcessLookupError:
            pass

    t0 = time.time()
    while time.time() - t0 < GRACE:
        alive = [p for p in pids if _alive(p)]
        if not alive:
            print("[watchdog] exited cleanly", flush=True)
            return
        time.sleep(2)

    for pid in pids:
        if _alive(pid):
            try:
                os.kill(pid, signal.SIGKILL)
                print(f"[watchdog] SIGKILL -> {pid}", flush=True)
            except ProcessLookupError:
                pass
    print("[watchdog] done", flush=True)


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


if __name__ == "__main__":
    main()
