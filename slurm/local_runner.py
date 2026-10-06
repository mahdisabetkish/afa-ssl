#!/usr/bin/env python
"""Mini-slurm for a laptop.

Reads a queue file (JSONL, one job per line: {"id": ..., "cmd": [...]}), runs the
jobs sequentially (or at most 2 at once) under `nice`/`ionice` with the thread and
memory caps, monitors RSS with psutil, retries failures, and resumes: jobs whose
status file says "done" are skipped.

    python slurm/local_runner.py slurm/jobs/main.jsonl [--max-concurrent 1] [--retries 2]
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from afa_ssl.utils.resources import THREAD_VARS, machine_state, read_max_temp, wait_until_cool  # noqa: E402

VENV_PY = os.environ.get("AFA_PY", sys.executable)


def log(msg: str, fh) -> None:
    line = f"[{time.strftime('%Y-%m-%dT%H:%M:%S')}] {msg}"
    print(line, flush=True)
    fh.write(line + "\n")
    fh.flush()


def job_status_path(job: dict) -> Path:
    if "status_file" in job:
        return Path(job["status_file"])
    return Path(job.get("out_dir", f"experiments/{job['id']}")) / "status.json"


def is_done(job: dict) -> bool:
    p = job_status_path(job)
    if not p.exists():
        return False
    try:
        return json.load(open(p)).get("status") == "done"
    except json.JSONDecodeError:
        return False


def launch(job: dict, threads: int, max_ram_gb: float, log_fh) -> subprocess.Popen:
    env = dict(os.environ)
    for v in THREAD_VARS:
        env[v] = str(threads)
    # waiting OpenMP threads sleep instead of spinning: avoids the oversubscription cliff when
    # another process (a browser, say) holds a core
    env.setdefault("OMP_WAIT_POLICY", "PASSIVE")
    env.setdefault("KMP_BLOCKTIME", "0")
    cmd = ["nice", "-n", "10", "ionice", "-c2", "-n7", VENV_PY] + job["cmd"] + ["--threads", str(threads)]
    out = Path(job.get("out_dir", f"experiments/{job['id']}"))
    out.mkdir(parents=True, exist_ok=True)
    stdout = open(out / "stdout.log", "a")
    log(f"launch {job['id']}: {' '.join(job['cmd'])}", log_fh)
    return subprocess.Popen(cmd, env=env, stdout=stdout, stderr=subprocess.STDOUT)


def rss_gb(proc: subprocess.Popen) -> float:
    import psutil

    try:
        p = psutil.Process(proc.pid)
        total = p.memory_info().rss
        for c in p.children(recursive=True):
            total += c.memory_info().rss
        return total / 1024**3
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("queue")
    ap.add_argument("--max-concurrent", type=int, default=1)
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--max-ram-gb", type=float, default=6.0)
    ap.add_argument("--poll", type=float, default=3.0)
    ap.add_argument("--max-load", type=float, default=6.0)
    ap.add_argument("--max-temp", type=float, default=85.0, help="do not start a job above this")
    ap.add_argument("--pause-temp", type=float, default=88.0, help="SIGSTOP a running job above this")
    ap.add_argument("--resume-temp", type=float, default=78.0, help="SIGCONT once below this")
    ap.add_argument("--pause-avail-gb", type=float, default=0.4, help="SIGSTOP when free memory is below this")
    ap.add_argument("--resume-avail-gb", type=float, default=0.8)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    args.max_concurrent = max(1, min(args.max_concurrent, 2))

    jobs = [json.loads(l) for l in open(args.queue) if l.strip() and not l.startswith("#")]
    Path("logs").mkdir(exist_ok=True)
    log_fh = open(f"logs/runner_{Path(args.queue).stem}.log", "a")
    pending = [j for j in jobs if not is_done(j)]
    log(f"queue {args.queue}: {len(jobs)} jobs, {len(jobs) - len(pending)} done, {len(pending)} pending", log_fh)
    if args.dry_run:
        for j in pending:
            print(j["id"])
        return
    attempts: dict[str, int] = {}
    running: dict[str, tuple[subprocess.Popen, dict, float]] = {}
    paused: dict[str, str] = {}  # job id -> reason while SIGSTOPped
    import psutil

    def watchdog() -> None:
        """Within-job protection: freeze the job when the CPU is too hot or memory is
        nearly exhausted, thaw it when the machine has recovered."""
        temp = read_max_temp() or 0.0
        avail = psutil.virtual_memory().available / 1024**3
        for jid, (proc, _, _) in running.items():
            if proc.poll() is not None:
                continue
            if jid not in paused:
                if temp > args.pause_temp:
                    paused[jid] = "temp"
                elif avail < args.pause_avail_gb:
                    paused[jid] = "memory"
                if jid in paused:
                    proc.send_signal(signal.SIGSTOP)
                    log(f"PAUSE {jid}: {paused[jid]} (temp={temp:.0f}C avail={avail:.2f} GB)", log_fh)
            else:
                ok_temp = temp < args.resume_temp
                ok_mem = avail > args.resume_avail_gb
                if (paused[jid] == "temp" and ok_temp and avail > args.pause_avail_gb) or \
                   (paused[jid] == "memory" and ok_mem and temp < args.pause_temp):
                    proc.send_signal(signal.SIGCONT)
                    log(f"RESUME {jid} (temp={temp:.0f}C avail={avail:.2f} GB)", log_fh)
                    del paused[jid]

    while pending or running:
        while pending and len(running) < args.max_concurrent:
            wait_until_cool(args.max_load if args.max_concurrent == 1 else args.max_load + 2,
                            args.max_temp, log=lambda m: log(m, log_fh))
            job = pending.pop(0)
            attempts[job["id"]] = attempts.get(job["id"], 0) + 1
            running[job["id"]] = (launch(job, args.threads, args.max_ram_gb, log_fh), job, time.time())
        time.sleep(args.poll)
        watchdog()
        for jid, (proc, job, t0) in list(running.items()):
            mem = rss_gb(proc)
            if mem > args.max_ram_gb:
                log(f"KILL {jid}: rss {mem:.2f} GB > {args.max_ram_gb} GB", log_fh)
                proc.kill()
            rc = proc.poll()
            if rc is None:
                continue
            del running[jid]
            paused.pop(jid, None)
            dt = time.time() - t0
            if rc == 0 and is_done(job):
                st = machine_state()
                log(f"done {jid} in {dt/60:.1f} min (load {st.load1:.1f}, temp {st.temp_c})", log_fh)
            else:
                log(f"FAILED {jid} rc={rc} after {dt/60:.1f} min (attempt {attempts[jid]})", log_fh)
                if attempts[jid] <= args.retries:
                    pending.append(job)
                else:
                    log(f"giving up on {jid}", log_fh)
    log("queue finished", log_fh)


if __name__ == "__main__":
    main()
