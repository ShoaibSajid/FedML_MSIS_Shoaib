#!/usr/bin/env python3
"""Run BDD val sequence evaluation in parallel across all GPUs."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path("/Shoaib/FedML_MSIS_Shoaib/python/app/fedcv/object_detection")
VAL_ROOT = Path("/datasets/BDDComplete/val")
SCRIPT = ROOT / "scripts" / "run_sequence_eval.py"
AGG = ROOT / "scripts" / "aggregate_val_results.py"

WEIGHTS = {
    "full_v8s": {
        "path": "/Shoaib/BDD100k_yolov8s/full_v8s/weights/best.pt",
        "kind": "s",
    },
    "full_v8m": {
        "path": "/Shoaib/BDD100k_yolv8m/full_v8m/weights/best.pt",
        "kind": "m",
    },
}


class GpuPool:
    def __init__(self, gpus: list[int], slots_s: int, slots_m: int) -> None:
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._slots = {g: {"s": slots_s, "m": slots_m} for g in gpus}

    def acquire(self, kind: str) -> int:
        with self._cond:
            while True:
                for gpu in sorted(self._slots):
                    if self._slots[gpu][kind] > 0:
                        self._slots[gpu][kind] -= 1
                        return gpu
                self._cond.wait(timeout=0.5)

    def release(self, gpu: int, kind: str) -> None:
        with self._cond:
            self._slots[gpu][kind] += 1
            self._cond.notify_all()


def run_task(
    seq: str,
    weight_label: str,
    weights_path: str,
    kind: str,
    gpu_pool: GpuPool,
    skip_if_done: bool,
) -> tuple[str, str, str, float]:
    out_root = ROOT / "runs" / "val_eval" / weight_label
    summary = out_root / seq / "report" / "summary.json"
    log_path = out_root / "logs" / f"{seq}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    if skip_if_done and summary.exists():
        return seq, weight_label, "skipped", 0.0

    gpu = gpu_pool.acquire(kind)
    t0 = time.time()
    status = "failed"
    lock_path = out_root / seq / ".running.lock"
    try:
        if lock_path.exists() and not summary.exists():
            # Another worker may still be running this sequence.
            return seq, weight_label, "locked", 0.0

        env = os.environ.copy()
        env["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)

        cmd = [
            "conda", "run", "-n", "shoaib", "python", str(SCRIPT),
            "--weights", weights_path,
            "--seq-dir", str(VAL_ROOT / seq),
            "--output-root", str(out_root),
            "--device", "0",
        ]
        if skip_if_done:
            cmd.append("--skip-if-done")

        with log_path.open("a") as logf:
            logf.write(f"\n=== GPU {gpu} ({kind}) started {time.strftime('%F %T')} ===\n")
            logf.flush()
            proc = subprocess.run(
                cmd,
                env=env,
                stdout=logf,
                stderr=subprocess.STDOUT,
                text=True,
            )
        status = "ok" if proc.returncode == 0 and summary.exists() else "failed"
    finally:
        gpu_pool.release(gpu, kind)

    elapsed = time.time() - t0
    return seq, weight_label, status, elapsed


def aggregate(weight_label: str) -> None:
    out_root = ROOT / "runs" / "val_eval" / weight_label
    csv_path = out_root / "summary_all_sequences.csv"
    subprocess.run(
        [
            "conda", "run", "-n", "shoaib", "python", str(AGG),
            "--results-root", str(out_root),
            "--output-csv", str(csv_path),
            "--improved-list", str(out_root / "sequences_improved_map50.txt"),
        ],
        check=True,
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--gpus", default="0,1,2,3,4,5,6", help="Comma-separated GPU ids")
    p.add_argument("--slots-s", type=int, default=3, help="Parallel v8s jobs per GPU")
    p.add_argument("--slots-m", type=int, default=2, help="Parallel v8m jobs per GPU")
    p.add_argument("--max-workers", type=int, default=0, help="Thread pool size (0=auto)")
    p.add_argument("--skip-if-done", action="store_true", default=True)
    p.add_argument("--weights", nargs="*", default=list(WEIGHTS.keys()))
    args = p.parse_args()

    gpus = [int(x) for x in args.gpus.split(",") if x.strip() != ""]
    sequences = sorted(p.name for p in VAL_ROOT.iterdir() if p.is_dir())
    gpu_pool = GpuPool(gpus, args.slots_s, args.slots_m)

    tasks: list[tuple[str, str, str, str]] = []
    for weight_label in args.weights:
        info = WEIGHTS[weight_label]
        for seq in sequences:
            tasks.append((seq, weight_label, info["path"], info["kind"]))

    max_workers = args.max_workers or len(gpus) * (args.slots_s + args.slots_m)
    print(
        f"GPUs={gpus} slots_s={args.slots_s} slots_m={args.slots_m} "
        f"workers={max_workers} sequences={len(sequences)} "
        f"tasks={len(tasks)}",
        flush=True,
    )

    ok = skipped = failed = 0
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(
                run_task, seq, wl, wp, kind, gpu_pool, args.skip_if_done
            ): (seq, wl)
            for seq, wl, wp, kind in tasks
        }
        for fut in as_completed(futures):
            seq, wl = futures[fut]
            try:
                seq, wl, status, elapsed = fut.result()
                if status == "ok":
                    ok += 1
                    print(f"[OK] {wl}/{seq} ({elapsed:.0f}s)", flush=True)
                elif status == "skipped":
                    skipped += 1
                elif status == "locked":
                    skipped += 1
                    print(f"[LOCK] {wl}/{seq} (already running)", flush=True)
                else:
                    failed += 1
                    print(f"[FAIL] {wl}/{seq}", flush=True)
            except Exception as exc:
                failed += 1
                print(f"[ERR] {wl}/{seq}: {exc}", file=sys.stderr, flush=True)

    for weight_label in args.weights:
        aggregate(weight_label)
        csv_path = ROOT / "runs" / "val_eval" / weight_label / "summary_all_sequences.csv"
        print(f"CSV: {csv_path}", flush=True)

    print(f"Done ok={ok} skipped={skipped} failed={failed}", flush=True)
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
