#!/usr/bin/env python3
"""Run v8s forward-recovery on N sequences (one model, one recovery pass)."""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path("/Shoaib/FedML_MSIS_Shoaib/python/app/fedcv/object_detection")
VAL = Path("/datasets/BDDComplete/val")
OUT = ROOT / "runs" / "val_eval"
SCRIPT = ROOT / "scripts" / "run_sequence_eval.py"
AGG = ROOT / "scripts" / "aggregate_val_results.py"
MODEL = "v8s"
RECOVERY = "forward"


def pending(force: bool) -> list[str]:
    seqs = []
    for s in sorted(VAL.iterdir()):
        if not s.is_dir():
            continue
        summary = OUT / s.name / MODEL / "artifacts" / "report" / "summary.json"
        if force or not summary.exists():
            seqs.append(s.name)
            continue
        data = json.loads(summary.read_text())
        if data.get("config", {}).get("recovery_mode") != RECOVERY:
            seqs.append(s.name)
    return seqs


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("-n", "--count", type=int, default=0, help="0 = all pending")
    p.add_argument("--device", default=os.environ.get("DEVICE", "0"))
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    env = os.environ.copy()
    env["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
    todo = pending(args.force)
    if args.count > 0:
        todo = todo[: args.count]
    print(f"v8s forward-only: {len(todo)} sequence(s)", flush=True)

    for seq in todo:
        log_dir = OUT / seq / MODEL / "artifacts" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / "pipeline.log"
        print(f"\n{'='*60}\n{seq} / {MODEL} / {RECOVERY}\n{'='*60}", flush=True)
        cmd = [
            "conda", "run", "-n", "shoaib", "python", str(SCRIPT),
            "--seq-dir", str(VAL / seq),
            "--output-root", str(OUT),
            "--model-tag", MODEL,
            "--recovery", RECOVERY,
            "--device", args.device,
        ]
        if args.force:
            cmd.append("--force")
        with log_path.open("a") as f:
            proc = subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT)
        if proc.returncode != 0:
            print(log_path.read_text().splitlines()[-20:], file=sys.stderr)
            sys.exit(1)
        d = json.loads((OUT / seq / MODEL / "artifacts" / "report" / "summary.json").read_text())
        g, r, i = d["generated_pseudo_labels"], d["recovered_pseudo_labels"], d["improvement_recovered_minus_generated"]
        tag = "IMPROVED" if i.get("map50", 0) > 0 else "no gain"
        print(
            f"  gen mAP@0.5={g['map50']:.3f} ({g.get('pred_boxes')} boxes)  "
            f"rec={r['map50']:.3f} ({r.get('pred_boxes')} boxes)  "
            f"delta={i.get('map50', 0):+.3f}  {tag}",
            flush=True,
        )

    subprocess.run(
        [
            "conda", "run", "-n", "shoaib", "python", str(AGG),
            "--results-root", str(OUT),
            "--model-tag", MODEL,
            "--recovery-mode", RECOVERY,
            "--output-csv", str(OUT / "summary_v8s_forward.csv"),
            "--improved-list", str(OUT / "sequences_improved_map50_v8s_forward.txt"),
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
