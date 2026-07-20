#!/usr/bin/env python3
"""Run N pending sequences (both models) non-interactively."""
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
MODELS = ("v8s", "v8m")


def done(seq: str) -> bool:
    return all((OUT / seq / m / "artifacts" / "report" / "summary.json").exists() for m in MODELS)


def pending() -> list[str]:
    return [s.name for s in sorted(VAL.iterdir()) if s.is_dir() and not done(s.name)]


def run_batch(count: int, device: str) -> None:
    env = os.environ.copy()
    env["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
    batch = pending()[:count]
    print(f"Running {len(batch)} sequence(s): {batch}", flush=True)
    for seq in batch:
        for model in MODELS:
            summary = OUT / seq / model / "artifacts" / "report" / "summary.json"
            if summary.exists():
                print(f"Skip {seq}/{model}", flush=True)
                continue
            log_dir = OUT / seq / model / "artifacts" / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            log_path = log_dir / "pipeline.log"
            print(f"\n{'='*60}\n{seq} / {model}\n{'='*60}", flush=True)
            cmd = [
                "conda", "run", "-n", "shoaib", "python", str(SCRIPT),
                "--seq-dir", str(VAL / seq),
                "--output-root", str(OUT),
                "--model-tag", model,
                "--device", device,
            ]
            with log_path.open("a") as f:
                proc = subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT)
            if proc.returncode != 0:
                tail = log_path.read_text().splitlines()[-15:]
                print("\n".join(tail), file=sys.stderr)
                sys.exit(1)
        print(f"\n--- Results: {seq} ---", flush=True)
        for model in MODELS:
            d = json.loads((OUT / seq / model / "artifacts" / "report" / "summary.json").read_text())
            g = d["generated_pseudo_labels"]
            r = d["recovered_pseudo_labels"]
            i = d["improvement_recovered_minus_generated"]
            tag = "IMPROVED" if i["map50"] > 0 else "no gain"
            print(
                f"  [{model}] gen mAP@0.5={g['map50']:.3f}  rec={r['map50']:.3f}  "
                f"delta={i['map50']:+.3f}  {tag}",
                flush=True,
            )
    for model in MODELS:
        subprocess.run(
            [
                "conda", "run", "-n", "shoaib", "python", str(AGG),
                "--results-root", str(OUT),
                "--model-tag", model,
                "--output-csv", str(OUT / f"summary_{model}.csv"),
                "--improved-list", str(OUT / f"sequences_improved_map50_{model}.txt"),
            ],
            check=True,
        )
    print("\nBatch complete.", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("-n", "--count", type=int, required=True)
    p.add_argument("--device", default=os.environ.get("DEVICE", "0"))
    args = p.parse_args()
    run_batch(args.count, args.device)
