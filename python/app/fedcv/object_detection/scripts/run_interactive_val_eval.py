#!/usr/bin/env python3
"""Interactive sequential eval: one sequence, both models (v8s + v8m), then prompt."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path("/Shoaib/FedML_MSIS_Shoaib/python/app/fedcv/object_detection")
SCRIPTS = ROOT / "scripts"
VAL_ROOT = Path("/datasets/BDDComplete/val")
OUTPUT_ROOT = ROOT / "runs" / "val_eval"
PROGRESS_FILE = OUTPUT_ROOT / "progress.json"
MODELS = ("v8s", "v8m")


def load_progress() -> dict:
    if PROGRESS_FILE.exists():
        return json.loads(PROGRESS_FILE.read_text())
    return {"completed": [], "last_sequence": None}


def save_progress(progress: dict) -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    PROGRESS_FILE.write_text(json.dumps(progress, indent=2))


def list_sequences() -> list[str]:
    return sorted(p.name for p in VAL_ROOT.iterdir() if p.is_dir())


def is_done(seq: str) -> bool:
    for model in MODELS:
        summary = OUTPUT_ROOT / seq / model / "artifacts" / "report" / "summary.json"
        if not summary.exists():
            return False
    return True


def next_pending(sequences: list[str], progress: dict) -> str | None:
    for seq in sequences:
        if not is_done(seq):
            return seq
    return None


def run_model(seq: str, model: str, device: str) -> dict:
    env = os.environ.copy()
    env["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
    cmd = [
        "conda", "run", "-n", "shoaib", "python", str(SCRIPTS / "run_sequence_eval.py"),
        "--seq-dir", str(VAL_ROOT / seq),
        "--output-root", str(OUTPUT_ROOT),
        "--model-tag", model,
        "--device", device,
        "--skip-if-done",
    ]
    log_dir = OUTPUT_ROOT / seq / model / "artifacts" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "pipeline.log"
    print(f"\n{'='*60}")
    print(f"Sequence: {seq}  |  Model: {model}")
    print(f"Output:   {OUTPUT_ROOT / seq / model / 'artifacts'}")
    print(f"{'='*60}\n", flush=True)
    with log_path.open("a") as logf:
        logf.write(f"\n=== started ===\n")
        proc = subprocess.run(cmd, env=env, stdout=logf, stderr=subprocess.STDOUT, text=True)
    if proc.returncode != 0:
        tail = log_path.read_text().splitlines()[-20:]
        print("\n".join(tail), file=sys.stderr)
        raise RuntimeError(f"Failed {seq}/{model} (see {log_path})")
    summary_path = OUTPUT_ROOT / seq / model / "artifacts" / "report" / "summary.json"
    return json.loads(summary_path.read_text())


def format_metrics(label: str, block: dict) -> str:
    if not block:
        return f"  {label}: (no data)"
    return (
        f"  {label}: P={block['precision']:.3f}  R={block['recall']:.3f}  "
        f"F1={block['f1']:.3f}  mAP@0.5={block['map50']:.3f}  mAP@0.5:0.95={block['map50_95']:.3f}"
    )


def print_sequence_results(seq: str, summaries: dict[str, dict]) -> None:
    print(f"\n{'#'*60}")
    print(f"RESULTS: {seq}")
    print(f"{'#'*60}")
    for model in MODELS:
        s = summaries.get(model)
        if not s:
            print(f"\n[{model}] NOT COMPLETED")
            continue
        gen = s.get("generated_pseudo_labels", {})
        rec = s.get("recovered_pseudo_labels", {})
        imp = s.get("improvement_recovered_minus_generated", {})
        print(f"\n[{model}]")
        print(format_metrics("Generated ", gen))
        print(format_metrics("Recovered ", rec))
        if imp:
            sign = "+" if imp.get("map50", 0) >= 0 else ""
            print(
                f"  Recovery Δ: mAP@0.5={sign}{imp.get('map50', 0):.3f}  "
                f"F1={imp.get('f1', 0):+.3f}  "
                f"{'IMPROVED' if imp.get('map50', 0) > 0 else 'no mAP gain'}"
            )
    print(f"\nArtifacts: {OUTPUT_ROOT / seq}/{{v8s,v8m}}/artifacts/")
    print(f"{'#'*60}\n")


def run_sequence_both_models(seq: str, device: str) -> dict[str, dict]:
    summaries = {}
    for model in MODELS:
        summaries[model] = run_model(seq, model, device)
    return summaries


def update_csvs() -> None:
    for model in MODELS:
        csv_path = OUTPUT_ROOT / f"summary_{model}.csv"
        subprocess.run(
            [
                "conda", "run", "-n", "shoaib", "python",
                str(SCRIPTS / "aggregate_val_results.py"),
                "--results-root", str(OUTPUT_ROOT),
                "--model-tag", model,
                "--output-csv", str(csv_path),
                "--improved-list", str(OUTPUT_ROOT / f"sequences_improved_map50_{model}.txt"),
            ],
            check=False,
        )


def process_sequences(seqs: list[str], device: str, progress: dict) -> None:
    for seq in seqs:
        if is_done(seq):
            print(f"Skip {seq}: both models already complete")
            continue
        summaries = run_sequence_both_models(seq, device)
        print_sequence_results(seq, summaries)
        if seq not in progress["completed"]:
            progress["completed"].append(seq)
        progress["last_sequence"] = seq
        save_progress(progress)
        update_csvs()


def ask_yes_no(prompt: str) -> bool:
    while True:
        ans = input(f"{prompt} [y/n]: ").strip().lower()
        if ans in ("y", "yes"):
            return True
        if ans in ("n", "no"):
            return False
        print("Please enter y or n.")


def ask_count(prompt: str, maximum: int) -> int:
    while True:
        ans = input(f"{prompt} (1-{maximum}): ").strip()
        try:
            n = int(ans)
            if 1 <= n <= maximum:
                return n
        except ValueError:
            pass
        print(f"Enter an integer between 1 and {maximum}.")


def main() -> None:
    device = os.environ.get("DEVICE", "0")
    sequences = list_sequences()
    progress = load_progress()
    pending = [s for s in sequences if not is_done(s)]

    print(f"Val sequences: {len(sequences)} total, {len(pending)} pending")
    print(f"Output layout: {OUTPUT_ROOT}/<seq_id>/{{v8s,v8m}}/artifacts/")
    print(f"Device: {device}\n")

    if not pending:
        print("All sequences complete for both models.")
        update_csvs()
        return

    # First sequence (or resume next pending)
    first = pending[0]
    print(f"Starting sequence 1/1: {first}")
    process_sequences([first], device, progress)

    while True:
        pending = [s for s in sequences if not is_done(s)]
        if not pending:
            print("\nAll sequences complete.")
            update_csvs()
            break
        if not ask_yes_no(f"\nContinue? ({len(pending)} sequences remaining)"):
            print("Stopped. Re-run this script to resume.")
            break
        n = ask_count("How many sequences to run next", len(pending))
        batch = pending[:n]
        print(f"\nRunning {len(batch)} sequence(s): {', '.join(batch)}")
        process_sequences(batch, device, progress)

    update_csvs()
    print(f"\nCSVs updated:")
    for model in MODELS:
        print(f"  {OUTPUT_ROOT / f'summary_{model}.csv'}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(130)
