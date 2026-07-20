#!/usr/bin/env python3
"""Aggregate per-sequence summaries into one CSV with mean row."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


METRICS = ("precision", "recall", "f1", "map50", "map50_95")


def load_summary(path: Path) -> dict | None:
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    gen = data.get("generated_pseudo_labels") or {}
    rec = data.get("recovered_pseudo_labels") or {}
    imp = data.get("improvement_recovered_minus_generated") or {}
    if not gen or not rec:
        return None
    seq = path.parent.parent.parent.parent.name
    model = path.parent.parent.parent.name
    cfg = data.get("config") or {}
    row = {
        "sequence": data.get("sequence", seq),
        "model": data.get("model", model),
        "recovery_mode": cfg.get("recovery_mode", ""),
        "num_images": gen.get("images", 0),
        "gt_boxes": gen.get("gt_boxes", gen.get("labels", 0)),
        "generated_pred_boxes": gen.get("pred_boxes"),
        "recovered_pred_boxes": rec.get("pred_boxes"),
        "delta_pred_boxes": imp.get("pred_boxes"),
    }
    for prefix, block in (("generated", gen), ("recovered", rec)):
        for m in METRICS:
            row[f"{prefix}_{m}"] = block.get(m, 0.0)
    for m in METRICS:
        row[f"delta_{m}"] = imp.get(m, rec.get(m, 0.0) - gen.get(m, 0.0))
    row["improved_map50"] = row["delta_map50"] > 0
    row["improved_map50_95"] = row["delta_map50_95"] > 0
    return row


def mean_row(rows: list[dict]) -> dict:
    if not rows:
        return {"sequence": "MEAN"}
    out = {"sequence": "MEAN", "model": rows[0].get("model", "")}
    skip = ("sequence", "model", "recovery_mode", "improved_map50", "improved_map50_95")
    numeric = [k for k in rows[0] if k not in skip and rows[0][k] is not None]
    for k in numeric:
        vals = [r[k] for r in rows if r.get(k) is not None]
        out[k] = sum(vals) / len(vals) if vals else ""
    out["improved_map50"] = sum(1 for r in rows if r["improved_map50"])
    out["improved_map50_95"] = sum(1 for r in rows if r["improved_map50_95"])
    out["improved_map50"] = f"{out['improved_map50']}/{len(rows)}"
    out["improved_map50_95"] = f"{out['improved_map50_95']}/{len(rows)}"
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results-root", required=True)
    p.add_argument("--model-tag", required=True, choices=["v8s", "v8m"])
    p.add_argument("--recovery-mode", default="", help="Filter by recovery_mode in summary")
    p.add_argument("--output-csv", required=True)
    p.add_argument("--improved-list", default="")
    args = p.parse_args()

    root = Path(args.results_root)
    pattern = f"*/{args.model_tag}/artifacts/report/summary.json"
    rows = []
    for summary in sorted(root.glob(pattern)):
        row = load_summary(summary)
        if not row:
            continue
        if args.recovery_mode and row.get("recovery_mode") != args.recovery_mode:
            continue
        rows.append(row)

    rows.sort(key=lambda r: r["sequence"])
    improved = [r["sequence"] for r in rows if r["delta_map50"] > 0]

    fieldnames = [
        "sequence", "model", "recovery_mode", "num_images", "gt_boxes",
        "generated_pred_boxes", "recovered_pred_boxes", "delta_pred_boxes",
        "generated_precision", "generated_recall", "generated_f1", "generated_map50", "generated_map50_95",
        "recovered_precision", "recovered_recall", "recovered_f1", "recovered_map50", "recovered_map50_95",
        "delta_precision", "delta_recall", "delta_f1", "delta_map50", "delta_map50_95",
        "improved_map50", "improved_map50_95",
    ]

    out_csv = Path(args.output_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(row)
        if rows:
            w.writerow(mean_row(rows))

    if args.improved_list:
        Path(args.improved_list).write_text("\n".join(improved) + ("\n" if improved else ""))

    print(f"Wrote {len(rows)} sequences to {out_csv}")
    print(f"Sequences with mAP@0.5 improvement: {len(improved)}/{len(rows)}")
    if improved:
        print(f"Improved: {', '.join(improved)}")
    if rows:
        mean_map50_gen = sum(r["generated_map50"] for r in rows) / len(rows)
        mean_map50_rec = sum(r["recovered_map50"] for r in rows) / len(rows)
        print(f"Mean mAP@0.5 generated: {mean_map50_gen:.4f}")
        print(f"Mean mAP@0.5 recovered:  {mean_map50_rec:.4f}")


if __name__ == "__main__":
    main()
