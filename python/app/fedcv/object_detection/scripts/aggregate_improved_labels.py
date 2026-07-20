#!/usr/bin/env python3
"""Summarize delta_analysis.json across sequences."""
import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results-root", required=True)
    p.add_argument("--model-tag", default="v8s")
    p.add_argument("--output-csv", required=True)
    args = p.parse_args()
    root = Path(args.results_root)
    rows = []
    for analysis in sorted(root.glob(f"*/{args.model_tag}/artifacts/report/delta_analysis.json")):
        d = json.loads(analysis.read_text())
        ev = d.get("eval") or {}
        high = ev.get("generated_high_conf") or {}
        rec = ev.get("recovered_forward") or {}
        imp = ev.get("recovered_improved_labels") or {}
        seq = analysis.parent.parent.parent.parent.name
        rows.append({
            "sequence": seq,
            "helpful_delta_boxes": d.get("helpful_delta_boxes", 0),
            "fp_delta_boxes": d.get("false_positive_delta_boxes", 0),
            "high_pred_boxes": d.get("high_boxes", 0),
            "improved_pred_boxes": d.get("improved_output_boxes", 0),
            "high_map50": high.get("map50"),
            "recovered_map50": rec.get("map50"),
            "improved_map50": imp.get("map50"),
            "delta_improved_vs_high": ev.get("delta_map50_vs_high"),
            "improved_map50_gain": (ev.get("delta_map50_vs_high") or 0) > 0,
        })
    rows.sort(key=lambda r: r["sequence"])
    improved = [r["sequence"] for r in rows if r.get("improved_map50_gain")]
    out = Path(args.output_csv)
    fields = list(rows[0].keys()) if rows else []
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"Wrote {len(rows)} rows to {out}")
    print(f"Sequences with improved mAP@0.5 (filtered): {len(improved)}/{len(rows)}")
    if improved:
        print(", ".join(improved))


if __name__ == "__main__":
    main()
