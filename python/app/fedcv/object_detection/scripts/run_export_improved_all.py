#!/usr/bin/env python3
"""Run export_recovered_improved_labels on all ready artifacts under runs/val_eval."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path("/Shoaib/FedML_MSIS_Shoaib/python/app/fedcv/object_detection")
VAL_EVAL = ROOT / "runs" / "val_eval"
EXPORT_SCRIPT = ROOT / "scripts" / "export_recovered_improved_labels.py"
AGG_SCRIPT = ROOT / "scripts" / "aggregate_improved_labels.py"


def load_export_fn():
    spec = importlib.util.spec_from_file_location("export_improved", EXPORT_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.export_improved_labels


def artifact_score(art: Path) -> int:
    """Higher = preferred when deduping (seq, model)."""
    parts = art.parts
    if art.name == "artifacts" and any(p in parts for p in ("v8s", "v8m")):
        return 3
    if "full_v8s" in parts or "full_v8m" in parts:
        return 1
    return 2


def is_ready_artifacts(art: Path) -> bool:
    gt = art / "ground_truth_3class"
    high = art / "pseudo_high_0.5"
    if not gt.is_dir() or not high.is_dir():
        return False
    if not any(gt.glob("*.txt")):
        return False
    return True


def discover_artifacts(results_root: Path) -> list[dict]:
    found: dict[tuple[str, str], dict] = {}
    for fw in sorted(results_root.rglob("Recover-FW")):
        if not fw.is_dir():
            continue
        art = fw.parent
        if not is_ready_artifacts(art):
            continue
        parts = art.parts
        model = "unknown"
        if "v8s" in parts:
            model = "v8s"
        elif "v8m" in parts:
            model = "v8m"
        elif "full_v8s" in parts:
            model = "v8s"
        elif "full_v8m" in parts:
            model = "v8m"
        seq = art.parent.name if art.name == "artifacts" else art.name
        if seq in ("full_v8s", "full_v8m", "v8s", "v8m", "artifacts"):
            seq = art.parent.parent.name
        key = (seq, model)
        item = {"artifacts_dir": art, "sequence": seq, "model": model}
        prev = found.get(key)
        if prev is None or artifact_score(art) > artifact_score(prev["artifacts_dir"]):
            found[key] = item
    # also Recover-Merged without Recover-FW (full recovery only)
    for merged in sorted(results_root.rglob("Recover-Merged")):
        if not merged.is_dir():
            continue
        art = merged.parent
        if (art / "Recover-FW").is_dir():
            continue
        if not is_ready_artifacts(art):
            continue
        parts = art.parts
        model = "v8s" if "v8s" in parts or "full_v8s" in parts else (
            "v8m" if "v8m" in parts or "full_v8m" in parts else "unknown"
        )
        seq = art.parent.name if art.name == "artifacts" else art.name
        if seq in ("full_v8s", "full_v8m", "v8s", "v8m", "artifacts"):
            seq = art.parent.parent.name
        key = (seq, model)
        if key in found:
            continue
        found[key] = {
            "artifacts_dir": art,
            "sequence": seq,
            "model": model,
            "recovered_dir": "Recover-Merged",
        }
    return sorted(found.values(), key=lambda x: (x["model"], x["sequence"], str(x["artifacts_dir"])))


def needs_run(art: Path, force: bool) -> bool:
    analysis = art / "report" / "delta_analysis.json"
    if force or not analysis.exists():
        return True
    if not (art / "recovered_improved_labels").is_dir():
        return True
    return False


def aggregate_all(results_root: Path) -> None:
    rows = []
    for analysis in sorted(results_root.rglob("report/delta_analysis.json")):
        d = json.loads(analysis.read_text())
        ev = d.get("eval") or {}
        high = ev.get("generated_high_conf") or {}
        rec = ev.get("recovered_forward") or {}
        imp = ev.get("recovered_improved_labels") or {}
        art = analysis.parent.parent
        parts = art.parts
        model = "v8s" if "v8s" in parts or "full_v8s" in parts else (
            "v8m" if "v8m" in parts or "full_v8m" in parts else "unknown"
        )
        seq = d.get("artifacts_dir", "").split("/")
        sequence = art.parent.parent.name if art.name == "artifacts" else art.parent.name
        if "val_eval" in parts:
            idx = parts.index("val_eval") + 1
            sequence = parts[idx] if parts[idx] not in ("full_v8s", "full_v8m") else parts[idx + 1]
        rows.append({
            "sequence": sequence,
            "model": model,
            "artifacts_dir": str(art),
            "helpful_delta_boxes": d.get("helpful_delta_boxes", 0),
            "fp_delta_boxes": d.get("false_positive_delta_boxes", 0),
            "delta_boxes": d.get("delta_boxes", 0),
            "high_pred_boxes": d.get("high_boxes", 0),
            "improved_pred_boxes": d.get("improved_output_boxes", 0),
            "high_map50": high.get("map50"),
            "recovered_map50": rec.get("map50"),
            "improved_map50": imp.get("map50"),
            "delta_improved_vs_high": ev.get("delta_map50_vs_high"),
            "improved_map50_gain": (ev.get("delta_map50_vs_high") or 0) > 0,
        })
    rows = [r for r in rows if r["improved_map50"] is not None or r["helpful_delta_boxes"]]
    if not rows:
        print("No delta_analysis.json files found.")
        return
    import csv
    out = results_root / "summary_improved_all.csv"
    fields = list(rows[0].keys())
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in sorted(rows, key=lambda x: (x["model"], x["sequence"])):
            w.writerow(r)
    improved = [r for r in rows if r["improved_map50_gain"]]
    print(f"\nWrote {len(rows)} rows -> {out}")
    print(f"Improved mAP@0.5 vs high-conf: {len(improved)}/{len(rows)}")
    by_model: dict[str, list] = {}
    for r in rows:
        by_model.setdefault(r["model"], []).append(r)
    for model, mr in by_model.items():
        imp_m = [x for x in mr if x["improved_map50_gain"]]
        print(f"  [{model}] {len(imp_m)}/{len(mr)} improved")
        if imp_m:
            print(f"    {', '.join(x['sequence'] for x in imp_m)}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results-root", default=str(VAL_EVAL))
    p.add_argument("--model", default="", help="Filter: v8s or v8m")
    p.add_argument("--force", action="store_true")
    p.add_argument("--no-eval", action="store_true", help="Skip val_v3 (faster)")
    p.add_argument("--device", default=os.environ.get("DEVICE", "0"))
    args = p.parse_args()

    results_root = Path(args.results_root)
    export_fn = load_export_fn()
    candidates = discover_artifacts(results_root)
    if args.model:
        candidates = [c for c in candidates if c["model"] == args.model]

    print(f"Found {len(candidates)} artifact dir(s) with recovery outputs", flush=True)
    ok = fail = skip = 0
    for item in candidates:
        art = item["artifacts_dir"]
        if not needs_run(art, args.force):
            print(f"Skip {item['sequence']} ({item['model']}): already done", flush=True)
            skip += 1
            continue
        recovered_dir = item.get("recovered_dir", "Recover-FW")
        print(f"\n{'='*60}\n{item['sequence']} / {item['model']}\n{art}\n{'='*60}", flush=True)
        try:
            stats = export_fn(
                art,
                recovered_dir_name=recovered_dir,
                run_eval=not args.no_eval,
                device=args.device,
            )
            delta = stats.get("improved_map50_vs_high")
            tag = "IMPROVED" if (delta or 0) > 0 else "no gain"
            delta_s = f"{delta:+.4f}" if delta is not None else "n/a"
            print(
                f"  helpful={stats.get('helpful_delta_boxes')}  "
                f"improved_map50={stats.get('improved_map50')}  "
                f"delta_vs_high={delta_s}  {tag}",
                flush=True,
            )
            ok += 1
        except Exception as exc:
            print(f"FAILED {item['sequence']}: {exc}", file=sys.stderr, flush=True)
            fail += 1

    aggregate_all(results_root)
    print(f"\nDone: ok={ok} skip={skip} fail={fail}", flush=True)
    if fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
