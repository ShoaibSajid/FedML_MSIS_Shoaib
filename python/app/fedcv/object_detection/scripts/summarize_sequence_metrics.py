#!/usr/bin/env python3
"""Parse val_v3 metric logs and write improvement summary."""
import argparse
import json
import re
from pathlib import Path


def parse_log(path: Path) -> dict:
    text = path.read_text()
    m = re.search(
        r"^\s*all\s+(\d+)\s+(\d+)\s+([\d.eE+-]+)\s+([\d.eE+-]+)\s+([\d.eE+-]+)\s+([\d.eE+-]+)",
        text,
        re.MULTILINE,
    )
    if not m:
        return {}
    images, labels, p, r, map50, map = m.groups()
    p, r, map50, map = float(p), float(r), float(map50), float(map)
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return {
        "images": int(images),
        "labels": int(labels),
        "precision": p,
        "recall": r,
        "f1": f1,
        "map50": map50,
        "map50_95": map,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--report-dir", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    report = Path(args.report_dir)
    generated = parse_log(report / "metrics_b1c66a42-6f7d68ca-YOLO.log")
    recovered = parse_log(report / "metrics_b1c66a42-6f7d68ca-Recovered.log")
    improvement = {}
    if generated and recovered:
        for k in ("precision", "recall", "f1", "map50", "map50_95"):
            improvement[k] = recovered[k] - generated[k]
    out = {
        "generated_pseudo_labels": generated,
        "recovered_pseudo_labels": recovered,
        "improvement_recovered_minus_generated": improvement,
    }
    Path(args.output).write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
