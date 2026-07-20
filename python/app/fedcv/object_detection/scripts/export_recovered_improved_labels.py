#!/usr/bin/env python3
"""
Export recovered labels that improve recall over high-conf baseline.

Pipeline per frame:
  1. Delta = recovered boxes with no same-class IoU overlap vs high-conf preds.
  2. Match high-conf preds to GT (greedy, IoU@0.5).
  3. Keep delta boxes that match GT objects NOT already matched by high-conf.
  4. Write recovered_improved_labels/ = high-conf + helpful delta boxes.

Also writes recovered_delta_only/ (helpful delta only) and report/delta_analysis.json.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from pathlib import Path

import numpy as np

ROOT = Path("/Shoaib/FedML_MSIS_Shoaib/python/app/fedcv/object_detection")
YOLO = ROOT / "Yolov5_DeepSORT_PseudoLabels" / "yolov5"
DATA = "data/bdd_b1c66a42.yaml"
IOU_GT = 0.5
IOU_DELTA = 0.5


def load_yolo_lines(path: Path) -> list[list[float]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        parts = line.split()
        cls = int(float(parts[0]))
        x, y, w, h = map(float, parts[1:5])
        conf = float(parts[5]) if len(parts) >= 6 else 0.51
        rows.append([cls, x, y, w, h, conf])
    return rows


def xywhn_to_xyxy(cls: int, x: float, y: float, w: float, h: float, w_img: int, h_img: int):
    xc, yc = x * w_img, y * h_img
    bw, bh = w * w_img, h * h_img
    return cls, xc - bw / 2, yc - bh / 2, xc + bw / 2, yc + bh / 2


def box_iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a[1], a[2], a[3], a[4]
    bx1, by1, bx2, by2 = b[1], b[2], b[3], b[4]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / (area_a + area_b - inter)


def to_xyxy_boxes(lines: list[list[float]], w_img: int, h_img: int) -> list[tuple]:
    return [xywhn_to_xyxy(int(r[0]), r[1], r[2], r[3], r[4], w_img, h_img) for r in lines]


def format_yolo_line(row: list[float], with_conf: bool = True) -> str:
    cls, x, y, w, h = row[0], row[1], row[2], row[3], row[4]
    conf = row[5] if len(row) >= 6 else 0.51
    if with_conf:
        return f"{int(cls)} {x:.6f} {y:.6f} {w:.6f} {h:.6f} {conf:.6f}"
    return f"{int(cls)} {x:.6f} {y:.6f} {w:.6f} {h:.6f}"


def overlaps_high(rec_box, high_xyxy: list[tuple], iou_thresh: float) -> bool:
    for hb in high_xyxy:
        if rec_box[0] != hb[0]:
            continue
        if box_iou(rec_box, hb) >= iou_thresh:
            return True
    return False


def match_preds_to_gt(preds_xyxy: list[tuple], gt_xyxy: list[tuple], iou_thresh: float) -> set[int]:
    """Return set of matched GT indices (greedy by IoU)."""
    if not preds_xyxy or not gt_xyxy:
        return set()
    pairs = []
    for pi, p in enumerate(preds_xyxy):
        for gi, g in enumerate(gt_xyxy):
            if p[0] != g[0]:
                continue
            iou = box_iou(p, g)
            if iou >= iou_thresh:
                pairs.append((iou, pi, gi))
    pairs.sort(reverse=True)
    matched_gt: set[int] = set()
    used_pred: set[int] = set()
    for iou, pi, gi in pairs:
        if pi in used_pred or gi in matched_gt:
            continue
        used_pred.add(pi)
        matched_gt.add(gi)
    return matched_gt


def link_image(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_symlink() and dst.resolve() == src.resolve():
        return
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    os.symlink(src.resolve(), dst)


def parse_val_log(path: Path) -> dict:
    if not path.exists():
        return {}
    text = path.read_text()
    m = re.search(
        r"^\s*all\s+(\d+)\s+(\d+)\s+([\d.eE+-]+)\s+([\d.eE+-]+)\s+([\d.eE+-]+)\s+([\d.eE+-]+)",
        text,
        re.MULTILINE,
    )
    if not m:
        return {}
    images, gt, p, r, map50, map95 = m.groups()
    pred_boxes = None
    counts = re.findall(r"^\s*(\d+)\s+\d+\s+\d+\s+(\d+)\s+(\d+)\s*$", text, re.MULTILINE)
    if counts:
        pred_boxes = int(counts[-1][2])
    return {
        "images": int(images),
        "gt_boxes": int(gt),
        "pred_boxes": pred_boxes,
        "precision": float(p),
        "recall": float(r),
        "map50": float(map50),
        "map50_95": float(map95),
    }


def run_map(artifacts: Path, seq_name: str, preds_dir: Path, run_name: str, device: str) -> dict:
    gt_dir = artifacts / "ground_truth_3class"
    report = artifacts / "report"
    env = os.environ.copy()
    env["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
    log_path = report / f"metrics_{run_name}.log"
    cmd = [
        "conda", "run", "-n", "shoaib", "python", "val_v3.py",
        "--data", DATA,
        "--name", run_name,
        "--gt", str(gt_dir),
        "--saved_predictions", str(preds_dir),
        "--exist-ok",
        "--device", device,
    ]
    proc = subprocess.run(cmd, cwd=YOLO, env=env, capture_output=True, text=True)
    log_path.write_text(proc.stdout + proc.stderr)
    if proc.returncode != 0:
        raise RuntimeError(f"val_v3 failed: {log_path}")
    return parse_val_log(log_path)


def infer_seq_name(artifacts: Path) -> str:
    if artifacts.name == "artifacts":
        return artifacts.parent.parent.name
    if artifacts.parent.name in ("full_v8s", "full_v8m"):
        return artifacts.name
    return artifacts.name


def export_improved_labels(
    artifacts_dir: Path,
    recovered_dir_name: str = "Recover-FW",
    high_dir_name: str = "pseudo_high_0.5",
    output_name: str = "recovered_improved_labels",
    iou_delta: float = IOU_DELTA,
    iou_gt: float = IOU_GT,
    run_eval: bool = True,
    device: str = "0",
) -> dict:
    artifacts = Path(artifacts_dir)
    gt_dir = artifacts / "ground_truth_3class"
    high_dir = artifacts / high_dir_name
    recovered_dir = artifacts / recovered_dir_name
    out_dir = artifacts / output_name
    delta_only_dir = artifacts / "recovered_delta_only"
    report_dir = artifacts / "report"

    if out_dir.exists():
        for p in out_dir.glob("*.txt"):
            p.unlink()
    else:
        out_dir.mkdir(parents=True, exist_ok=True)
    if delta_only_dir.exists():
        for p in delta_only_dir.glob("*.txt"):
            p.unlink()
    else:
        delta_only_dir.mkdir(parents=True, exist_ok=True)

    stats = {
        "artifacts_dir": str(artifacts),
        "high_conf_dir": high_dir_name,
        "recovered_dir": recovered_dir_name,
        "output_dir": output_name,
        "iou_delta_vs_high": iou_delta,
        "iou_gt_match": iou_gt,
        "frames": 0,
        "high_boxes": 0,
        "recovered_boxes": 0,
        "delta_boxes": 0,
        "helpful_delta_boxes": 0,
        "redundant_delta_boxes": 0,
        "false_positive_delta_boxes": 0,
        "improved_output_boxes": 0,
        "frames_with_helpful_delta": 0,
        "per_frame": [],
    }

    gt_txts = sorted(gt_dir.glob("*.txt"))
    seq_name = infer_seq_name(artifacts)

    for gt_txt in gt_txts:
        stem = gt_txt.stem
        jpg = gt_dir / f"{stem}.jpg"
        if not jpg.exists():
            jpg = gt_dir.parent / f"{stem}.jpg"
        if not jpg.exists():
            jpg = gt_dir.parent.parent / f"{stem}.jpg"
        if not jpg.exists():
            continue

        import cv2
        im = cv2.imread(str(jpg))
        if im is None:
            continue
        h_img, w_img = im.shape[:2]

        gt_lines = load_yolo_lines(gt_txt)
        high_lines = load_yolo_lines(high_dir / f"{stem}.txt")
        rec_lines = load_yolo_lines(recovered_dir / f"{stem}.txt")

        gt_xyxy = to_xyxy_boxes(gt_lines, w_img, h_img)
        high_xyxy = to_xyxy_boxes(high_lines, w_img, h_img)

        matched_gt_high = match_preds_to_gt(high_xyxy, gt_xyxy, iou_gt)
        unmatched_gt = set(range(len(gt_xyxy))) - matched_gt_high

        helpful_rows: list[list[float]] = []
        redundant = fp = delta_count = 0

        for row in rec_lines:
            rec_xyxy = xywhn_to_xyxy(int(row[0]), row[1], row[2], row[3], row[4], w_img, h_img)
            if overlaps_high(rec_xyxy, high_xyxy, iou_delta):
                continue
            delta_count += 1

            best_gi, best_iou = -1, 0.0
            for gi, g in enumerate(gt_xyxy):
                if rec_xyxy[0] != g[0]:
                    continue
                iou = box_iou(rec_xyxy, g)
                if iou > best_iou:
                    best_iou, best_gi = iou, gi

            if best_iou >= iou_gt and best_gi in unmatched_gt:
                helpful_rows.append(row)
                unmatched_gt.remove(best_gi)
            elif best_iou >= iou_gt:
                redundant += 1
            else:
                fp += 1

        improved_lines = [format_yolo_line(r) for r in high_lines] + [
            format_yolo_line(r) for r in helpful_rows
        ]
        delta_lines = [format_yolo_line(r) for r in helpful_rows]

        (out_dir / f"{stem}.txt").write_text("\n".join(improved_lines) + ("\n" if improved_lines else ""))
        (delta_only_dir / f"{stem}.txt").write_text("\n".join(delta_lines) + ("\n" if delta_lines else ""))
        link_image(jpg, out_dir / f"{stem}.jpg")

        frame_stat = {
            "frame": stem,
            "high": len(high_lines),
            "recovered": len(rec_lines),
            "delta": delta_count,
            "helpful": len(helpful_rows),
            "redundant_delta": redundant,
            "fp_delta": fp,
            "improved_total": len(improved_lines),
        }
        stats["per_frame"].append(frame_stat)
        stats["frames"] += 1
        stats["high_boxes"] += len(high_lines)
        stats["recovered_boxes"] += len(rec_lines)
        stats["delta_boxes"] += delta_count
        stats["helpful_delta_boxes"] += len(helpful_rows)
        stats["redundant_delta_boxes"] += redundant
        stats["false_positive_delta_boxes"] += fp
        stats["improved_output_boxes"] += len(improved_lines)
        if helpful_rows:
            stats["frames_with_helpful_delta"] += 1

    eval_metrics = {}
    if run_eval and stats["frames"] > 0:
        high_metrics = parse_val_log(report_dir / f"metrics_{seq_name}-YOLO.log")
        rec_metrics = parse_val_log(report_dir / f"metrics_{seq_name}-Recovered.log")
        improved_metrics = run_map(
            artifacts, seq_name, out_dir, f"{seq_name}-Improved", device
        )
        eval_metrics = {
            "generated_high_conf": high_metrics,
            "recovered_forward": rec_metrics,
            "recovered_improved_labels": improved_metrics,
            "delta_map50_vs_high": improved_metrics.get("map50", 0)
            - high_metrics.get("map50", 0),
            "delta_map50_vs_recovered": improved_metrics.get("map50", 0)
            - rec_metrics.get("map50", 0),
        }
        stats["improved_map50"] = improved_metrics.get("map50")
        stats["improved_map50_vs_high"] = eval_metrics["delta_map50_vs_high"]

    stats["eval"] = eval_metrics
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "delta_analysis.json").write_text(json.dumps(stats, indent=2))
    return stats


def main() -> None:
    p = argparse.ArgumentParser(description="Export recovered_improved_labels (high + helpful delta).")
    p.add_argument("--artifacts-dir", required=True, help=".../<seq>/v8s/artifacts")
    p.add_argument("--recovered-dir", default="Recover-FW")
    p.add_argument("--high-dir", default="pseudo_high_0.5")
    p.add_argument("--output-name", default="recovered_improved_labels")
    p.add_argument("--iou-delta", type=float, default=IOU_DELTA)
    p.add_argument("--iou-gt", type=float, default=IOU_GT)
    p.add_argument("--no-eval", action="store_true")
    p.add_argument("--device", default="0")
    args = p.parse_args()

    stats = export_improved_labels(
        Path(args.artifacts_dir),
        recovered_dir_name=args.recovered_dir,
        high_dir_name=args.high_dir,
        output_name=args.output_name,
        iou_delta=args.iou_delta,
        iou_gt=args.iou_gt,
        run_eval=not args.no_eval,
        device=args.device,
    )
    print(json.dumps({k: v for k, v in stats.items() if k != "per_frame"}, indent=2))


if __name__ == "__main__":
    main()
