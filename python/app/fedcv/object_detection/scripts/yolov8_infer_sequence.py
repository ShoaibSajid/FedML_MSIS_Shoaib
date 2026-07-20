#!/usr/bin/env python3
"""Run YOLOv8 inference on a BDD sequence and save 3-class YOLO txt labels."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from ultralytics import YOLO

BDD_TO_3CLASS = {
    0: 0, 1: 0, 2: 0, 3: 0, 5: 0, 6: 0, 11: 0,
    7: 1,
    8: 2, 12: 2,
}


def remap_and_save(result, out_txt: Path) -> int:
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        out_txt.write_text("")
        return 0

    xywhn = boxes.xywhn.cpu().numpy()
    confs = boxes.conf.cpu().numpy()
    clss = boxes.cls.cpu().numpy().astype(int)

    lines = []
    for (x, y, w, h), conf, cls in zip(xywhn, confs, clss):
        if cls not in BDD_TO_3CLASS:
            continue
        lines.append(
            f"{BDD_TO_3CLASS[cls]} {x:.6f} {y:.6f} {w:.6f} {h:.6f} {conf:.6f}"
        )
    out_txt.write_text("\n".join(lines) + ("\n" if lines else ""))
    return len(lines)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True)
    p.add_argument("--seq-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--conf", type=float, required=True)
    p.add_argument("--iou", type=float, default=0.6)
    p.add_argument("--device", default="0")
    p.add_argument("--batch-size", type=int, default=8)
    args = p.parse_args()

    seq_dir = Path(args.seq_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    images = sorted(seq_dir.glob("*.jpg"))
    if not images:
        raise SystemExit(f"No images found in {seq_dir}")

    model = YOLO(args.weights)
    total = 0
    results = model.predict(
        source=[str(img) for img in images],
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        verbose=False,
        save=False,
        stream=True,
        batch=args.batch_size,
    )
    for img, result in zip(images, results):
        total += remap_and_save(result, out_dir / f"{img.stem}.txt")

    print(f"Saved {len(images)} label files ({total} boxes) to {out_dir}")


if __name__ == "__main__":
    main()
