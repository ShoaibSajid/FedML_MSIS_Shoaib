#!/usr/bin/env python3
"""Inference -> recovery -> per-sequence mAP for one BDD val sequence."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import shutil
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path

ROOT = Path("/Shoaib/FedML_MSIS_Shoaib/python/app/fedcv/object_detection")
RECOVERY = ROOT / "Yolov5_DeepSORT_PseudoLabels"
YOLO = RECOVERY / "yolov5"
SCRIPTS = ROOT / "scripts"

LC = "0.1"
HC = "0.5"
IOU = "0.6"
TRACK_MATCH_IOU = 0.7
DS = "resnet50_MSMT17"
DATA = "data/bdd_b1c66a42.yaml"

WEIGHTS = {
    "v8s": "/Shoaib/BDD100k_yolov8s/full_v8s/weights/best.pt",
    "v8m": "/Shoaib/BDD100k_yolv8m/full_v8m/weights/best.pt",
}


def sequence_artifacts_dir(output_root: Path, seq_name: str, model_tag: str) -> Path:
    return output_root / seq_name / model_tag / "artifacts"


def pipeline_config(recovery_mode: str) -> dict:
    return {
        "recovery_mode": recovery_mode,
        "low_conf": float(LC),
        "high_conf": float(HC),
        "iou_nms": float(IOU),
        "track_match_iou": TRACK_MATCH_IOU,
        "deepsort_model": DS,
    }


def run(cmd: list[str], cwd: Path | None = None, env: dict | None = None) -> None:
    print(">>", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, env=env, check=True)


def resolve_device(requested: str) -> str:
    if os.environ.get("CUDA_VISIBLE_DEVICES", "").strip() != "":
        return "0"
    return requested


def link_images_for_recovery(seq_dir: Path, label_dir: Path) -> int:
    label_dir.mkdir(parents=True, exist_ok=True)
    images = sorted(seq_dir.glob("*.jpg"))
    for src in images:
        dst = label_dir / src.name
        if dst.is_symlink() and dst.resolve() == src.resolve():
            continue
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        os.symlink(src.resolve(), dst)
    have = len(list(label_dir.glob("*.jpg")))
    if have != len(images):
        raise RuntimeError(
            f"Image link incomplete for {label_dir.name}: {have}/{len(images)}"
        )
    return have


@contextmanager
def sequence_lock(lock_path: Path):
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w") as lockf:
        try:
            fcntl.flock(lockf, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(f"Sequence already running (lock: {lock_path})")
        lockf.write(str(os.getpid()))
        lockf.flush()
        try:
            yield
        finally:
            fcntl.flock(lockf, fcntl.LOCK_UN)


def parse_log(path: Path) -> dict:
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
    images, gt_boxes, p, r, map50, map95 = m.groups()
    p, r, map50, map95 = float(p), float(r), float(map50), float(map95)
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0

    pred_boxes = None
    counts = re.findall(
        r"^\s*(\d+)\s+\d+\s+\d+\s+(\d+)\s+(\d+)\s*$",
        text,
        re.MULTILINE,
    )
    if counts:
        pred_boxes = int(counts[-1][2])

    return {
        "images": int(images),
        "gt_boxes": int(gt_boxes),
        "pred_boxes": pred_boxes,
        "precision": p,
        "recall": r,
        "f1": f1,
        "map50": map50,
        "map50_95": map95,
    }


def write_summary(
    report: Path,
    seq_name: str,
    model_tag: str,
    recovery_mode: str,
    recovered_source: str,
) -> dict:
    generated = parse_log(report / f"metrics_{seq_name}-YOLO.log")
    recovered = parse_log(report / f"metrics_{seq_name}-Recovered.log")
    improvement: dict = {}
    if generated and recovered:
        for k in ("precision", "recall", "f1", "map50", "map50_95"):
            improvement[k] = recovered[k] - generated[k]
        if generated.get("pred_boxes") is not None and recovered.get("pred_boxes") is not None:
            improvement["pred_boxes"] = recovered["pred_boxes"] - generated["pred_boxes"]
    out = {
        "sequence": seq_name,
        "model": model_tag,
        "config": pipeline_config(recovery_mode),
        "recovered_eval_source": recovered_source,
        "generated_pseudo_labels": generated,
        "recovered_pseudo_labels": recovered,
        "improvement_recovered_minus_generated": improvement,
        "improved_map50": improvement.get("map50", 0) > 0,
    }
    (report / "summary.json").write_text(json.dumps(out, indent=2))
    return out


def summary_is_current(summary_path: Path, recovery_mode: str, model_tag: str) -> bool:
    if not summary_path.exists():
        return False
    data = json.loads(summary_path.read_text())
    cfg = data.get("config") or {}
    return (
        data.get("model") == model_tag
        and cfg.get("recovery_mode") == recovery_mode
        and "gt_boxes" in (data.get("generated_pseudo_labels") or {})
    )


def run_sequence(
    seq_dir: Path,
    model_tag: str,
    output_root: Path,
    device: str = "0",
    skip_if_done: bool = False,
    force: bool = False,
    recovery_mode: str = "forward",
    weights: str | None = None,
) -> dict | None:
    seq_name = seq_dir.name
    out = sequence_artifacts_dir(output_root, seq_name, model_tag)
    report = out / "report"
    summary_path = report / "summary.json"
    lock_path = out / ".running.lock"
    weights = weights or WEIGHTS[model_tag]
    device = resolve_device(device)

    if skip_if_done and not force and summary_is_current(summary_path, recovery_mode, model_tag):
        print(f"Skip {seq_name}/{model_tag}: summary exists ({recovery_mode})")
        return json.loads(summary_path.read_text())

    gt_3class = out / "ground_truth_3class"
    lc_labels = out / f"pseudo_low_{LC}"
    hc_labels = out / f"pseudo_high_{HC}"
    fw_labels = out / "Recover-FW"
    bw_labels = out / "Recover-BW"
    merged = out / "Recover-Merged"

    try:
        with sequence_lock(lock_path):
            _run_pipeline(
                seq_dir, seq_name, model_tag, out, report, gt_3class,
                lc_labels, hc_labels, fw_labels, bw_labels, merged,
                weights, device, recovery_mode,
            )
    except RuntimeError as exc:
        if "already running" in str(exc):
            print(f"Skip {seq_name}/{model_tag}: {exc}")
            return None
        raise

    return json.loads(summary_path.read_text())


def _export_improved_labels(out: Path, seq_name: str, recovery_mode: str, device: str) -> None:
    recovered_dir = "Recover-Merged" if recovery_mode == "full" else "Recover-FW"
    export_script = SCRIPTS / "export_recovered_improved_labels.py"
    run(
        [
            "conda", "run", "-n", "shoaib", "python", str(export_script),
            "--artifacts-dir", str(out),
            "--recovered-dir", recovered_dir,
            "--device", device,
        ]
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--weights", default="")
    p.add_argument("--seq-dir", required=True)
    p.add_argument("--output-root", required=True)
    p.add_argument("--model-tag", required=True, choices=["v8s", "v8m"])
    p.add_argument("--recovery", default="forward", choices=["forward", "full"])
    p.add_argument("--device", default="0")
    p.add_argument("--skip-if-done", action="store_true")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    weights = args.weights or WEIGHTS[args.model_tag]
    summary = run_sequence(
        Path(args.seq_dir),
        args.model_tag,
        Path(args.output_root),
        device=args.device,
        skip_if_done=args.skip_if_done,
        force=args.force,
        recovery_mode=args.recovery,
        weights=weights,
    )
    if summary:
        print(json.dumps(summary, indent=2))


def _run_pipeline(
    seq_dir: Path,
    seq_name: str,
    model_tag: str,
    out: Path,
    report: Path,
    gt_3class: Path,
    lc_labels: Path,
    hc_labels: Path,
    fw_labels: Path,
    bw_labels: Path,
    merged: Path,
    weights: str,
    device: str,
    recovery_mode: str,
) -> None:
    report.mkdir(parents=True, exist_ok=True)
    gt_3class.mkdir(parents=True, exist_ok=True)

    run(
        [
            "conda", "run", "-n", "shoaib", "python", str(SCRIPTS / "remap_gt_3class.py"),
            "--src", str(seq_dir), "--dst", str(gt_3class), "--link-images",
        ]
    )

    image_list = sorted(seq_dir.glob("*.jpg"))
    list_file = out / "image_list.txt"
    list_file.write_text("\n".join(str(x) for x in image_list) + "\n")
    shutil.copy2(list_file, report / "image_list.txt")

    py = ["conda", "run", "-n", "shoaib", "python", str(SCRIPTS / "yolov8_infer_sequence.py")]
    for conf, dst in ((LC, lc_labels), (HC, hc_labels)):
        if dst.exists():
            shutil.rmtree(dst)
        dst.mkdir(parents=True, exist_ok=True)
        run(
            py + [
                "--weights", weights,
                "--seq-dir", str(seq_dir),
                "--out-dir", str(dst),
                "--conf", conf,
                "--iou", IOU,
                "--device", device,
            ]
        )

    n_images = link_images_for_recovery(seq_dir, lc_labels)
    print(f"Linked {n_images} images into {lc_labels} for recovery", flush=True)

    env = os.environ.copy()
    env["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"

    track = [
        "conda", "run", "-n", "shoaib", "python", "trackv2_from_file.py",
        "--source", str(lc_labels),
        "--conf-thres", HC,
        "--low-conf-thres", LC,
        "--high-conf-thres", HC,
        "--iou-thres", IOU,
        "--save-txt", "--exist-ok",
        "--deep_sort_model", DS,
        "--yolo_model", weights,
        "--device", device,
    ]

    if fw_labels.exists():
        shutil.rmtree(fw_labels)
    run(track + ["--output", str(fw_labels)], cwd=RECOVERY, env=env)

    if recovery_mode == "full":
        if bw_labels.exists():
            shutil.rmtree(bw_labels)
        run(track + ["--output", str(bw_labels), "--reverse"], cwd=RECOVERY, env=env)
        if merged.exists():
            shutil.rmtree(merged)
        run(
            [
                "conda", "run", "-n", "shoaib", "python", "merge_forward_backward_v2.py",
                "--forward", str(fw_labels),
                "--backward", str(bw_labels),
                "--merged", str(merged),
            ],
            cwd=RECOVERY,
            env=env,
        )
        link_images_for_recovery(seq_dir, merged)
        recovered_preds = merged
        recovered_source = "Recover-Merged"
    else:
        recovered_preds = fw_labels
        recovered_source = "Recover-FW"

    val_cmd_base = [
        "conda", "run", "-n", "shoaib", "python", "val_v3.py",
        "--data", DATA,
        "--gt", str(gt_3class),
        "--exist-ok",
        "--device", device,
    ]
    for preds, run_name in (
        (hc_labels, f"{seq_name}-YOLO"),
        (recovered_preds, f"{seq_name}-Recovered"),
    ):
        log_path = report / f"metrics_{run_name}.log"
        proc = subprocess.run(
            val_cmd_base + ["--name", run_name, "--saved_predictions", str(preds)],
            cwd=YOLO,
            env=env,
            capture_output=True,
            text=True,
        )
        log_path.write_text(proc.stdout + proc.stderr)
        if proc.returncode != 0:
            print(proc.stdout)
            print(proc.stderr, file=sys.stderr)
            raise subprocess.CalledProcessError(proc.returncode, proc.args)

    summary = write_summary(report, seq_name, model_tag, recovery_mode, recovered_source)
    print(json.dumps(summary, indent=2))

    _export_improved_labels(out, seq_name, recovery_mode, device)


if __name__ == "__main__":
    main()
