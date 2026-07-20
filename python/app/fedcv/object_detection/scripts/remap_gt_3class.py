#!/usr/bin/env python3
"""Remap BDD 13-class YOLO GT labels to 3-class (vehicle, rider, pedestrian)."""
from pathlib import Path
import argparse
import os

BDD_TO_3CLASS = {
    0: 0, 1: 0, 2: 0, 3: 0, 5: 0, 6: 0, 11: 0,
    7: 1,
    8: 2, 12: 2,
}


def remap_file(src: Path, dst: Path) -> int:
    lines_out = []
    for line in src.read_text().splitlines():
        if not line.strip():
            continue
        parts = line.split()
        cls = int(float(parts[0]))
        if cls not in BDD_TO_3CLASS:
            continue
        parts[0] = str(BDD_TO_3CLASS[cls])
        lines_out.append(" ".join(parts))
    dst.write_text("\n".join(lines_out) + ("\n" if lines_out else ""))
    return len(lines_out)


def link_or_copy_image(src: Path, dst: Path, use_symlink: bool) -> None:
    if use_symlink:
        if dst.is_symlink() and dst.resolve() == src.resolve():
            return
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        os.symlink(src.resolve(), dst)
    else:
        dst.write_bytes(src.read_bytes())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True)
    p.add_argument("--dst", required=True)
    p.add_argument("--copy-images", action="store_true")
    p.add_argument("--link-images", action="store_true")
    args = p.parse_args()
    src, dst = Path(args.src), Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)
    use_symlink = args.link_images or not args.copy_images
    total = 0
    for txt in sorted(src.glob("*.txt")):
        n = remap_file(txt, dst / txt.name)
        total += n
        jpg = src / f"{txt.stem}.jpg"
        if jpg.exists() and (args.copy_images or args.link_images):
            link_or_copy_image(jpg, dst / jpg.name, use_symlink)
    mode = "symlinked" if use_symlink else "copied"
    print(f"Remapped {len(list(src.glob('*.txt')))} files, {total} boxes kept, images {mode}")


if __name__ == "__main__":
    main()
