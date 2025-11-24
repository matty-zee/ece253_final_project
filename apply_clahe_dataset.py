"""
Apply CLAHE to a FlyingChairs-style dataset (e.g., the darkened subset) and
write results to a new folder, copying .flo files unchanged.

Defaults assume input from Flyingchairs_100_dark/data and output to
FlyingChairs_CLAHE/data.

Usage:
    python apply_clahe_dataset.py --src Flyingchairs_100_dark/data --dst FlyingChairs_CLAHE/data
"""

import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np


def apply_clahe_color(img: np.ndarray, clip_limit: float = 2.0, tile_grid: int = 8) -> np.ndarray:
    """Apply CLAHE on the L channel in LAB space."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid, tile_grid))
    l2 = clahe.apply(l)
    lab2 = cv2.merge([l2, a, b])
    return cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)


def process_pair(base: str, src: Path, dst: Path, clip_limit: float, tile_grid: int):
    img1_path = src / f"{base}_img1.ppm"
    img2_path = src / f"{base}_img2.ppm"
    flow_path = src / f"{base}_flow.flo"

    if not img1_path.exists() or not img2_path.exists() or not flow_path.exists():
        print(f"Skipping {base}: missing files.")
        return

    img1 = cv2.imread(str(img1_path), cv2.IMREAD_COLOR)
    img2 = cv2.imread(str(img2_path), cv2.IMREAD_COLOR)
    if img1 is None or img2 is None:
        print(f"Skipping {base}: failed to read images.")
        return

    out1 = apply_clahe_color(img1, clip_limit=clip_limit, tile_grid=tile_grid)
    out2 = apply_clahe_color(img2, clip_limit=clip_limit, tile_grid=tile_grid)

    dst.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dst / img1_path.name), out1)
    cv2.imwrite(str(dst / img2_path.name), out2)
    shutil.copy2(flow_path, dst / flow_path.name)


def main():
    parser = argparse.ArgumentParser(description="Apply CLAHE to a FlyingChairs-format dataset.")
    parser.add_argument("--src", type=Path, default=Path("Flyingchairs_100_dark/data"),
                        help="Source directory containing *_img1.ppm, *_img2.ppm, *_flow.flo")
    parser.add_argument("--dst", type=Path, default=Path("FlyingChairs_100_CLAHE/data"),
                        help="Destination directory to write CLAHE-processed images and copied flows")
    parser.add_argument("--clip-limit", type=float, default=2.0, help="CLAHE clipLimit parameter")
    parser.add_argument("--tile-grid", type=int, default=8, help="CLAHE tileGridSize parameter (square)")
    args = parser.parse_args()

    img1_files = sorted(args.src.glob("*_img1.ppm"))
    print(f"Found {len(img1_files)} image pairs in {args.src}")

    for img1_path in img1_files:
        base = img1_path.stem.replace("_img1", "")
        process_pair(base, args.src, args.dst, args.clip_limit, args.tile_grid)

    print(f"Done. Output written to {args.dst}")


if __name__ == "__main__":
    main()
