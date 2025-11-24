"""
Generate a darkened FlyingChairs subset with per-pair vignettes.

For each image pair (img1/img2) in the source directory, this script applies
the same dark vignette to both images (to mimic a fixed illumination field),
with random center and bright-region size for each pair. Flow files are
copied untouched. Output is written to Flyingchairs_100_dark/data by default.

Usage:
    python make_dark_vignette.py --src FlyingChairs_100/data --dst Flyingchairs_100_dark/data --seed 42
"""

import argparse
import random
import shutil
from pathlib import Path

import cv2
import numpy as np


def build_vignette_mask(h: int, w: int, center: tuple[float, float], r_inner: float, r_outer: float,
                        min_factor: float = 0.2) -> np.ndarray:
    """Create a radial vignette mask in [min_factor, 1], soft falloff between r_inner and r_outer."""
    yy, xx = np.mgrid[0:h, 0:w]
    dist = np.sqrt((xx - center[0]) ** 2 + (yy - center[1]) ** 2)

    mask = np.ones((h, w), dtype=np.float32)
    ring = dist > r_inner
    mask[ring] = np.clip(1.0 - (dist[ring] - r_inner) / (r_outer - r_inner), 0.0, 1.0)
    mask = min_factor + (1.0 - min_factor) * mask
    return mask


def process_pair(base: str, src_dir: Path, dst_dir: Path, rng: random.Random,
                 frac_range=(0.3, 0.6), min_factor=0.2, falloff_scale=2.3):
    img1_path = src_dir / f"{base}_img1.ppm"
    img2_path = src_dir / f"{base}_img2.ppm"
    flow_path = src_dir / f"{base}_flow.flo"

    if not img1_path.exists() or not img2_path.exists() or not flow_path.exists():
        print(f"Skipping {base}: missing files.")
        return

    img1 = cv2.imread(str(img1_path), cv2.IMREAD_COLOR)
    img2 = cv2.imread(str(img2_path), cv2.IMREAD_COLOR)
    if img1 is None or img2 is None:
        print(f"Skipping {base}: failed to read images.")
        return

    h, w = img1.shape[:2]
    # Random fully-bright diameter fraction of min dimension.
    frac = rng.uniform(*frac_range)
    r_inner = 0.5 * frac * min(h, w)
    # Ensure bright region stays fully inside the image.
    cx_min, cx_max = r_inner, max(r_inner, w - r_inner)
    cy_min, cy_max = r_inner, max(r_inner, h - r_inner)
    cx = w * 0.5 if cx_max <= cx_min else rng.uniform(cx_min, cx_max)
    cy = h * 0.5 if cy_max <= cy_min else rng.uniform(cy_min, cy_max)
    # Outer radius controls falloff to darker region.
    r_outer = min(0.8 * np.hypot(h, w), max(r_inner * falloff_scale, r_inner + 1.0))

    mask = build_vignette_mask(h, w, (cx, cy), r_inner, r_outer, min_factor=min_factor)

    def apply(img):
        return np.clip(img.astype(np.float32) * mask[..., None], 0, 255).astype(np.uint8)

    out1 = apply(img1)
    out2 = apply(img2)

    dst_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dst_dir / img1_path.name), out1)
    cv2.imwrite(str(dst_dir / img2_path.name), out2)
    shutil.copy2(flow_path, dst_dir / flow_path.name)


def main():
    parser = argparse.ArgumentParser(description="Create darkened vignette FlyingChairs subset.")
    parser.add_argument("--src", type=Path, default=Path("FlyingChairs_100/data"),
                        help="Source data directory with *_img1.ppm, *_img2.ppm, *_flow.flo files.")
    parser.add_argument("--dst", type=Path, default=Path("Flyingchairs_100_dark/data"),
                        help="Output directory to write darkened files.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible vignettes.")
    parser.add_argument("--min-bright", type=float, default=0.2, help="Minimum brightness factor at darkest region.")
    parser.add_argument("--bright-fraction", type=float, nargs=2, default=(0.3, 0.6),
                        metavar=("MIN", "MAX"),
                        help="Range for fully bright diameter fraction of image size.")
    parser.add_argument("--falloff-scale", type=float, default=2.3,
                        help="Multiplier on inner radius to set where the vignette fully darkens (larger = gentler slope).")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    src_files = sorted(args.src.glob("*_img1.ppm"))
    print(f"Found {len(src_files)} pairs in {args.src}")

    for img1_path in src_files:
        base = img1_path.stem.replace("_img1", "")
        process_pair(base, args.src, args.dst, rng,
                     frac_range=tuple(args.bright_fraction),
                     min_factor=args.min_bright,
                     falloff_scale=args.falloff_scale)

    print(f"Done. Darkened pairs written to {args.dst}")


if __name__ == "__main__":
    main()
