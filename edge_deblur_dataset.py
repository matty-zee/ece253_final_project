import argparse
from pathlib import Path
import shutil

import cv2
import numpy as np


def edge_enhance(img: np.ndarray, amount: float, bilateral_d: int, sigma_color: float, sigma_space: float) -> np.ndarray:
    """Edge-aware sharpening using bilateral smoothing + Laplacian boost."""
    smooth = cv2.bilateralFilter(img, d=bilateral_d, sigmaColor=sigma_color, sigmaSpace=sigma_space)
    lap = cv2.Laplacian(smooth, cv2.CV_32F, ksize=3)
    sharpened = img.astype(np.float32) + amount * lap
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def process_dataset(src: Path, dst: Path, amount: float, bilateral_d: int, sigma_color: float, sigma_space: float, max_count: int | None) -> None:
    files = sorted(src.glob("*_img[12].ppm"))
    if max_count:
        files = files[: max_count * 2]
    total = len(files)
    copied_flows = 0
    for idx, img_path in enumerate(files, 1):
        rel = img_path.relative_to(src)
        out_img = dst / rel
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"Failed to read image: {img_path}")
        out_img.parent.mkdir(parents=True, exist_ok=True)
        enhanced = edge_enhance(img, amount, bilateral_d, sigma_color, sigma_space)
        cv2.imwrite(str(out_img), enhanced)

        base = img_path.name.split("_img")[0]
        flow_src = src / f"{base}_flow.flo"
        flow_dst = dst / f"{base}_flow.flo"
        if flow_src.exists() and not flow_dst.exists():
            flow_dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(flow_src, flow_dst)
            copied_flows += 1

        if total > 0 and idx % max(1, total // 10) == 0:
            print(f"Edge-deblurred {idx}/{total} images ({idx / total:.0%})")
    print(f"Done. Edge-deblurred {total} images, copied {copied_flows} flow files.")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Edge-based deblurring for FlyingChairs-style datasets.")
    p.add_argument("--src", type=Path, required=True, help="Source dataset directory (blurred).")
    p.add_argument("--dst", type=Path, required=True, help="Destination dataset directory.")
    p.add_argument("--amount", type=float, default=1.0, help="Edge boost factor.")
    p.add_argument("--bilateral-d", type=int, default=9, help="Bilateral filter diameter.")
    p.add_argument("--sigma-color", type=float, default=75.0, help="Bilateral sigmaColor.")
    p.add_argument("--sigma-space", type=float, default=75.0, help="Bilateral sigmaSpace.")
    p.add_argument("--max-count", type=int, default=None, help="Optional number of pairs to process.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    process_dataset(
        args.src,
        args.dst,
        amount=args.amount,
        bilateral_d=args.bilateral_d,
        sigma_color=args.sigma_color,
        sigma_space=args.sigma_space,
        max_count=args.max_count,
    )


if __name__ == "__main__":
    main()
