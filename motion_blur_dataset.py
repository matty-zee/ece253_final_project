import argparse
from pathlib import Path
import shutil

import cv2
import numpy as np


def build_motion_kernel(length: int, angle: float, width: int = 1) -> np.ndarray:
    """Create a 2D motion blur PSF (point spread function)."""
    length = max(1, int(length))
    width = max(1, int(width))
    kernel = np.zeros((length, length), dtype=np.float32)
    cv2.line(
        kernel,
        (0, length // 2),
        (length - 1, length // 2),
        color=1.0,
        thickness=width,
    )
    center = (length - 1) / 2.0
    rot = cv2.getRotationMatrix2D((center, center), angle, 1.0)
    rotated = cv2.warpAffine(kernel, rot, (length, length))
    rotated_sum = rotated.sum()
    if rotated_sum > 0:
        rotated /= rotated_sum
    return rotated


def apply_motion_blur(image_path: Path, kernel: np.ndarray, dst_path: Path) -> None:
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Failed to read image: {image_path}")
    blurred = cv2.filter2D(img, ddepth=-1, kernel=kernel, borderType=cv2.BORDER_REPLICATE)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dst_path), blurred)


def process_dataset(src: Path, dst: Path, kernel: np.ndarray, max_count: int | None = None) -> None:
    files = sorted(src.glob("*_img[12].ppm"))
    if max_count:
        files = files[:max_count * 2]  # two images per pair
    total = len(files)
    copied_flows = 0
    for idx, img_path in enumerate(files, 1):
        rel = img_path.relative_to(src)
        out_img = dst / rel
        apply_motion_blur(img_path, kernel, out_img)

        base = img_path.name.split("_img")[0]
        flow_src = src / f"{base}_flow.flo"
        flow_dst = dst / f"{base}_flow.flo"
        if flow_src.exists() and not flow_dst.exists():
            flow_dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(flow_src, flow_dst)
            copied_flows += 1

        if total > 0 and idx % max(1, total // 10) == 0:
            print(f"Blurred {idx}/{total} images ({idx / total:.0%})")
    print(f"Done. Blurred {total} images, copied {copied_flows} flow files.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply synthetic motion blur to a FlyingChairs-style dataset.")
    parser.add_argument("--src", type=Path, required=True, help="Source dataset directory (with *_img1.ppm, *_img2.ppm, *_flow.flo).")
    parser.add_argument("--dst", type=Path, required=True, help="Destination dataset directory.")
    parser.add_argument("--length", type=int, default=15, help="Kernel length in pixels.")
    parser.add_argument("--angle", type=float, default=15.0, help="Kernel angle in degrees.")
    parser.add_argument("--width", type=int, default=1, help="Kernel thickness.")
    parser.add_argument("--max-count", type=int, default=None,
                        help="Optional number of pairs to process (for quick sweeps).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    kernel = build_motion_kernel(args.length, args.angle, args.width)
    process_dataset(args.src, args.dst, kernel, max_count=args.max_count)


if __name__ == "__main__":
    main()
