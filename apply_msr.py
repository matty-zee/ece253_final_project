import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np


def msr(img: np.ndarray, scales: list[int]) -> np.ndarray:
    """Multi-Scale Retinex on BGR image."""
    img_f = img.astype(np.float32) + 1.0
    retinex = np.zeros_like(img_f)
    for sigma in scales:
        ksize = int(((sigma - 0.8) / 0.15) + 2.0)
        if ksize % 2 == 0:
            ksize += 1
        blur = cv2.GaussianBlur(img_f, (ksize, ksize), sigma)
        retinex += np.log(img_f) - np.log(blur + 1e-6)
    retinex /= len(scales)

    # Normalize to 0-255 per channel
    out = np.zeros_like(img_f)
    for c in range(3):
        channel = retinex[..., c]
        channel = (channel - channel.min()) / (channel.max() - channel.min() + 1e-6)
        out[..., c] = channel * 255.0
    return np.clip(out, 0, 255).astype(np.uint8)


def process_pair(base: str, src: Path, dst: Path, scales: list[int]):
    img1_path = src / f"{base}_img1.ppm"
    img2_path = src / f"{base}_img2.ppm"
    flow_path = src / f"{base}_flow.flo"

    if not (img1_path.exists() and img2_path.exists() and flow_path.exists()):
        print(f"Skipping {base}: missing files.")
        return

    img1 = cv2.imread(str(img1_path), cv2.IMREAD_COLOR)
    img2 = cv2.imread(str(img2_path), cv2.IMREAD_COLOR)
    if img1 is None or img2 is None:
        print(f"Skipping {base}: failed to read images.")
        return

    out1 = msr(img1, scales)
    out2 = msr(img2, scales)

    dst.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dst / img1_path.name), out1)
    cv2.imwrite(str(dst / img2_path.name), out2)
    shutil.copy2(flow_path, dst / flow_path.name)


def main():
    parser = argparse.ArgumentParser(description="Apply Multi-Scale Retinex to a FlyingChairs dataset.")
    parser.add_argument("--src", type=Path, default=Path("Dataset/FlyingChairs_100/data"))
    parser.add_argument("--dst", type=Path, default=Path("Dataset/FlyingChairs_100_MSR/data"))
    parser.add_argument("--scales", type=int, nargs="+", default=[15, 80, 250])
    args = parser.parse_args()

    img1_files = sorted(args.src.glob("*_img1.ppm"))
    print(f"Found {len(img1_files)} image pairs in {args.src}")

    for img1_path in img1_files:
        base = img1_path.stem.replace("_img1", "")
        process_pair(base, args.src, args.dst, args.scales)

    print(f"Done. Output written to {args.dst}")


if __name__ == "__main__":
    main()
