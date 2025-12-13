import argparse
from pathlib import Path
import shutil

import cv2
import numpy as np

from blind_deconvolution import alternating_blind_deconv


def deblur_image(src_path: Path, dst_path: Path) -> None:
    img = cv2.imread(str(src_path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Failed to read image: {src_path}")
    restored, _ = alternating_blind_deconv(img.astype("float32") / 255.0)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dst_path), restored)


def process_dataset(src: Path, dst: Path, max_count: int | None = None) -> None:
    files = sorted(src.glob("*_img[12].ppm"))
    if max_count:
        files = files[:max_count * 2]  # two images per pair
    total = len(files)
    copied_flows = 0
    for idx, img_path in enumerate(files, 1):
        rel = img_path.relative_to(src)
        out_img = dst / rel
        deblur_image(img_path, out_img)

        base = img_path.name.split("_img")[0]
        flow_src = src / f"{base}_flow.flo"
        flow_dst = dst / f"{base}_flow.flo"
        if flow_src.exists() and not flow_dst.exists():
            flow_dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(flow_src, flow_dst)
            copied_flows += 1

        if total > 0 and idx % max(1, total // 10) == 0:
            print(f"Deblurred {idx}/{total} images ({idx / total:.0%})")
    print(f"Done. Deblurred {total} images, copied {copied_flows} flow files.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Blind deconvolution over a FlyingChairs-style dataset.")
    parser.add_argument("--src", type=Path, required=True, help="Blurred dataset directory.")
    parser.add_argument("--dst", type=Path, required=True, help="Destination deblurred dataset directory.")
    parser.add_argument("--max-count", type=int, default=None,
                        help="Optional number of pairs to process (for quick sweeps).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    process_dataset(
        args.src,
        args.dst,
        max_count=args.max_count,
    )


if __name__ == "__main__":
    main()
