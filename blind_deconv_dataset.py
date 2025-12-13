import argparse
from pathlib import Path
import shutil

import cv2
import numpy as np

from blind_deconvolution import alternating_blind_deconv


def deblur_image(src_path: Path, dst_path: Path, **kwargs) -> None:
    img = cv2.imread(str(src_path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Failed to read image: {src_path}")
    restored, _ = alternating_blind_deconv(img.astype("float32") / 255.0, **kwargs)
    # Scale back to 8-bit and clamp to keep LET-NET demo happy.
    restored = (np.clip(restored, 0.0, 1.0) * 255.0).astype("uint8")
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dst_path), restored)


def process_dataset(src: Path, dst: Path, max_count: int | None = None, **kwargs) -> None:
    files = sorted(src.glob("*_img[12].ppm"))
    if max_count:
        files = files[:max_count * 2]  # two images per pair
    total = len(files)
    copied_flows = 0
    for idx, img_path in enumerate(files, 1):
        rel = img_path.relative_to(src)
        out_img = dst / rel
        deblur_image(img_path, out_img, **kwargs)

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
    parser.add_argument("--kernel-size", type=int, default=15, help="Kernel size for RL deconvolution.")
    parser.add_argument("--outer-iters", type=int, default=6, help="Alternating iterations.")
    parser.add_argument("--image-iters", type=int, default=12, help="Image RL iterations.")
    parser.add_argument("--kernel-iters", type=int, default=8, help="Kernel RL iterations.")
    parser.add_argument("--max-count", type=int, default=None,
                        help="Optional number of pairs to process (for quick sweeps).")
    parser.add_argument("--no-luma-only", action="store_true", help="Disable luma-only deconvolution (default uses LAB L-channel).")
    parser.add_argument("--denoise-sigma", type=float, default=0.0, help="Optional pre-blur sigma to suppress ringing.")
    parser.add_argument("--pyramid-levels", type=int, default=4, help="Number of pyramid levels.")
    parser.add_argument("--scale-factor", type=float, default=0.5, help="Downsample factor per pyramid level.")
    parser.add_argument("--alpha", type=float, default=0.8, help="Hyper-Laplacian exponent (<1).")
    parser.add_argument("--lambda-img", type=float, default=0.003, help="Image prior weight.")
    parser.add_argument("--kernel-lambda", type=float, default=0.001, help="Kernel L2 prior weight.")
    parser.add_argument("--image-iters-coarse", type=int, default=20, help="Image iters at coarse levels.")
    parser.add_argument("--kernel-iters-coarse", type=int, default=12, help="Kernel iters at coarse levels.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    process_dataset(
        args.src,
        args.dst,
        max_count=args.max_count,
        kernel_size=args.kernel_size,
        outer_iters=args.outer_iters,
        image_iters=args.image_iters,
        kernel_iters=args.kernel_iters,
        luma_only=not args.no_luma_only,
        denoise_sigma=args.denoise_sigma,
        pyramid_levels=args.pyramid_levels,
        scale_factor=args.scale_factor,
        alpha=args.alpha,
        lam_img=args.lambda_img,
        ker_lam=args.kernel_lambda,
        image_iters_coarse=args.image_iters_coarse,
        kernel_iters_coarse=args.kernel_iters_coarse,
    )


if __name__ == "__main__":
    main()
