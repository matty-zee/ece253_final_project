import argparse
import math
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
from scipy.signal import fftconvolve

EPS = 1e-8


def richardson_lucy(channel: np.ndarray, psf: np.ndarray, iterations: int) -> np.ndarray:
    """Non-blind Richardson-Lucy deconvolution on a single channel."""
    estimate = np.clip(channel.astype(np.float32), 0.0, 1.0)
    psf_flip = psf[::-1, ::-1]
    for _ in range(iterations):
        conv = fftconvolve(estimate, psf, mode="same") + EPS
        relative_blur = channel / conv
        estimate *= fftconvolve(relative_blur, psf_flip, mode="same")
        estimate = np.clip(estimate, 0.0, 1.0)
    return estimate


def update_kernel(observed: np.ndarray, latent: np.ndarray, kernel: np.ndarray, iterations: int) -> np.ndarray:
    """RL-style kernel refinement using the current latent estimate."""
    k = kernel
    latent_flip = latent[::-1, ::-1]
    for _ in range(iterations):
        conv = fftconvolve(latent, k, mode="same") + EPS
        relative = observed / conv
        update_full = fftconvolve(relative, latent_flip, mode="same")

        # Extract centered patch matching kernel size
        start_r = (update_full.shape[0] - k.shape[0]) // 2
        start_c = (update_full.shape[1] - k.shape[1]) // 2
        update_patch = update_full[start_r : start_r + k.shape[0], start_c : start_c + k.shape[1]]

        k *= update_patch
        k = np.clip(k, 0.0, None)
        s = k.sum()
        if s > 0:
            k /= s
    return k


def alternating_blind_deconv(
    image: np.ndarray,
    kernel_size: int = 15,
    outer_iters: int = 6,
    image_iters: int = 12,
    kernel_iters: int = 8,
) -> Tuple[np.ndarray, np.ndarray]:
    """Simple alternating blind deconvolution using RL updates."""
    k = np.ones((kernel_size, kernel_size), dtype=np.float32)
    k /= k.sum()

    if image.ndim == 2:
        channels = [image]
        is_color = False
    else:
        channels = cv2.split(image)
        is_color = True

    latent_channels = [c.copy() for c in channels]

    for _ in range(outer_iters):
        # Update latent image channels
        latent_channels = [richardson_lucy(c, k, image_iters) for c in latent_channels]

        # Use luminance to stabilize kernel estimation
        if is_color:
            latent_gray = 0.2989 * latent_channels[0] + 0.5870 * latent_channels[1] + 0.1140 * latent_channels[2]
            observed_gray = 0.2989 * channels[0] + 0.5870 * channels[1] + 0.1140 * channels[2]
        else:
            latent_gray = latent_channels[0]
            observed_gray = channels[0]

        k = update_kernel(observed_gray, latent_gray, k, kernel_iters)

    latent = latent_channels[0] if not is_color else cv2.merge(latent_channels)
    latent = np.clip(latent * 255.0, 0, 255).astype(np.uint8)
    return latent, k


def process_image(path: Path, output: Path, **kwargs) -> None:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Could not read image: {path}")
    img_f = img.astype(np.float32) / 255.0
    restored, kernel = alternating_blind_deconv(img_f, **kwargs)
    cv2.imwrite(str(output), restored)
    print(f"Wrote deblurred image to {output} with kernel sum {kernel.sum():.6f}")


def process_video(path: Path, output: Path, fps_override: float | None, **kwargs) -> None:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")

    fps = fps_override or cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 1 or math.isnan(fps):
        fps = 24.0

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output), fourcc, fps, (w, h))

    frames = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        restored, _ = alternating_blind_deconv(frame.astype(np.float32) / 255.0, **kwargs)
        writer.write(restored)
        frames += 1

    cap.release()
    writer.release()
    print(f"Processed {frames} frames to {output} at {fps:.2f} FPS.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Blind deconvolution (alternating RL).")
    parser.add_argument("--input", required=True, type=Path, help="Input image or video path.")
    parser.add_argument("--output", required=True, type=Path, help="Output path.")
    parser.add_argument("--kernel-size", type=int, default=15, help="Square kernel size.")
    parser.add_argument("--outer-iters", type=int, default=6, help="Number of alternating iterations.")
    parser.add_argument("--image-iters", type=int, default=12, help="RL iterations for the image update.")
    parser.add_argument("--kernel-iters", type=int, default=8, help="RL iterations for the kernel update.")
    parser.add_argument("--fps", type=float, default=None, help="Optional FPS override for video output.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.input.suffix.lower() in {".mp4", ".mov", ".avi", ".mkv"}:
        process_video(
            args.input,
            args.output,
            fps_override=args.fps,
            kernel_size=args.kernel_size,
            outer_iters=args.outer_iters,
            image_iters=args.image_iters,
            kernel_iters=args.kernel_iters,
        )
    else:
        process_image(
            args.input,
            args.output,
            kernel_size=args.kernel_size,
            outer_iters=args.outer_iters,
            image_iters=args.image_iters,
            kernel_iters=args.kernel_iters,
        )


if __name__ == "__main__":
    main()
