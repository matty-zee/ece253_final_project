import argparse
import math
from pathlib import Path
from typing import Tuple, List

import cv2
import numpy as np
from scipy.signal import fftconvolve

EPS = 1e-8


def _grad(img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    gx = np.zeros_like(img)
    gy = np.zeros_like(img)
    gx[:, :-1] = img[:, 1:] - img[:, :-1]
    gy[:-1, :] = img[1:, :] - img[:-1, :]
    return gx, gy


def _div(gx: np.ndarray, gy: np.ndarray) -> np.ndarray:
    div = np.zeros_like(gx)
    div[:, :-1] += gx[:, :-1]
    div[:, 1:] -= gx[:, :-1]
    div[:-1, :] += gy[:-1, :]
    div[1:, :] -= gy[:-1, :]
    return div


def prior_shrinkage(img: np.ndarray, alpha: float, lam: float, step: float = 0.2) -> np.ndarray:
    """Simple proximal-like shrinkage for hyper-Laplacian gradient prior."""
    gx, gy = _grad(img)
    mag = np.sqrt(gx * gx + gy * gy + 1e-12)
    weight = np.power(mag, alpha - 2.0)  # alpha<1 => negative exponent
    gx_shrunk = gx * (1.0 - step * lam * weight)
    gy_shrunk = gy * (1.0 - step * lam * weight)
    # Prevent negative scaling from over-shrinkage
    gx_shrunk = np.where(gx_shrunk * gx < 0, 0, gx_shrunk)
    gy_shrunk = np.where(gy_shrunk * gy < 0, 0, gy_shrunk)
    return img + _div(gx_shrunk - gx, gy_shrunk - gy)


def richardson_lucy(channel: np.ndarray, psf: np.ndarray, iterations: int, alpha: float, lam: float) -> np.ndarray:
    """Non-blind Richardson-Lucy deconvolution on a single channel with hyper-Laplacian prior."""
    estimate = np.clip(channel.astype(np.float32), 0.0, 1.0)
    psf_flip = psf[::-1, ::-1]
    for _ in range(iterations):
        conv = fftconvolve(estimate, psf, mode="same") + EPS
        relative_blur = channel / conv
        estimate *= fftconvolve(relative_blur, psf_flip, mode="same")
        estimate = prior_shrinkage(np.clip(estimate, 0.0, 1.0), alpha=alpha, lam=lam)
        estimate = np.clip(estimate, 0.0, 1.0)
    return estimate


def update_kernel(observed: np.ndarray, latent: np.ndarray, kernel: np.ndarray, iterations: int, ker_lam: float) -> np.ndarray:
    """RL-style kernel refinement using the current latent estimate with L2 prior."""
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
        if ker_lam > 0:
            k *= np.exp(-ker_lam * k * k)
        k = np.clip(k, 0.0, None)
        s = k.sum()
        if s > 0:
            k /= s
    return k


def build_pyramid(img: np.ndarray, levels: int, scale: float) -> List[np.ndarray]:
    pyr = [img]
    for _ in range(1, levels):
        down = cv2.resize(pyr[-1], (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        pyr.append(down)
    return pyr[::-1]  # coarse to fine


def alternating_blind_deconv(
    image: np.ndarray,
    kernel_size: int = 15,
    outer_iters: int = 6,
    image_iters: int = 12,
    kernel_iters: int = 8,
    luma_only: bool = True,
    denoise_sigma: float = 0.0,
    pyramid_levels: int = 4,
    scale_factor: float = 0.5,
    alpha: float = 0.8,
    lam_img: float = 0.003,
    ker_lam: float = 0.001,
    image_iters_coarse: int = 20,
    kernel_iters_coarse: int = 12,
) -> Tuple[np.ndarray, np.ndarray]:
    """Multiscale alternating blind deconvolution with hyper-Laplacian prior."""
    if denoise_sigma > 0:
        image = cv2.GaussianBlur(image, (0, 0), denoise_sigma)

    if luma_only and image.ndim == 3:
        lab = cv2.cvtColor((image * 255.0).astype(np.uint8), cv2.COLOR_BGR2LAB).astype(np.float32) / 255.0
        l, a, b = cv2.split(lab)
        work_img = l
        chroma = (a, b)
        is_color = True
    elif image.ndim == 2:
        work_img = image
        is_color = False
        chroma = None
    else:
        work_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        is_color = True
        chroma = None

    pyr = build_pyramid(work_img, pyramid_levels, scale_factor)
    k = np.ones((kernel_size, kernel_size), dtype=np.float32)
    k /= k.sum()

    for level_idx, img_lvl in enumerate(pyr):
        if level_idx > 0:
            # Upsample kernel to current level
            new_sz = (img_lvl.shape[1], img_lvl.shape[0])
            k = cv2.resize(k, (kernel_size, kernel_size), interpolation=cv2.INTER_CUBIC)
            k = np.clip(k, 0, None)
            s = k.sum()
            if s > 0:
                k /= s

        img_channels = [img_lvl]
        latent_channels = [c.copy() for c in img_channels]

        is_fine = (level_idx == pyramid_levels - 1)
        i_iters = image_iters if is_fine else image_iters_coarse
        k_iters = kernel_iters if is_fine else kernel_iters_coarse

        for _ in range(outer_iters):
            latent_channels = [
                richardson_lucy(c, k, iterations=i_iters, alpha=alpha, lam=lam_img) for c in latent_channels
            ]
            latent_gray = latent_channels[0]
            observed_gray = img_channels[0]
            k = update_kernel(observed_gray, latent_gray, k, k_iters, ker_lam)

    restored_l = np.clip(latent_channels[0], 0.0, 1.0)
    if luma_only and is_color and chroma is not None:
        merged = cv2.merge([restored_l, chroma[0], chroma[1]])
        bgr = cv2.cvtColor((merged * 255.0).astype(np.uint8), cv2.COLOR_LAB2BGR)
        latent = bgr
    else:
        latent = np.clip(restored_l * 255.0, 0, 255).astype(np.uint8)
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
    parser.add_argument("--no-luma-only", action="store_true", help="Disable luma-only deconvolution (default uses LAB L-channel only).")
    parser.add_argument("--denoise-sigma", type=float, default=0.0, help="Optional pre-blur sigma to suppress noise/ringing.")
    parser.add_argument("--pyramid-levels", type=int, default=4, help="Number of pyramid levels (coarse-to-fine).")
    parser.add_argument("--scale-factor", type=float, default=0.5, help="Downsample factor per pyramid level.")
    parser.add_argument("--alpha", type=float, default=0.8, help="Hyper-Laplacian exponent (<1 suppresses ringing).")
    parser.add_argument("--lambda-img", type=float, default=0.003, help="Image prior weight for hyper-Laplacian.")
    parser.add_argument("--kernel-lambda", type=float, default=0.001, help="Kernel L2 prior weight to prevent kernel bloating.")
    parser.add_argument("--image-iters-coarse", type=int, default=20, help="Image iterations at coarse levels.")
    parser.add_argument("--kernel-iters-coarse", type=int, default=12, help="Kernel iterations at coarse levels.")
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
    else:
        process_image(
            args.input,
            args.output,
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
