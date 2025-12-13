import argparse
import math
from pathlib import Path

import cv2
import numpy as np
from scipy.signal import fftconvolve

EPS = 1e-8


def _grad(img):
    gx = np.zeros_like(img)
    gy = np.zeros_like(img)
    gx[:, :-1] = img[:, 1:] - img[:, :-1]
    gy[:-1, :] = img[1:, :] - img[:-1, :]
    return gx, gy


def _div(gx, gy):
    div = np.zeros_like(gx)
    div[:, :-1] += gx[:, :-1]
    div[:, 1:] -= gx[:, :-1]
    div[:-1, :] += gy[:-1, :]
    div[1:, :] -= gy[:-1, :]
    return div


def prior_shrinkage(img, alpha, lam, step=0.2):

    gx, gy = _grad(img)
    mag = np.sqrt(gx * gx + gy * gy + 1e-12)
    weight = np.power(mag, alpha - 2.0)  # alpha<1 => negative exponent
    gx_shrunk = gx * (1.0 - step * lam * weight)
    gy_shrunk = gy * (1.0 - step * lam * weight)
    # Prevent negative scaling from over-shrinkage
    gx_shrunk = np.where(gx_shrunk * gx < 0, 0, gx_shrunk)
    gy_shrunk = np.where(gy_shrunk * gy < 0, 0, gy_shrunk)
    return img + _div(gx_shrunk - gx, gy_shrunk - gy)


def richardson_lucy(channel, psf, iterations, alpha, lam):

    estimate = np.clip(channel.astype(np.float32), 0.0, 1.0)
    psf_flip = psf[::-1, ::-1]
    for _ in range(iterations):
        conv = fftconvolve(estimate, psf, mode="same") + EPS
        relative_blur = channel / conv
        estimate *= fftconvolve(relative_blur, psf_flip, mode="same")
        estimate = prior_shrinkage(np.clip(estimate, 0.0, 1.0), alpha=alpha, lam=lam)
        estimate = np.clip(estimate, 0.0, 1.0)
    return estimate


def update_kernel(observed, latent, kernel, iterations, ker_lam):

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


def build_pyramid(img, levels, scale):
    pyr = [img]
    for _ in range(1, levels):
        down = cv2.resize(pyr[-1], (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        pyr.append(down)
    return pyr[::-1]  # coarse to fine


def alternating_blind_deconv(
    image,
):

    # Default hyperparameters chosen for general-purpose blind deconvolution
    kernel_size = 15
    outer_iters = 6
    image_iters = 12
    kernel_iters = 8
    luma_only = True
    denoise_sigma = 0.0
    pyramid_levels = 4
    scale_factor = 0.5
    alpha = 0.8
    lam_img = 0.003
    ker_lam = 0.001
    image_iters_coarse = 20
    kernel_iters_coarse = 12

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


def process_image(path, output):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Could not read image: {path}")
    img_f = img.astype(np.float32) / 255.0
    restored, kernel = alternating_blind_deconv(img_f)
    cv2.imwrite(str(output), restored)
    print(f"Wrote deblurred image to {output} with kernel sum {kernel.sum():.6f}")


def process_video(path, output):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
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
        restored, _ = alternating_blind_deconv(frame.astype(np.float32) / 255.0)
        writer.write(restored)
        frames += 1

    cap.release()
    writer.release()
    print(f"Processed {frames} frames to {output} at {fps:.2f} FPS.")


def parse_args():
    parser = argparse.ArgumentParser(description="Blind deconvolution (alternating RL).")
    parser.add_argument("--input", required=True, type=Path, help="Input image or video path.")
    parser.add_argument("--output", required=True, type=Path, help="Output path.")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.input.suffix.lower() in {".mp4", ".mov", ".avi", ".mkv"}:
        process_video(args.input, args.output)
    else:
        process_image(args.input, args.output)


if __name__ == "__main__":
    main()
