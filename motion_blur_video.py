import argparse
import math
from pathlib import Path

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


def apply_motion_blur_frame(frame: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Apply motion blur PSF to a single frame."""
    return cv2.filter2D(frame, ddepth=-1, kernel=kernel, borderType=cv2.BORDER_REPLICATE)


def process_video(input_path: Path, output_path: Path, kernel: np.ndarray, fps_override: float | None):
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {input_path}")

    fps = fps_override or cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 1 or math.isnan(fps):
        fps = 24.0

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_count = 0
    last_reported = -1
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        blurred = apply_motion_blur_frame(frame, kernel)
        writer.write(blurred)
        frame_count += 1
        # Lightweight progress feedback every 5%
        if total_frames > 0:
            pct = int((frame_count / total_frames) * 20)  # 20 buckets = 5% steps
            if pct != last_reported:
                print(f"Processed {frame_count}/{total_frames} frames ({pct * 5}%)")
                last_reported = pct

    cap.release()
    writer.release()
    return frame_count, fps, (w, h)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Add synthetic motion blur to a video.")
    parser.add_argument("--input", required=True, type=Path, help="Source video (.mp4)")
    parser.add_argument("--output", required=True, type=Path, help="Destination blurred video (.mp4)")
    parser.add_argument("--length", type=int, default=15, help="Kernel length in pixels")
    parser.add_argument("--angle", type=float, default=15.0, help="Kernel angle in degrees")
    parser.add_argument("--width", type=int, default=1, help="Kernel thickness for the line PSF")
    parser.add_argument("--fps", type=float, default=None, help="Optional FPS override for the output video")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    kernel = build_motion_kernel(args.length, args.angle, args.width)
    frames, fps, shape = process_video(args.input, args.output, kernel, args.fps)
    print(f"Blurred {frames} frames at {fps:.2f} FPS, resolution {shape[0]}x{shape[1]}")


if __name__ == "__main__":
    main()