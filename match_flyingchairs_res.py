import argparse
from pathlib import Path

import cv2
import numpy as np


def letterbox(frame, target_w=512, target_h=384):
    h, w = frame.shape[:2]
    scale = min(target_w / w, target_h / h, 1.0)  # never upscale
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

    top = (target_h - new_h) // 2
    bottom = target_h - new_h - top
    left = (target_w - new_w) // 2
    right = target_w - new_w - left

    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    return padded


def main():
    parser = argparse.ArgumentParser(description="Scale video to FlyingChairs resolution (512x384) with aspect ratio preserved.")
    parser.add_argument("--input", type=Path, default=Path("library_outside.mp4"), help="Input video path.")
    parser.add_argument("--output", type=Path, default=Path("library_outside_resized.mp4"), help="Output video path.")
    args = parser.parse_args()

    cap = cv2.VideoCapture(str(args.input))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open input video: {args.input}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 1 or np.isnan(fps):
        fps = 24.0

    writer = cv2.VideoWriter(
        str(args.output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (512, 384),
    )

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        out = letterbox(frame, 512, 384)
        writer.write(out)
        frame_count += 1

    cap.release()
    writer.release()
    print(f"Wrote {frame_count} frames to {args.output}")


if __name__ == "__main__":
    main()
