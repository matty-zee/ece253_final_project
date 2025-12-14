"""
Extract consecutive frame pairs from a video, match SIFT keypoints, and write sparse
optical flow .flo files for each pair (max 300 keypoints per pair).

- Input video: library_outside.mp4 (default; override with --video)
- Output directory: library_outside_flow (default; override with --out)
- Frames are saved alongside flow as <index>_img1.ppm, <index>_img2.ppm, <index>_flow.flo

Flow encoding: only matched inlier keypoints are populated; other pixels are zero.
This is sparse flow stored in Middlebury .flo format for compatibility with existing
readers, but note that invalid regions remain zero.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np


def write_flo(path: Path, flow: np.ndarray):
    """Write a HxWx2 float32 flow array to .flo (Middlebury) format."""
    h, w, _ = flow.shape
    with path.open("wb") as f:
        f.write(b"PIEH")
        np.array([w], dtype=np.int32).tofile(f)
        np.array([h], dtype=np.int32).tofile(f)
        flow.astype(np.float32).tofile(f)


def sift_matches(frame1: np.ndarray, frame2: np.ndarray, max_kp: int = 300):
    """Detect/match SIFT keypoints and return filtered inlier correspondences."""
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()
    kpts1, desc1 = sift.detectAndCompute(gray1, None)
    kpts2, desc2 = sift.detectAndCompute(gray2, None)
    if desc1 is None or desc2 is None:
        return np.empty((0, 2), dtype=np.float32), np.empty((0, 2), dtype=np.float32)

    matcher = cv2.BFMatcher(cv2.NORM_L2)
    raw_matches = matcher.knnMatch(desc1, desc2, k=2)

    good = []
    for m, n in raw_matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)

    if not good:
        return np.empty((0, 2), dtype=np.float32), np.empty((0, 2), dtype=np.float32)

    # Sort by distance and keep top candidates before RANSAC.
    good = sorted(good, key=lambda m: m.distance)[: max_kp * 3]
    pts1 = np.float32([kpts1[m.queryIdx].pt for m in good])
    pts2 = np.float32([kpts2[m.trainIdx].pt for m in good])

    F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, 1.0, 0.99, 2000)
    if F is None or mask is None:
        return np.empty((0, 2), dtype=np.float32), np.empty((0, 2), dtype=np.float32)

    inliers = mask.ravel().astype(bool)
    pts1_in = pts1[inliers]
    pts2_in = pts2[inliers]

    # Limit to max_kp best (closest distance) inliers.
    dists = np.linalg.norm(pts1_in - pts2_in, axis=1)
    order = np.argsort(dists)
    keep = order[: max_kp]
    return pts1_in[keep], pts2_in[keep]


def process_video(video_path: Path, out_dir: Path, max_kp: int = 300):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    out_dir.mkdir(parents=True, exist_ok=True)

    idx = 1
    ret, prev = cap.read()
    if not ret:
        raise RuntimeError("Could not read first frame.")

    # Prepare visualization writer.
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 1 or np.isnan(fps):
        fps = 24.0
    h, w = prev.shape[:2]
    vis_path = out_dir.parent / "library_outside_keypoints.mp4"
    writer = cv2.VideoWriter(
        str(vis_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (w, h),  # single-frame overlay
    )

    while True:
        ret, curr = cap.read()
        if not ret:
            break

        pts1, pts2 = sift_matches(prev, curr, max_kp=max_kp)
        h, w = prev.shape[:2]
        flow = np.zeros((h, w, 2), dtype=np.float32)
        for p1, p2 in zip(pts1, pts2):
            x, y = int(round(p1[0])), int(round(p1[1]))
            if 0 <= x < w and 0 <= y < h:
                flow[y, x, 0] = p2[0] - p1[0]
                flow[y, x, 1] = p2[1] - p1[1]

        base = f"{idx:05d}"
        cv2.imwrite(str(out_dir / f"{base}_img1.ppm"), prev)
        cv2.imwrite(str(out_dir / f"{base}_img2.ppm"), curr)
        write_flo(out_dir / f"{base}_flow.flo", flow)

        # Visualization: single-frame overlay with motion arrows (prev frame, arrows to next).
        vis = prev.copy()
        for p1, p2 in zip(pts1, pts2):
            p1_int = tuple(int(round(v)) for v in p1)
            p2_int = tuple(int(round(v)) for v in p2)
            cv2.arrowedLine(vis, p1_int, p2_int, (255, 0, 0), 1, tipLength=0.2)
            cv2.circle(vis, p1_int, 3, (0, 255, 0), -1)
        cv2.putText(vis, f"Pair {idx:05d} | matches: {len(pts1)}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        writer.write(vis)

        idx += 1
        prev = curr

    cap.release()
    writer.release()
    print(f"Saved {idx - 1} flow files to {out_dir}")
    print(f"Visualization video written to {vis_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate sparse .flo files from video using SIFT + RANSAC.")
    parser.add_argument("--video", type=Path, default=Path("library_outside.mp4"), help="Input video path.")
    parser.add_argument("--out", type=Path, default=Path("library_outside_flow/data"), help="Output directory.")
    parser.add_argument("--max-kp", type=int, default=300, help="Maximum keypoints per frame pair.")
    args = parser.parse_args()
    process_video(args.video, args.out, max_kp=args.max_kp)


if __name__ == "__main__":
    main()
