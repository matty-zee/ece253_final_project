"""
Sweep CLAHE parameters on a FlyingChairs-style dataset and plot mean correct
tracking ratio (CTR) vs. clipLimit and tileGridSize using the LET-NET demo.

For each (clipLimit, tileGridSize) pair:
  1) Apply CLAHE to all *_img1.ppm/_img2.ppm in the source dataset (flows copied).
  2) Run LET-NET demo with --chairs on the CLAHE-processed set.
  3) Collect mean CTR from the generated letnet_chairs_metrics.csv.
Finally, render a 3D surface plot of CTR over the parameter grid and save per-run
metrics into the output directory.

Example:
python plot_clahe_ctr_surface.py \\
  --src FlyingChairs_100/data \\
  --demo-bin LET-NET/build/demo \\
  --model-param LET-NET/model/model.param \\
  --model-bin LET-NET/model/model.bin \\
  --clip-limits 2.0 3.0 4.0 \\
  --tile-sizes 4 6 8 \\
  --out-dir clahe_sweep_results
"""

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


def apply_clahe_color(img: np.ndarray, clip_limit: float, tile_grid: int) -> np.ndarray:
    """Apply CLAHE on the L channel in LAB space."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid, tile_grid))
    l2 = clahe.apply(l)
    lab2 = cv2.merge([l2, a, b])
    return cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)


def make_clahe_dataset(src: Path, dst: Path, clip_limit: float, tile_grid: int):
    """Apply CLAHE to all triplets in src and write to dst."""
    dst.mkdir(parents=True, exist_ok=True)
    img1_files = sorted(src.glob("*_img1.ppm"))
    for img1_path in img1_files:
        base = img1_path.stem.replace("_img1", "")
        img2_path = src / f"{base}_img2.ppm"
        flow_path = src / f"{base}_flow.flo"
        if not (img2_path.exists() and flow_path.exists()):
            print(f"Skipping {base}: missing pair/flow.")
            continue
        img1 = cv2.imread(str(img1_path), cv2.IMREAD_COLOR)
        img2 = cv2.imread(str(img2_path), cv2.IMREAD_COLOR)
        if img1 is None or img2 is None:
            print(f"Skipping {base}: failed to read images.")
            continue
        out1 = apply_clahe_color(img1, clip_limit, tile_grid)
        out2 = apply_clahe_color(img2, clip_limit, tile_grid)
        cv2.imwrite(str(dst / img1_path.name), out1)
        cv2.imwrite(str(dst / img2_path.name), out2)
        shutil.copy2(flow_path, dst / flow_path.name)


def run_letnet_chairs(demo_bin: Path, model_param: Path, model_bin: Path, data_dir: Path, workdir: Path) -> pd.DataFrame:
    """Run LET-NET demo in --chairs mode and return the metrics CSV as DataFrame."""
    cmd = [
        str(demo_bin),
        str(model_param),
        str(model_bin),
        str(data_dir),
        "--chairs",
    ]
    subprocess.run(cmd, check=True, cwd=workdir)
    metrics_path = workdir / "letnet_chairs_metrics.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Metrics file not found at {metrics_path}")
    return pd.read_csv(metrics_path)


def main():
    parser = argparse.ArgumentParser(description="Sweep CLAHE params and plot CTR surface.")
    parser.add_argument("--src", type=Path, default=Path("FlyingChairs_100/data"),
                        help="Source dataset directory (triplets).")
    parser.add_argument("--demo-bin", type=Path, default=Path("LET-NET/build/demo"),
                        help="Path to built LET-NET demo binary.")
    parser.add_argument("--model-param", type=Path, default=Path("LET-NET/model/model.param"),
                        help="Path to LET-NET model param file.")
    parser.add_argument("--model-bin", type=Path, default=Path("LET-NET/model/model.bin"),
                        help="Path to LET-NET model bin file.")
    parser.add_argument("--clip-limits", type=float, nargs="+", default=[2.0, 3.0, 4.0],
                        help="List of CLAHE clipLimit values to sweep.")
    parser.add_argument("--tile-sizes", type=int, nargs="+", default=[4, 6, 8],
                        help="List of CLAHE tileGridSize values (square) to sweep.")
    parser.add_argument("--out-dir", type=Path, default=Path("clahe_sweep_results"),
                        help="Directory to store per-run metrics and the plot.")
    args = parser.parse_args()

    # Resolve binaries and models to absolute paths so they work from temp dirs.
    src_dir = args.src.resolve()
    demo_bin = args.demo_bin.resolve()
    model_param = args.model_param.resolve()
    model_bin = args.model_bin.resolve()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    grid_ctr = np.zeros((len(args.tile_sizes), len(args.clip_limits)), dtype=float)

    for i, tile in enumerate(args.tile_sizes):
        for j, clip in enumerate(args.clip_limits):
            print(f"Running clipLimit={clip}, tileGridSize={tile}...")
            with tempfile.TemporaryDirectory() as tmpdir_str:
                tmpdir = Path(tmpdir_str)
                tmp_data = tmpdir / "data"
                make_clahe_dataset(src_dir, tmp_data, clip_limit=clip, tile_grid=tile)

                df = run_letnet_chairs(
                    demo_bin=demo_bin,
                    model_param=model_param,
                    model_bin=model_bin,
                    data_dir=tmp_data,
                    workdir=tmpdir,
                )
                mean_ctr = df["correct_tracking_ratio"].mean()
                grid_ctr[i, j] = mean_ctr

                # Archive metrics
                metrics_out = args.out_dir / f"metrics_clip{clip}_tile{tile}.csv"
                df.to_csv(metrics_out, index=False)
                print(f"  mean CTR={mean_ctr:.4f}, metrics saved to {metrics_out}")

    # Plot surface
    Clip, Tile = np.meshgrid(args.clip_limits, args.tile_sizes)
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(Clip, Tile, grid_ctr, cmap="viridis", edgecolor="none", alpha=0.9)
    ax.set_xlabel("clipLimit")
    ax.set_ylabel("tileGridSize")
    ax.set_zlabel("Mean CTR")
    ax.set_title("Mean Correct Tracking Ratio vs. CLAHE parameters")
    plt.tight_layout()
    plot_path = args.out_dir / "ctr_surface.png"
    plt.savefig(plot_path, dpi=200)
    print(f"Surface plot saved to {plot_path}")


if __name__ == "__main__":
    main()
