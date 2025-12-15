import argparse
import itertools
import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


def msr_two_scale(img: np.ndarray, sigmas: tuple[float, float]) -> np.ndarray:
    """Two-scale Retinex on BGR image."""
    img_f = img.astype(np.float32) + 1.0
    retinex = np.zeros_like(img_f)
    for sigma in sigmas:
        ksize = int(((sigma - 0.8) / 0.15) + 2.0)
        if ksize % 2 == 0:
            ksize += 1
        blur = cv2.GaussianBlur(img_f, (ksize, ksize), sigma)
        retinex += np.log(img_f) - np.log(blur + 1e-6)
    retinex /= len(sigmas)

    out = np.zeros_like(img_f)
    for c in range(3):
        channel = retinex[..., c]
        channel = (channel - channel.min()) / (channel.max() - channel.min() + 1e-6)
        out[..., c] = channel * 255.0
    return np.clip(out, 0, 255).astype(np.uint8)


def process_dataset(src: Path, dst: Path, sigmas: tuple[float, float]):
    dst.mkdir(parents=True, exist_ok=True)
    for img1_path in sorted(src.glob("*_img1.ppm")):
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
        out1 = msr_two_scale(img1, sigmas)
        out2 = msr_two_scale(img2, sigmas)
        cv2.imwrite(str(dst / img1_path.name), out1)
        cv2.imwrite(str(dst / img2_path.name), out2)
        shutil.copy2(flow_path, dst / flow_path.name)


def run_letnet_chairs(demo_bin: Path, model_param: Path, model_bin: Path, data_dir: Path, workdir: Path) -> pd.DataFrame:
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
    parser = argparse.ArgumentParser(description="Sweep two-scale MSR and plot CTR.")
    parser.add_argument("--src", type=Path, default=Path("Dataset/FlyingChairs_100/data"),
                        help="Source dataset directory (triplets).")
    parser.add_argument("--sigma1-range", type=float, nargs=3, metavar=("START", "END", "STEP"),
                        default=[15, 40, 10],
                        help="Range for first sigma (start end step).")
    parser.add_argument("--sigma2-range", type=float, nargs=3, metavar=("START", "END", "STEP"),
                        default=[80, 200, 40],
                        help="Range for second sigma (start end step).")
    parser.add_argument("--out-dir", type=Path, default=Path("msr_two_scale_sweep"),
                        help="Directory to store per-run metrics and plot.")
    args = parser.parse_args()

    src_dir = args.src.resolve()
    demo_bin = Path("LET-NET/build/demo").resolve()
    model_param = Path("LET-NET/model/model.param").resolve()
    model_bin = Path("LET-NET/model/model.bin").resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    def frange(start: float, end: float, step: float):
        vals = []
        v = start
        while v <= end + 1e-6:
            vals.append(v)
            v += step
        return vals

    sigma1_vals_grid = frange(*args.sigma1_range)
    sigma2_vals_grid = frange(*args.sigma2_range)
    scale_pairs = list(itertools.product(sigma1_vals_grid, sigma2_vals_grid))

    grid_ctr = np.zeros((len(sigma2_vals_grid), len(sigma1_vals_grid)), dtype=float)

    # Baseline without MSR.
    print("Running baseline (no MSR) on source dataset...")
    with tempfile.TemporaryDirectory() as base_tmp_str:
        base_tmp = Path(base_tmp_str)
        df_base = run_letnet_chairs(
            demo_bin=demo_bin,
            model_param=model_param,
            model_bin=model_bin,
            data_dir=src_dir,
            workdir=base_tmp,
        )
        baseline_ctr = df_base["correct_tracking_ratio"].mean()
        base_metrics_out = args.out_dir / "metrics_baseline_no_msr.csv"
        df_base.to_csv(base_metrics_out, index=False)
        print(f"  baseline mean CTR={baseline_ctr:.4f}, metrics saved to {base_metrics_out}")

    for i, sig1 in enumerate(sigma1_vals_grid):
        for j, sig2 in enumerate(sigma2_vals_grid):
            sigmas = (sig1, sig2)
            print(f"Processing sigmas={sigmas}...")
            with tempfile.TemporaryDirectory() as tmpdir_str:
                tmpdir = Path(tmpdir_str)
                tmp_data = tmpdir / "data"
                process_dataset(src_dir, tmp_data, sigmas)
                df = run_letnet_chairs(
                    demo_bin=demo_bin,
                    model_param=model_param,
                    model_bin=model_bin,
                    data_dir=tmp_data,
                    workdir=tmpdir,
                )
                mean_ctr = df["correct_tracking_ratio"].mean()
                grid_ctr[j, i] = mean_ctr
                metrics_out = args.out_dir / f"metrics_msr_sigma{sigmas[0]}_{sigmas[1]}.csv"
                df.to_csv(metrics_out, index=False)
                print(f"  mean CTR={mean_ctr:.4f}, metrics saved to {metrics_out}")

    Sig1, Sig2 = np.meshgrid(sigma1_vals_grid, sigma2_vals_grid)
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(Sig1, Sig2, grid_ctr, cmap="viridis", edgecolor="none", alpha=0.9)
    baseline_plane = np.ones_like(grid_ctr) * baseline_ctr
    ax.plot_surface(Sig1, Sig2, baseline_plane, color="red", alpha=0.2, edgecolor="none")
    ax.set_xlabel("Sigma 1")
    ax.set_ylabel("Sigma 2")
    ax.set_zlabel("Mean CTR")
    ax.set_title("Two-scale MSR sweep (LET-NET CTR)")
    fig.colorbar(surf, shrink=0.6, aspect=10, label="Mean CTR")
    # Legend proxies
    legend_patches = [
        mpatches.Patch(color="red", alpha=0.2, label="Baseline (no MSR)"),
        mpatches.Patch(color=plt.cm.viridis(0.6), label="MSR surface"),
    ]
    ax.legend(handles=legend_patches, loc="upper right")
    plt.tight_layout()
    plot_path = args.out_dir / "msr_two_scale_ctr.png"
    plt.savefig(plot_path, dpi=200)
    print(f"CTR surface plot saved to {plot_path}")


if __name__ == "__main__":
    main()
