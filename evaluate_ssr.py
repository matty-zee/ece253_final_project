"""
Apply Single-Scale Retinex (SSR) to a FlyingChairs-style dataset for multiple
sigma values, run LET-NET `--chairs` evaluation for each, and plot mean correct
tracking ratio (CTR) versus sigma.

Usage example:
python plot_ssr_ctr_vs_sigma.py \
  --src Flyingchairs_100_dark/data \
  --demo-bin LET-NET/build/demo \
  --model-param LET-NET/model/model.param \
  --model-bin LET-NET/model/model.bin \
  --sigmas 5 10 15 20 \
  --out-dir ssr_sweep_results
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


def get_ksize(sigma: float) -> int:
    return int(((sigma - 0.8) / 0.15) + 2.0)


def get_gaussian_blur(img: np.ndarray, ksize: int, sigma: float) -> np.ndarray:
    sep_k = cv2.getGaussianKernel(ksize, sigma)
    kernel = np.outer(sep_k, sep_k)
    return cv2.filter2D(img, -1, kernel)


def ssr(img: np.ndarray, sigma: float) -> np.ndarray:
    eps = 1.0
    ksize = get_ksize(sigma) if sigma > 0 else 3
    blurred = get_gaussian_blur(img, ksize=ksize, sigma=sigma)
    res = np.log10(img.astype(np.float32) + eps) - np.log10(blurred + eps)
    res = res - res.min()
    res = res / (res.max() + 1e-6)
    return np.clip(res * 255.0, 0, 255).astype(np.uint8)


def process_dataset_with_ssr(src: Path, dst: Path, sigma: float):
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
        out1 = ssr(img1, sigma)
        out2 = ssr(img2, sigma)
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
    parser = argparse.ArgumentParser(description="Plot CTR vs sigma for SSR-processed datasets.")
    parser.add_argument("--src", type=Path, default=Path("Flyingchairs_100_dark/data"),
                        help="Source dataset directory (triplets).")
    parser.add_argument("--sigmas", type=float, nargs="+", default=[5, 10, 15, 20],
                        help="Sigma values to sweep for SSR.")
    parser.add_argument("--out-dir", type=Path, default=Path("ssr_sweep_results"),
                        help="Directory to store per-run metrics and the plot.")
    parser.add_argument("--export-sigma", type=float, default=None,
                        help="If set, persist the SSR-processed dataset for this sigma.")
    parser.add_argument("--export-dir", type=Path, default=Path("CLAHE_100_SSr/data"),
                        help="Where to write the exported SSR dataset (used with --export-sigma).")
    args = parser.parse_args()

    src_dir = args.src.resolve()
    demo_bin = Path("LET-NET/build/demo").resolve()
    model_param = Path("LET-NET/model/model.param").resolve()
    model_bin = Path("LET-NET/model/model.bin").resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    sigma_vals = []
    ctr_means = []

    # Baseline without SSR.
    print("Running baseline (no SSR) on source dataset...")
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
        base_metrics_out = args.out_dir / "metrics_baseline_no_ssr.csv"
        df_base.to_csv(base_metrics_out, index=False)
        print(f"  baseline mean CTR={baseline_ctr:.4f}, metrics saved to {base_metrics_out}")

    for sigma in args.sigmas:
        print(f"Processing sigma={sigma}...")
        with tempfile.TemporaryDirectory() as tmpdir_str:
            tmpdir = Path(tmpdir_str)
            tmp_data = tmpdir / "data"
            process_dataset_with_ssr(src_dir, tmp_data, sigma)
            df = run_letnet_chairs(
                demo_bin=demo_bin,
                model_param=model_param,
                model_bin=model_bin,
                data_dir=tmp_data,
                workdir=tmpdir,
            )
            mean_ctr = df["correct_tracking_ratio"].mean()
            sigma_vals.append(sigma)
            ctr_means.append(mean_ctr)
            metrics_out = args.out_dir / f"metrics_ssr_sigma{sigma}.csv"
            df.to_csv(metrics_out, index=False)
            print(f"  mean CTR={mean_ctr:.4f}, metrics saved to {metrics_out}")
            # Optional export of processed dataset for a specific sigma.
            if args.export_sigma is not None and abs(sigma - args.export_sigma) < 1e-6:
                export_dir = args.export_dir
                export_dir.mkdir(parents=True, exist_ok=True)
                for f in tmp_data.iterdir():
                    shutil.copy2(f, export_dir / f.name)
                print(f"  exported SSR dataset for sigma={sigma} to {export_dir}")

    plt.figure(figsize=(8, 5))
    plt.plot(sigma_vals, ctr_means, marker="o")
    plt.axhline(baseline_ctr, color="red", linestyle="--", label="Baseline (no SSR)")
    plt.xlabel("Sigma")
    plt.ylabel("Mean correct tracking ratio")
    plt.title("SSR sigma sweep (LET-NET CTR)")
    plt.grid(True)
    plt.legend()
    plot_path = args.out_dir / "ssr_ctr_vs_sigma.png"
    plt.savefig(plot_path, dpi=200)
    print(f"CTR plot saved to {plot_path}")


if __name__ == "__main__":
    main()
