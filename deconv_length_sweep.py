"""
Sweep blind-deconvolution kernel lengths on the blurred Library_Out dataset,
evaluate LET-NET tracking against ground-truth .flo, and plot mean CTR vs kernel length.
"""

import csv
import os
import subprocess
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt

from blind_deconv_dataset import process_dataset


@dataclass
class SweepConfig:
    name: str
    kernel_size: int


HEADLESS_ENV = {"HEADLESS": "1"}


def run_letnet(data_dir: Path, letnet_dir: Path) -> float:
    env = os.environ.copy()
    env.update(HEADLESS_ENV)
    cmd = [
        "./build/demo",
        "./model/model.param",
        "./model/model.bin",
        str(data_dir.resolve()),
        "--chairs",
    ]
    res = subprocess.run(cmd, cwd=str(letnet_dir), capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"LET-NET failed ({res.returncode}):\n{res.stdout}\n{res.stderr}")
    metrics_path = letnet_dir / "letnet_chairs_metrics.csv"
    if not metrics_path.exists():
        raise FileNotFoundError("letnet_chairs_metrics.csv not produced")
    metrics_copy = data_dir / "letnet_chairs_metrics.csv"
    metrics_copy.write_bytes(metrics_path.read_bytes())

    ctrs: List[float] = []
    with metrics_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            ctrs.append(float(row["ctr"]))
    return sum(ctrs) / len(ctrs) if ctrs else 0.0


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Sweep blind deconv kernel sizes and evaluate LET-NET CTR.")
    parser.add_argument("--src", type=Path, required=True, help="FlyingChairs-style dataset (blurred).")
    parser.add_argument("--out-root", type=Path, required=True, help="Output directory for deconv splits + summary.")
    parser.add_argument("--letnet-dir", type=Path, default=Path("LET-NET"), help="LET-NET repo dir containing build/demo.")
    parser.add_argument("--max-pairs", type=int, default=8, help="Pairs per config to deconv/eval.")
    parser.add_argument("--kernels", type=int, nargs="+", default=[9, 15, 25], help="Kernel sizes to sweep.")
    args = parser.parse_args()

    configs = [SweepConfig(f"k{ks:02d}", ks) for ks in args.kernels]
    max_pairs = args.max_pairs
    OUT_ROOT = args.out_root
    SRC = args.src
    LETNET_DIR = args.letnet_dir

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    summary = []

    for cfg in configs:
        dst = OUT_ROOT / cfg.name
        if (dst / "letnet_chairs_metrics.csv").exists():
            print(f"Skipping {cfg.name}, metrics already present.")
            mean_ctr = run_letnet(dst, LETNET_DIR)
        else:
            print(f"Processing {cfg.name} (kernel_size={cfg.kernel_size})...")
            process_dataset(
                src=SRC,
                dst=dst,
                max_count=max_pairs,
                kernel_size=cfg.kernel_size,
                outer_iters=2,
                image_iters=6,
                kernel_iters=4,
                luma_only=True,
                denoise_sigma=0.0,
                pyramid_levels=2,
                scale_factor=0.5,
                alpha=0.8,
                lam_img=0.003,
                ker_lam=0.001,
                image_iters_coarse=10,
                kernel_iters_coarse=6,
            )
            mean_ctr = run_letnet(dst, LETNET_DIR)
        print(f"  mean CTR: {mean_ctr:.4f}")
        row = asdict(cfg)
        row["mean_ctr"] = mean_ctr
        summary.append(row)

    # Save summary CSV
    summary_path = OUT_ROOT / "summary.csv"
    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary[0].keys())
        writer.writeheader()
        writer.writerows(summary)

    # Plot
    plt.figure(figsize=(6, 4))
    plt.plot([s["kernel_size"] for s in summary], [s["mean_ctr"] for s in summary], marker="o")
    plt.xlabel("Blind deconvolution kernel size")
    plt.ylabel("Mean CTR (LET-NET, threshold 1.5px, scaled flow)")
    plt.ylim(0, 1.05)
    plt.grid(True)
    plt.title("Effect of deconvolution kernel length on LET-NET accuracy")
    plt.tight_layout()
    plot_path = OUT_ROOT / "summary.png"
    plt.savefig(plot_path, dpi=180)
    print(f"Summary CSV: {summary_path}")
    print(f"Plot: {plot_path}")


if __name__ == "__main__":
    main()
