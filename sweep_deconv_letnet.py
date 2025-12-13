import csv
import os
import shutil
import subprocess
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict

import matplotlib.pyplot as plt

from blind_deconv_dataset import process_dataset


@dataclass
class DeconvConfig:
    name: str
    kernel_size: int = 15
    denoise_sigma: float = 0.0
    alpha: float = 0.8
    outer_iters: int = 2
    image_iters: int = 6
    kernel_iters: int = 4
    pyramid_levels: int = 2
    max_pairs: int = 10  # limit workload for sweeps to keep runtime manageable

    def as_kwargs(self) -> Dict:
        return dict(
            kernel_size=self.kernel_size,
            outer_iters=self.outer_iters,
            image_iters=self.image_iters,
            kernel_iters=self.kernel_iters,
            luma_only=True,
            denoise_sigma=self.denoise_sigma,
            pyramid_levels=self.pyramid_levels,
            scale_factor=0.5,
            alpha=self.alpha,
            lam_img=0.003,
            ker_lam=0.001,
            image_iters_coarse=12,
            kernel_iters_coarse=8,
        )


SRC_DIR = Path("Library_Out/data_motion_blur")
DEST_ROOT = Path("Library_Out/deconv_sweeps")
LETNET_DIR = Path("LET-NET")


def run_letnet_chairs(data_dir: Path) -> float:
    """Run LET-NET demo in chairs mode and return mean CTR."""
    data_dir = data_dir.resolve()
    env = os.environ.copy()
    env.update({"HEADLESS": "1"})
    cmd = [
        "./build/demo",
        "./model/model.param",
        "./model/model.bin",
        str(data_dir),
        "--chairs",
    ]
    completed = subprocess.run(cmd, cwd=str(LETNET_DIR), env=env, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(f"LET-NET run failed (code {completed.returncode}):\n{completed.stdout}\n{completed.stderr}")
    metrics_path = LETNET_DIR / "letnet_chairs_metrics.csv"
    if not metrics_path.exists():
        raise RuntimeError("letnet_chairs_metrics.csv not found after LET-NET run")
    # Copy metrics to keep per-config logs.
    shutil.copy2(metrics_path, data_dir / "letnet_chairs_metrics.csv")
    # Compute mean CTR.
    ctrs = []
    with metrics_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            ctrs.append(float(row["ctr"]))
    if not ctrs:
        return 0.0
    return sum(ctrs) / len(ctrs)


def read_mean_ctr_from_file(path: Path) -> float:
    ctrs = []
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            ctrs.append(float(row["ctr"]))
    return sum(ctrs) / len(ctrs) if ctrs else 0.0


def main() -> None:
    configs: List[DeconvConfig] = [
        DeconvConfig(name="k11_fast", kernel_size=11, denoise_sigma=0.0, alpha=0.8),
        DeconvConfig(name="k15_dn05", kernel_size=15, denoise_sigma=0.5, alpha=0.8),
        DeconvConfig(name="k21_alpha07", kernel_size=21, denoise_sigma=0.0, alpha=0.7),
    ]

    results = []
    DEST_ROOT.mkdir(parents=True, exist_ok=True)

    for cfg in configs:
        dst = DEST_ROOT / cfg.name
        print(f"Processing config {cfg.name} -> {dst}")
        metrics_file = dst / "letnet_chairs_metrics.csv"
        if not metrics_file.exists():
            process_dataset(
                src=SRC_DIR,
                dst=dst,
                max_count=cfg.max_pairs,
                **cfg.as_kwargs(),
            )
            mean_ctr = run_letnet_chairs(dst)
        else:
            print("Metrics already present, skipping recompute.")
            mean_ctr = read_mean_ctr_from_file(metrics_file)
        print(f"{cfg.name}: mean CTR {mean_ctr:.4f}")
        row = asdict(cfg)
        row["mean_ctr"] = mean_ctr
        results.append(row)

    # Write summary CSV
    summary_path = DEST_ROOT / "deconv_letnet_summary.csv"
    fieldnames = list(results[0].keys())
    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    # Plot
    labels = [r["name"] for r in results]
    ctrs = [r["mean_ctr"] for r in results]
    plt.figure(figsize=(6, 4))
    plt.bar(labels, ctrs, color="#4C72B0")
    plt.ylim(0, 1.05)
    plt.ylabel("Mean CTR")
    plt.xlabel("Blind deconv config")
    plt.title("LET-NET accuracy after blind deconvolution sweep")
    plt.tight_layout()
    plot_path = DEST_ROOT / "deconv_letnet_summary.png"
    plt.savefig(plot_path, dpi=160)
    print(f"Summary CSV: {summary_path}")
    print(f"Plot: {plot_path}")


if __name__ == "__main__":
    main()
