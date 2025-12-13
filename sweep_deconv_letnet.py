import csv
import os
import shutil
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt

from blind_deconv_dataset import process_dataset


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
    # With simplified blind deconvolution defaults, configs are labels only.
    configs = [
        {"name": "default_run_1", "max_pairs": 10},
        {"name": "default_run_2", "max_pairs": 10},
        {"name": "default_run_3", "max_pairs": 10},
    ]

    results = []
    DEST_ROOT.mkdir(parents=True, exist_ok=True)

    for cfg in configs:
        dst = DEST_ROOT / cfg["name"]
        print(f"Processing config {cfg['name']} -> {dst}")
        metrics_file = dst / "letnet_chairs_metrics.csv"
        if not metrics_file.exists():
            process_dataset(
                src=SRC_DIR,
                dst=dst,
                max_count=cfg["max_pairs"],
            )
            mean_ctr = run_letnet_chairs(dst)
        else:
            print("Metrics already present, skipping recompute.")
            mean_ctr = read_mean_ctr_from_file(metrics_file)
        print(f"{cfg['name']}: mean CTR {mean_ctr:.4f}")
        row = {"name": cfg["name"], "max_pairs": cfg["max_pairs"], "mean_ctr": mean_ctr}
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
