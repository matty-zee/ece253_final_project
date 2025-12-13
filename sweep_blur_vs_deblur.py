import argparse
import csv
import subprocess
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt


def run_cmd(cmd, cwd=None):
    subprocess.run(cmd, check=True, cwd=cwd)


def read_mean_ctr(metrics_path):
    total = 0
    acc = 0.0
    with metrics_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            acc += float(row["ctr"])
            total += 1
    if total == 0:
        return 0.0
    return acc / total


def eval_dataset(data_dir, demo_bin, model_param, model_bin, workdir):
    cmd = [
        str(demo_bin),
        str(model_param),
        str(model_bin),
        str(data_dir),
        "--chairs",
    ]
    run_cmd(cmd, cwd=workdir)
    metrics_path = workdir / "letnet_chairs_metrics.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Metrics file not found: {metrics_path}")
    return read_mean_ctr(metrics_path)


def parse_args():
    p = argparse.ArgumentParser(description="Plot LET-NET CTR vs motion blur length (with/without deblur).")
    p.add_argument("--src", type=Path, required=True, help="Source FlyingChairs-style dataset (triplets).")
    p.add_argument("--blur-lengths", type=int, nargs="+", default=[5, 10, 15, 20], help="Motion blur kernel lengths to sweep.")
    p.add_argument("--angle", type=float, default=20.0, help="Motion blur angle in degrees.")
    p.add_argument("--width", type=int, default=1, help="Motion blur kernel thickness.")
    p.add_argument("--max-count", type=int, default=50, help="Number of pairs to process per sweep value (speed).")
    p.add_argument("--deblur", action="store_true", help="Also evaluate after blind deconvolution.")
    p.add_argument("--edge-deblur", action="store_true", help="Also evaluate edge-based deblur.")
    p.add_argument("--edge-amount", type=float, default=1.0, help="Edge deblur boost amount.")
    p.add_argument("--edge-bilateral-d", type=int, default=9, help="Edge deblur bilateral diameter.")
    p.add_argument("--edge-sigma-color", type=float, default=75.0, help="Edge deblur sigmaColor.")
    p.add_argument("--edge-sigma-space", type=float, default=75.0, help="Edge deblur sigmaSpace.")
    p.add_argument("--demo-bin", type=Path, default=Path("LET-NET/build/demo"), help="LET-NET demo binary.")
    p.add_argument("--model-param", type=Path, default=Path("LET-NET/model/model.param"), help="LET-NET model param.")
    p.add_argument("--model-bin", type=Path, default=Path("LET-NET/model/model.bin"), help="LET-NET model bin.")
    p.add_argument("--out-plot", type=Path, default=Path("blur_vs_deblur_ctr.png"), help="Where to save the plot.")
    return p.parse_args()


def main():
    args = parse_args()
    src = args.src.resolve()
    demo_bin = args.demo_bin.resolve()
    model_param = args.model_param.resolve()
    model_bin = args.model_bin.resolve()
    blur_ctrs = []
    deblur_ctrs = []
    edge_ctrs = []

    for length in args.blur_lengths:
        print(f"Processing blur length={length}...")
        with tempfile.TemporaryDirectory() as tmpdir_str:
            tmpdir = Path(tmpdir_str)
            blur_dir = tmpdir / "blurred"
            deblur_dir = tmpdir / "deblurred"
            edge_dir = tmpdir / "edge_deblurred"

            run_cmd(
                [
                    "python3",
                    "motion_blur_dataset.py",
                    "--src",
                    str(src),
                    "--dst",
                    str(blur_dir),
                    "--length",
                    str(length),
                    "--angle",
                    str(args.angle),
                    "--width",
                    str(args.width),
                    "--max-count",
                    str(args.max_count),
                ]
            )
            blur_ctr = eval_dataset(blur_dir, demo_bin, model_param, model_bin, workdir=tmpdir)
            blur_ctrs.append(blur_ctr)
            print(f"  mean CTR blurred: {blur_ctr:.4f}")

            if args.deblur:
                run_cmd(
                    [
                        "python3",
                        "blind_deconv_dataset.py",
                        "--src",
                        str(blur_dir),
                        "--dst",
                        str(deblur_dir),
                        "--max-count",
                        str(args.max_count),
                    ]
                )
                deblur_ctr = eval_dataset(deblur_dir, demo_bin, model_param, model_bin, workdir=tmpdir)
                deblur_ctrs.append(deblur_ctr)
                print(f"  mean CTR deblurred: {deblur_ctr:.4f}")

            if args.edge_deblur:
                run_cmd(
                    [
                        "python3",
                        "edge_deblur_dataset.py",
                        "--src",
                        str(blur_dir),
                        "--dst",
                        str(edge_dir),
                        "--amount",
                        str(args.edge_amount),
                        "--bilateral-d",
                        str(args.edge_bilateral_d),
                        "--sigma-color",
                        str(args.edge_sigma_color),
                        "--sigma-space",
                        str(args.edge_sigma_space),
                        "--max-count",
                        str(args.max_count),
                    ]
                )
                edge_ctr = eval_dataset(edge_dir, demo_bin, model_param, model_bin, workdir=tmpdir)
                edge_ctrs.append(edge_ctr)
                print(f"  mean CTR edge-deblurred: {edge_ctr:.4f}")

    plt.figure(figsize=(8, 5))
    plt.plot(args.blur_lengths, blur_ctrs, marker="o", label="Blurred")
    if args.deblur:
        plt.plot(args.blur_lengths, deblur_ctrs, marker="o", label="Deblurred (blind RL)")
    if args.edge_deblur:
        plt.plot(args.blur_lengths, edge_ctrs, marker="o", label="Deblurred (edge-based)")
    plt.xlabel("Motion blur length (pixels)")
    plt.ylabel("Mean correct tracking ratio")
    plt.title(f"LET-NET CTR vs blur length (angle={args.angle} deg)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out_plot, dpi=200)
    print(f"Plot saved to {args.out_plot}")


if __name__ == "__main__":
    main()
