import argparse
import random
import shutil
from pathlib import Path


def find_bases(src: Path):
    # Use *_flow.flo as the anchor to ensure all components exist.
    bases = []
    for flow_path in src.glob("*_flow.flo"):
        stem = flow_path.stem
        if stem.endswith("_flow"):
            bases.append(stem[:-5])  # remove '_flow'
    return sorted(bases)


def copy_triplet(base: str, src: Path, dst: Path):
    img1 = src / f"{base}_img1.ppm"
    img2 = src / f"{base}_img2.ppm"
    flow = src / f"{base}_flow.flo"
    if not (img1.exists() and img2.exists() and flow.exists()):
        print(f"Skipping {base}: missing one of the required files.")
        return False
    shutil.copy2(img1, dst / img1.name)
    shutil.copy2(img2, dst / img2.name)
    shutil.copy2(flow, dst / flow.name)
    return True


def main():
    parser = argparse.ArgumentParser(description="Subset a FlyingChairs dataset.")
    parser.add_argument("--src", type=Path, default=Path("FlyingChairs_release/data"),
                        help="Source directory containing FlyingChairs triplets.")
    parser.add_argument("--dst", type=Path, required=True,
                        help="Destination directory to create (triplets copied inside).")
    parser.add_argument("--count", type=int, default=100,
                        help="Number of triplets to include in the subset.")
    parser.add_argument("--mode", choices=["sequential", "random"], default="sequential",
                        help="Selection mode for picking triplets.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Seed used when mode is random.")
    args = parser.parse_args()

    src = args.src
    dst = args.dst

    if not src.exists():
        raise FileNotFoundError(f"Source directory not found: {src}")

    dst.mkdir(parents=True, exist_ok=True)

    bases = find_bases(src)
    if not bases:
        raise FileNotFoundError(f"No *_flow.flo files found in {src}")

    if args.mode == "random":
        random.seed(args.seed)
        selected = random.sample(bases, k=min(args.count, len(bases)))
        print(f"Randomly selected {len(selected)} triplets (seed={args.seed}).")
    else:
        selected = bases[: args.count]
        print(f"Selected first {len(selected)} triplets (sorted order).")

    copied = 0
    for base in selected:
        if copy_triplet(base, src, dst):
            copied += 1

    print(f"Done. Copied {copied} triplets to {dst}")


if __name__ == "__main__":
    main()
