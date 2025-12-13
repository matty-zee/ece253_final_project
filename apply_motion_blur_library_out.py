import glob
import os
from pathlib import Path
from typing import Tuple

import numpy as np
import cv2
from PIL import Image

DATA_DIR = Path("Library_Out/data")
OUT_DIR = Path("Library_Out/data_motion_blur")
OUT_DIR.mkdir(exist_ok=True)

# Strong horizontal motion blur kernel: long streak across the center row.
# Increase/decrease KERNEL_SIZE for stronger/weaker smearing.
KERNEL_SIZE = 45
kernel = np.zeros((KERNEL_SIZE, KERNEL_SIZE), dtype=np.float32)
mid = KERNEL_SIZE // 2
kernel[mid, :] = 1.0 / KERNEL_SIZE

def find_content_bbox(img: Image.Image) -> Tuple[int, int, int, int]:
    """Return (left, upper, right, lower) bounding box for non-black pixels."""
    arr = np.array(img)
    mask = arr.any(axis=2)
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    if len(rows) == 0 or len(cols) == 0:
        return (0, 0, img.width, img.height)
    return (int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1)

def process_image(path: Path) -> None:
    img = Image.open(path)
    bbox = find_content_bbox(img)
    content = img.crop(bbox)
    content_np = np.array(content)
    blurred_np = cv2.filter2D(content_np, -1, kernel)
    blurred_content = Image.fromarray(blurred_np)
    result = img.copy()
    result.paste(blurred_content, bbox)
    out_path = OUT_DIR / path.name
    result.save(out_path)
    print(f"Saved {out_path}")

def main() -> None:
    files = sorted(glob.glob(str(DATA_DIR / "*_img*.ppm")))
    if not files:
        raise SystemExit("No images found to process.")
    for f in files:
        process_image(Path(f))

if __name__ == "__main__":
    main()
