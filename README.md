# ece253_final_project
Matthew Zheng, Denver Pereira

## Build LET-NET demo
```bash
cd LET-NET
mkdir -p build
cd build
cmake ..
make -j4
```
Ensure OpenCV and ncnn are installed; adjust `ncnn_DIR` in `LET-NET/CMakeLists.txt` if needed.

## Evaluate LET-NET on FlyingChairs
From repo root (or adjust paths accordingly):
- Full dataset:  
  `./LET-NET/build/demo LET-NET/model/model.param LET-NET/model/model.bin FlyingChairs_release/data --chairs`
- Subset with sampling (e.g., 50 pairs, seed 1234):  
  `./LET-NET/build/demo LET-NET/model/model.param LET-NET/model/model.bin FlyingChairs_release/data --chairs 50 1234`
Per-pair metrics are written to `letnet_chairs_metrics.csv` in the working directory.

## Creating FlyingChair Subsets
- First 100 files subset (already generated once): `FlyingChairs_100/data`
- Arbitrary subset (sequential or random):  
  `python subset_flyingchairs.py --src FlyingChairs_release/data --dst FlyingChairs_rand50 --count 50 --mode random --seed 123`

## Applying Artificial Darkening
- Darkened vignette variant:  
  `python make_dark_vignette.py --src FlyingChairs_100/data --dst Flyingchairs_100_dark/data --seed 42 --falloff-scale 3.0`

## Motion blur generation and blind deconvolution
- Add motion blur to a video:  
  `python3 motion_blur_video.py --input input.mp4 --output blurred.mp4 --length 15 --angle 20`
  (Optional flags: `--width` to thicken the PSF line, `--fps` to override detected FPS.)
- Blind deconvolution on an image or video (alternating Richardson-Lucy updates):  
  `python3 blind_deconvolution.py --input blurred.mp4 --output deblurred.mp4`  
  For images, point to a `.png/.jpg`. Algorithm parameters use robust internal defaults now.
- FlyingChairs sweeps / metrics:  
  - Blur a dataset: `python3 motion_blur_dataset.py --src FlyingChairs_100/data --dst FlyingChairs_100_blur/data --length 15 --angle 20` (add `--max-count` to limit pairs).  
  - Deblur: `python3 blind_deconv_dataset.py --src FlyingChairs_100_blur/data --dst FlyingChairs_100_deblur/data --max-count 50`  
  - LET-NET eval with CTR CSV: `./LET-NET/build/demo LET-NET/model/model.param LET-NET/model/model.bin FlyingChairs_100_blur/data --chairs` (writes `letnet_chairs_metrics.csv`, prints mean CTR).  
  - Plot CTR vs blur length (blurred vs deblurred, limited pairs for speed):  
    `python3 sweep_blur_vs_deblur.py --src FlyingChairs_100/data --blur-lengths 5 10 15 20 --angle 20 --max-count 50 --deblur --out-plot blur_vs_deblur_ctr.png`

## Evaluating LET-NET performance
- FlyingChairs evaluation (full or sampled):  
  `./LET-NET/build/demo LET-NET/model/model.param LET-NET/model/model.bin FlyingChairs_release/data --chairs [sample_count] [seed]`
- Custom video/dataset (e.g., library_outside.mp4):  
  `./LET-NET/build/demo LET-NET/model/model.param LET-NET/model/model.bin library_outside_flow/data --chairs`

## CLAHE/ MSR evaluation
- CLAHE parameter sweep:  
  `python plot_clahe_ctr_surface.py --src FlyingChairs_100/data --clip-limits 2.0 3.0 4.0 --tile-sizes 4 6 8 --out-dir clahe_sweep_results`

- MSR parameter sweep:  
  `python evaluate_msr.py --src Dataset/FlyingChairs_100/data --sigma1-range 15 30 5 --sigma2-range 15 30 5 --out-dir Numerical_Results_CLAHE_MSR`

  `--sigma1-range` and `--sigma2-range` has three args: (start, end, step) for evaluating performance for ranges of different variance.
