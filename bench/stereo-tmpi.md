# Stereo bench: TMPI vs naive warp on VDA-S depth (RTX 5080, 2026-11-18)

**Setup.** Left eye = the input frame. Right eye = rendered at baseline b = 0.04 (max parallax ≈ 2.3% of width) from
VDA-S disparity, normalised with one clip-wide range. Two renderers on the same camera model:
- **TMPI** (Meta, ICCV 2023, CC-BY-NC, research only): tiled multiplane images, 4 planes per 128-px tile. Its
  DPT depth is replaced with VDA-S. Run unmodified through its PyTorch renderer on the GPU.
- **Naive warp:** backward warp by disparity with border stretch (the old `sbsconvert.py` approach).
Each runs on raw VDA-S and on the flow-smoothed VDA-S (α 0.3) from [bench-5080](bench-5080.md).

| Clip | Method | Depth | fps | View flicker (excess over input) | TMPI plane jitter ratio |
|---|---|---|---|---|---|
| Rollercoaster | TMPI | raw | 7.7 | +1.13 | 1.50 |
| | TMPI | smoothed | 7.6 | +0.55 | 1.45 |
| | warp | raw | ~1900 | +1.08 | n/a |
| | warp | smoothed | ~2000 | +0.31 | n/a |
| Tokyo walk | TMPI | raw | 6.3 | −0.08 | 1.74 |
| | TMPI | smoothed | 6.2 | −0.17 | 1.32 |
| | warp | raw | ~1100 | −0.03 | n/a |
| | warp | smoothed | ~1200 | −0.10 | n/a |
| Bears | TMPI | raw | 6.2 | +0.10 | 1.25 |
| | TMPI | smoothed | 6.2 | +0.06 | 1.22 |
| | warp | raw | ~1000 | +0.11 | n/a |
| | warp | smoothed | ~1100 | +0.08 | n/a |

- **View flicker:** median over frames of mean |R_t − warp(R_{t−1})| in 0–255 RGB on RAFT fwd/bwd-consistent pixels
  of the rendered right views. The "excess" subtracts the same measure on the input frames (rollercoaster 2.39,
  Tokyo 3.40, bears 0.31). Negative = smoother than the input, mostly from the views being slightly blurred.
- **Plane jitter ratio:** per tile, the mean frame-to-frame change of TMPI's plane depths divided by the mean change
  of the input disparity in that tile (median over tiles). 1 = planes move only as much as the depth does.

> **Update (same day):** the ground-truth Sintel benchmark ([sintel-stereo](sintel-stereo.md)) refines read #1.
> TMPI beats *naive backward warp* at edges, but a classical z-buffer splat + inpaint beats TMPI overall. TMPI only
> wins inside the disoccluded gaps.

## Read
1. **Edges and gap fill: TMPI wins clearly against naive warp, by eye.** Naive warp shreds thin structure (rollercoaster beams torn into
   triangles) and doubles background objects (bears: duplicated Jeep wheels). TMPI keeps beams whole and fills
   behind edges plausibly, with slight softening. See `out/smoke_tmpi.png` and `out/<clip>/stereo_sheet.png`.
   There's no number here yet: these clips have no ground-truth right view.
2. **Over time, the depth matters more than the renderer.** Smoothing VDA-S cuts the rollercoaster view-flicker
   excess about 2× for TMPI (1.13 → 0.55) and about 3.5× for warp (1.08 → 0.31). On Tokyo and bears both renderers
   are within ~0.1 RGB units of the input's own noise floor.
3. **TMPI's per-frame k-means adds jitter of its own:** its plane depths move 1.2–1.7× as much as the input depth
   does. Smoothed depth lowers it (Tokyo 1.74 → 1.32) but doesn't remove it. Fix: warm-start each tile's k-means
   from the previous frame's planes, carried along flow. That should also make it fast (1 iteration instead of 50).
   This is the flow-caching idea applied to the layers.
4. **Speed:** TMPI 6–8 fps at ~1024 px wide, completely unoptimised (Python tile loop, 50-iteration k-means, per-plane
   scatter renderer). Warp is effectively free. Depth (VDA-S) is ~40 fps, so the renderer is now the bottleneck.

## Caveats
- **The view-flicker metric is weak.** RAFT on the rendered views absorbs geometric wobble (the views move
  consistently, just wrongly), so it mostly catches photometric popping. It doesn't separate TMPI from warp, and
  blur pushes it below the floor. A proper stereo benchmark needs **ground-truth right views** (e.g. MPI-Sintel
  stereo, Spring) to score PSNR/SSIM/LPIPS at edges and in disoccluded regions, plus a geometric-wobble measure.
- TMPI was trained on COCO with DPT depth. Feeding VDA-S works, but it is off-distribution.
- Right-eye-only rendering (left = input) is the worst case for disocclusion. Symmetric ±b/2 halves it.
- 3 short clips, one run each, no ground truth.

## Files
`stereo_views.py` (reusable API: `TMPIStereo`, `warp_stereo`, disparity normalisation), `eval_stereo.py`,
`smoke_tmpi.py`. Outputs: `out/<clip>/stereo_*.json`, `out/<clip>/sbs_*.mp4`, `out/<clip>/stereo_sheet.png`.
TMPI itself: clone `facebookresearch/tmpi` into `bench/tmpi` (weights included).
