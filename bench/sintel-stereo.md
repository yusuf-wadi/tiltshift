# Ground-truth stereo benchmark: MPI-Sintel (RTX 5080, 2026-11-18)

**Task:** from the left frame alone, synthesise the right view (true camera 10 cm to the right) and score it against
the real right frame. MPI-Sintel stereo training split, final pass: 23 sequences, 1,064 frames, 1024×436.
**Depth:** *oracle* = true disparity (isolates the renderer and its gap filling); *VDA-S* = Video Depth Anything
Small on the left video, with one scale + shift per sequence fitted to the true disparity (full pipeline, fair scale).
**Methods:** naive warp (backward warp by disparity), **splat** (z-buffer forward splat + OpenCV Telea inpaint for the
gaps), **TMPI** (Meta, tiled multiplane images, 4 planes per 128-px tile). Identity (right = left) scores 16.4 dB /
LPIPS 0.369 as the floor.
**Regions:** *disocc* = right-view pixels no visible left pixel lands on (must be invented); *edges* = true depth
edges (±2 px); the out-of-frame strip at the right edge is excluded everywhere.

## All 1,064 frames

| Depth | Method | PSNR ↑ | PSNR gaps ↑ | PSNR edges ↑ | SSIM ↑ | LPIPS ↓ | Temporal err ↓ | ms/frame |
|---|---|---|---|---|---|---|---|---|
| oracle | warp | 21.99 | **18.39** | 17.72 | 0.791 | 0.187 | 9.82 | 0.8 |
| oracle | splat | **25.27** | 16.39 | **24.94** | **0.907** | **0.124** | **6.02** | 85 |
| oracle | TMPI | 23.68 | 17.88 | 21.18 | 0.815 | 0.175 | 7.51 | 122 |
| VDA-S | warp | 20.31 | **16.94** | 16.99 | 0.703 | 0.207 | 12.58 | 0.8 |
| VDA-S | splat | **21.40** | 16.89 | **18.52** | **0.728** | **0.170** | 11.47 | 92 |
| VDA-S | TMPI | 20.34 | 16.61 | 17.58 | 0.683 | 0.220 | **11.39** | 121 |

## By parallax size (share of the right view that is disoccluded)

| Frames | Depth | warp PSNR / gaps | splat PSNR / gaps | TMPI PSNR / gaps | splat vs TMPI LPIPS |
|---|---|---|---|---|---|
| small < 2% (100) | oracle | 22.8 / 17.5 | **29.9** / 18.1 | 28.2 / **20.9** | 0.062 vs 0.120 |
| | VDA-S | 20.6 / 16.3 | **21.8** / 16.8 | 20.3 / 16.7 | 0.144 vs 0.193 |
| medium 2–6% (328) | oracle | 25.6 / **20.5** | **28.5** / 17.8 | 26.6 / 19.2 | 0.075 vs 0.147 |
| | VDA-S | 22.5 / 17.7 | **23.0** / 17.5 | 21.8 / 17.0 | 0.113 vs 0.183 |
| large > 6% (636) | oracle | 20.0 / **17.4** | **22.9** / 15.4 | 21.5 / 16.7 | 0.158 vs 0.198 |
| | VDA-S | 19.1 / 16.7 | **20.5** / 16.6 | 19.6 / 16.4 | 0.204 vs 0.244 |

Temporal error = mean |(P_t − P_{t−1}) − (G_t − G_{t−1})| over valid pixels, 0–255 RGB, against the true right video.

## Read
1. **Classical splat + inpaint beats TMPI overall** at every parallax size, with true or predicted depth: higher
   PSNR, SSIM, edge PSNR; lower LPIPS (TMPI better on LPIPS in only 47% of oracle frames, 30% with VDA-S).
   TMPI's 4 planes per tile can't represent exact depth, so covered pixels and edges come out softened and slightly
   misplaced.
2. **TMPI's real strength is the gaps.** With true depth and small parallax, it fills the disoccluded regions
   +2.8 dB better than Telea inpaint (20.9 vs 18.1). Naive warp scores well in gaps on PSNR only because stretched
   background is a blurry average, which PSNR rewards.
3. **Depth is the biggest lever by far.** On small-parallax frames, swapping true disparity for VDA-S costs splat
   8.1 dB (29.9 → 21.8) and narrows all method differences to ~1.5 dB. Better or metric depth (VDA Metric-Small,
   temporal alignment, the smoothing work) is worth more than a better renderer.
4. **Obvious next experiment: a hybrid.** Splat for every pixel that has a source, TMPI (or another learned
   inpainter) only inside the holes. That should take splat's 29.9 dB overall plus TMPI's +2.8 dB in the gaps.
5. **This corrects the earlier TMPI read.** "TMPI clearly beats warp at edges" holds against *naive backward warp*,
   but a proper z-buffer splat is better than both outside the gaps.

## Caveats
- Sintel's parallax is extreme (up to ~135 px, 13% of the width; 60% of frames have > 6% disoccluded). Consumer
  stereo is ~2–3%, so the "small" bin is the most product-relevant and has only 100 frames.
- Sintel is synthetic (Blender), and the final pass has motion blur and fog. TMPI was trained on COCO with DPT depth;
  VDA-S was not tuned on Sintel.
- VDA-S is aligned with one scale + shift per sequence using the true disparity, so it's an upper bound for
  relative depth without ground truth.
- PSNR in the gaps rewards blur. Masked perceptual scores (LPIPS on the gaps) would be fairer to TMPI.

## Files
`sintel_stereo.py` (benchmark), `sintel_breakdown.py` (parallax bins), `stereo_views.py` (adds `splat_stereo`).
Results: `out/sintel/summary.json`, `out/sintel/breakdown.json`, per-frame JSON per method. Data:
`~/datasets/sintel-stereo` (download link in the script header).
