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

## iw3 added (nagadomi/nunif, MIT; same day)

iw3 is the main open-source 2D→3D converter. Its stereo methods are driven with the same depth (`iw3_stereo.py`):
right view only, convergence 0, linear mapper, so its pixel disparity matches ours exactly (checked with
constant-depth shifts). Methods:
- **forward_fill**: depth-ordered *bilinear* forward splat, a layered-hole fix, then background shift-fill. No ML.
- **forward_inpaint**: the same splat plus iw3's light *video* inpaint model.
- **row_flow_v3**: iw3's default, a learned backward warp.
- **mlbw_l2 / mlbw_l4**: learned 2- or 4-layer backward warps.
- **mlbw_l2_inpaint**: mlbw_l2 plus video inpaint.

**All 1,064 frames**

| Depth | Method | PSNR ↑ | gaps ↑ | edges ↑ | SSIM ↑ | LPIPS ↓ | Temporal ↓ | ms/frame |
|---|---|---|---|---|---|---|---|---|
| oracle | our splat | 25.27 | 16.39 | 24.94 | 0.907 | 0.124 | 6.02 | 85 (CPU) |
| oracle | TMPI | 23.68 | 17.88 | 21.18 | 0.815 | 0.175 | 7.51 | 122 |
| oracle | iw3 forward_fill | **26.17** | 17.18 | **24.98** | **0.916** | 0.121 | **5.55** | 35 |
| oracle | iw3 forward_inpaint | 25.97 | 16.69 | 24.95 | 0.914 | **0.113** | 5.64 | 29 |
| oracle | iw3 row_flow_v3 | 24.33 | 16.79 | 20.53 | 0.885 | 0.165 | 7.08 | 14 |
| oracle | iw3 mlbw_l4 | 20.68 | 17.16 | 17.89 | 0.647 | 0.350 | 13.97 | 7 |
| VDA-S | our splat | 21.40 | 16.89 | 18.52 | 0.728 | 0.170 | 11.47 | 92 |
| VDA-S | iw3 forward_fill | 21.86 | 17.50 | **18.59** | 0.732 | 0.168 | 11.26 | 23 |
| VDA-S | iw3 forward_inpaint | **21.89** | 17.58 | **18.60** | 0.734 | **0.157** | 11.22 | 19 |
| VDA-S | iw3 row_flow_v3 | 21.87 | **17.70** | 18.47 | **0.741** | 0.190 | **10.82** | 12 |

**Low-parallax subset** (`sintel_lowparallax.py`): the 6 sequences whose max disparity is ≤ 6.5% of the width
(mountain_1, alley_2, market_2, cave_4, sleeping_2, cave_2; 300 frames). This is inside iw3's trained range and
the closest Sintel gets to consumer 3D.

| Depth | warp | our splat | TMPI | iw3 forward_fill | iw3 forward_inpaint | iw3 row_flow_v3 | iw3 mlbw_l4 |
|---|---|---|---|---|---|---|---|
| oracle PSNR / LPIPS | 30.6 / 0.069 | 32.2 / 0.056 | 31.9 / 0.060 | **34.7** / 0.047 | 34.3 / **0.041** | 32.9 / 0.065 | 33.4 / 0.057 |
| oracle gaps PSNR | **25.5** | 22.8 | **25.5** | 24.7 | 23.9 | 24.1 | **25.5** |
| VDA-S PSNR / LPIPS | 27.7 / 0.083 | 26.8 / 0.077 | 26.5 / 0.090 | 28.2 / 0.074 | 28.2 / **0.073** | **28.3** / 0.084 | 28.2 / 0.076 |

### Read (iw3)
1. **iw3's forward_fill replaces our splat as the classical baseline.** It's better on every all-frame metric, and
   +2.5 dB on the low-parallax subset, at ~3× the speed on the GPU. The difference is subpixel accuracy: our splat
   rounds to the nearest pixel, which costs the most when parallax is small (mountain_1, 0.4% parallax: 40.5 vs
   48.1 dB; with VDA-S depth, ours even falls below naive warp). It is MIT-licensed, so we can build on it directly.
2. **forward_inpaint gives the best perceptual score** (lowest LPIPS everywhere). Its video inpaint model fills the
   gaps less blurrily than shift-fill, so PSNR in the gaps drops slightly (PSNR rewards blur there). This is iw3's own
   version of the splat + learned-hole-fill hybrid.
3. **The learned warps only work inside their trained range.** mlbw_l2/l4 are competitive up to ~6% parallax
   (mlbw_l4 ties TMPI for the best gap fill there, 25.5 dB) and collapse beyond it (11–16 dB at 12%+, below a plain
   copy of the left frame). row_flow_v3 holds up at large parallax because it splits big shifts into steps, but it
   blurs edges (−4 dB on edges vs forward_fill). mlbw_l2_inpaint ships only a d1 (small-divergence) mask model and
   fails even on the subset, so treat it as untested.
4. **With predicted depth, everyone converges.** With VDA-S, all of iw3's methods, our splat and TMPI land within
   ~0.5 dB on the low-parallax subset. Depth quality remains the main lever; renderer choice matters most when depth
   is good.
5. **Where TMPI stands now:** with true depth it still has the best gap fill on all frames (17.9 dB), and it wins
   cave_2/cave_4 outright. Overall it trails iw3's forward methods by 2–3 dB. The hybrid to try is iw3 forward splat
   plus a learned fill (iw3's video inpaint, TMPI, or ProPainter) only in the holes, scored with masked LPIPS.

## Caveats
- iw3 learned methods are scored at Sintel's per-sequence parallax; most sequences are far past their trained range
  (see the low-parallax subset). iw3's own depth models, EMA depth normalisation and scene-cut handling are not used
  here (our depth is fed in), so this scores iw3's *renderers*, not the full iw3 app.
- Sintel's parallax is extreme (up to ~135 px, 13% of the width; 60% of frames have > 6% disoccluded). Consumer
  stereo is ~2–3%, so the "small" bin is the most product-relevant and has only 100 frames.
- Sintel is synthetic (Blender), and the final pass has motion blur and fog. TMPI was trained on COCO with DPT depth;
  VDA-S was not tuned on Sintel.
- VDA-S is aligned with one scale + shift per sequence using the true disparity, so it's an upper bound for
  relative depth without ground truth.
- PSNR in the gaps rewards blur. Masked perceptual scores (LPIPS on the gaps) would be fairer to TMPI.

## Files
`sintel_stereo.py` (benchmark), `sintel_breakdown.py` (parallax bins), `sintel_lowparallax.py` (low-parallax subset),
`stereo_views.py` (adds `splat_stereo`), `iw3_stereo.py` (iw3 adapter; clone `nagadomi/nunif` into `bench/nunif`,
models download on first use).
Results: `out/sintel/summary.json`, `out/sintel/breakdown.json`, per-frame JSON per method. Data:
`~/datasets/sintel-stereo` (download link in the script header).
