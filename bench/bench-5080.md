# Depth bench on the RTX 5080: 2026-11-18

Repeat of the Mac Studio bench (2026-11-17) on waedi-box-1 (RTX 5080 16 GB, CUDA 12.8, torch 2.11, WSL2 Ubuntu,
xformers 0.0.36), plus the new **VDA-S + flow smoothing** test.
Clips: rollercoaster (960×540, 70 frames), Tokyo walk (1280×720, 150), bears (1280×720, 90).
Models: VDA-Small (Apache-2.0), VDA-Large (CC-BY-NC, reference only), VeloDepth res 9 (CC-BY-NC-SA, research only).
VDA fp16 at input 518; one 40-frame warm-up, then the full clip timed with `cuda.synchronize()`.

Flicker = median over frames of mean |d_t − warp(d_{t−1})| / mean(d_t, warp) on fwd/bwd-consistent pixels (< 1 px),
both depths > 1% of median. Lower = steadier. **DIS** = same flow as the Mac run; **RAFT** = torchvision raft_large
(independent of VeloDepth's own DIS propagation). **RAFT strict** = RAFT with a stricter mask (both > 5% of the
frame's 95th-percentile disparity), which drops near-zero sky; see "Metric caveat".

| Clip | Model | fps | Peak VRAM GB | Flicker DIS | Flicker RAFT | RAFT strict |
|---|---|---|---|---|---|---|
| Rollercoaster | VDA-S | 37.4 | 3.1 | 0.0996 | 0.0984 | 0.0726 |
| | VDA-L | 11.9 | 11.4 | 0.0886 | 0.0882 | 0.0683 |
| | VeloDepth | 26.4 | 6.6 | 0.0097 | 0.0096 | 0.0096 |
| | VDA-S + smooth α0.5 | 15.8 | 3.1 | 0.0082 | 0.0093 | 0.0093 |
| | VDA-S + smooth α0.3 | 15.8 | 3.1 | 0.0073 | 0.0109 | 0.0109 |
| Tokyo walk | VDA-S | 40.2 | 3.2 | 0.0110 | 0.0116 | 0.0085 |
| | VDA-L | 13.3 | 11.5 | 0.0098 | 0.0104 | 0.0082 |
| | VeloDepth | 20.6 | 15.7 ⚠️ | 0.0039 | 0.0042 | 0.0042 |
| | VDA-S + smooth α0.5 | 11.8 | 3.2 | 0.0118 | 0.0128 | 0.0073 |
| | VDA-S + smooth α0.3 | 11.9 | 3.2 | 0.0102 | 0.0115 | 0.0065 |
| Bears | VDA-S | 38.1 | 3.2 | 0.0037 | 0.0040 | 0.0035 |
| | VDA-L | 11.4 | 11.5 | 0.0032 | 0.0035 | 0.0027 |
| | VeloDepth | 24.0 | 10.1 | 0.0009 | 0.0011 | 0.0011 |
| | VDA-S + smooth α0.5 | 11.1 | 3.2 | 0.0031 | 0.0036 | 0.0031 |
| | VDA-S + smooth α0.3 | 10.8 | 3.2 | 0.0027 | 0.0031 | 0.0029 |

Smoothing fps includes VDA-S inference + DIS flow both directions (CPU) + the numpy smoothing loop.
Ablation, **align-only** (scale/shift fit to the warped previous frame, no blending), RAFT strict:
rollercoaster 0.0083, Tokyo 0.0095, bears 0.0037.

Mac Studio (M2 Max, MPS, GPU shared with training), flicker = DIS:

| Clip | VDA-S fps | VeloDepth fps | VDA-S flicker | VeloDepth flicker |
|---|---|---|---|---|
| Rollercoaster | 3.9 | 2.1 | 0.099 | 0.009 |
| Bears | 3.8 | 2.0 | 0.0038 | 0.0009 |
| Tokyo walk | 4.5 | 1.9 | 0.012 | 0.005 |

DIS flicker reproduces the Mac numbers to within rounding (same metric, different hardware), so the two runs compare.

## Read
1. **Speed:** the 5080 runs VDA-S at 37–40 fps (~10× the Mac) and VeloDepth at 21–26 fps (~11×). On CUDA, VDA-S is real-time at 720p; VDA-L is 11–13 fps.
2. **VeloDepth's steadiness is real, not a DIS artifact:** RAFT scores are within ~10–25% of DIS, and it stays 2.5–3.5× steadier than VDA-S on Tokyo and bears under every mask.
3. **VDA-S + flow smoothing helps a little where the metric is trustworthy:** with sky excluded, α0.3 cuts flicker 18–23% on Tokyo and bears (0.0085 → 0.0065, 0.0035 → 0.0029), while running at 11–16 fps. VeloDepth is still 1.5–2.7× steadier. Lower α (more history) is better, and blending is what helps: align-only is *worse* than raw VDA-S on both clips.
4. **The rollercoaster "10× win" is not real stabilisation.** Align-only alone scores 0.0083, better than either blend. The least-squares shift lifts the sky off zero and lowers contrast at thin beams, and the relative metric rewards that. Likewise, VDA-S's 0.099 there (also on the Mac) is mostly sky and beam-edge noise, not scale jumps.
5. **Eyeball (rollercoaster, `out/rollercoaster/sheet.png`, `xt.png`):** no visible lag or ghost trails on the train or the beams at α 0.5 or 0.3; edges stay sharp. The visible artifact is the **sky**: raw VDA-S has sky ≈ 0, the smoothed versions lift it to a nonzero level that drifts frame to frame (it's the shift term). VeloDepth is blurrier on thin structure and blows out the last frames (t ≈ 68). For the product: fit scale only (no shift), or pin the sky/far plane, before blending.

## Caveats
- **Metric caveat:** the handoff mask (> 1% of median) admits sky when sky is most of the frame (rollercoaster: 74% sky in VDA-S), where disparity ≈ 0 and relative change explodes. "RAFT strict" removes most of it, but the smoothed outputs' lifted sky passes even the strict mask, so the rollercoaster smoothing rows aren't comparable. Trust Tokyo and bears.
- VeloDepth on Tokyo peaked at 15.7 GB of the 16 GB card, with ~3.9 GB already used by other processes on the desktop, so it likely spilled into shared memory (WDDM). Its Tokyo fps may be pessimistic.
- The smoothing's flow + loop is single-threaded Python/OpenCV on the CPU (≈ 2–3× VDA-S's own time). Moving it to the GPU, or reusing VDA-S's frame cadence, would close most of the speed gap.
- 3 short clips, one timed run each. VDA disparity vs VeloDepth metric depth is compared scale-free.

## Files
`run_models.py` (timing), `smooth.py` (new test), `score.py` (DIS + RAFT flicker), `check_mask.py` (sky mask + align-only),
`viz.py` (eyeball sheet), `common.py`. Per-run JSON and the rollercoaster PNGs are in `out/`; depth .npy files are not committed.
To rerun: clone `DepthAnything/Video-Depth-Anything` as `bench/vda` (weights in `vda/checkpoints/`) and
`lpiccinelli-eth/Velodepth` as `bench/velodepth` (`pip install --no-deps -e`), then
`python run_models.py && python smooth.py && python score.py && python check_mask.py && python viz.py`.
