"""Robustness check on the flicker metric (RAFT flow).

The handoff mask keeps pixels with depth > 1% of the median. When sky fills most of the frame, VDA's sky
disparity is ~0, the median is ~0 and the sky passes the mask, so tiny absolute noise becomes huge relative
change. Here: (1) sky share per clip, (2) flicker with a stricter mask (both > 5% of the frame's 95th
percentile, in disparity space so VeloDepth depth is inverted first), (3) align-only ablation (alpha = 1:
scale/shift alignment to the warped previous frame, no blending).
Writes out/<clip>/check.json.
"""
import json

import numpy as np

from common import CLIPS, OUT, Raft, consistent, dis_flow, load_depth, read_clip, warp
from smooth import smooth

MODELS = ["vda_s", "vda_l", "velodepth", "vda_s_smooth_a0.5", "vda_s_smooth_a0.3", "vda_s_align_only"]


def disp(d, model):
    return 1.0 / np.maximum(d, 1e-6) if model == "velodepth" else d


def main():
    raft = Raft()
    for clip in CLIPS:
        frames, _ = read_clip(clip)
        T = len(frames)
        d = {m: load_depth(clip, m) for m in MODELS if m != "vda_s_align_only"}
        dis = [(dis_flow(frames[t], frames[t - 1]), dis_flow(frames[t - 1], frames[t])) for t in range(1, T)]
        d["vda_s_align_only"] = smooth(d["vda_s"], dis, 1.0)
        raw = d["vda_s"]
        sky = float(np.mean(raw < 0.05 * np.percentile(raw, 95, axis=(1, 2), keepdims=True)))
        loose = {m: [] for m in MODELS}
        strict = {m: [] for m in MODELS}
        for t in range(1, T):
            fwd, bwd = raft.pair(frames[t], frames[t - 1])
            valid = consistent(fwd, bwd)
            for m in MODELS:
                a, b = d[m][t], warp(d[m][t - 1], fwd)
                rel = np.abs(a - b) / np.maximum((a + b) / 2, 1e-12)
                ml = valid & (a > 0.01 * np.median(a)) & (b > 0.01 * np.median(d[m][t - 1]))
                pa, pb = disp(a, m), disp(b, m)
                thr_a = 0.05 * np.percentile(disp(d[m][t], m), 95)
                thr_b = 0.05 * np.percentile(disp(d[m][t - 1], m), 95)
                ms = valid & (pa > thr_a) & (pb > thr_b)
                loose[m].append(float(rel[ml].mean()) if ml.sum() > 100 else np.nan)
                strict[m].append(float(rel[ms].mean()) if ms.sum() > 100 else np.nan)
        res = {"clip": clip, "sky_frac_vda_s": sky,
               "raft_loose": {m: float(np.nanmedian(v)) for m, v in loose.items()},
               "raft_strict": {m: float(np.nanmedian(v)) for m, v in strict.items()}}
        (OUT / clip / "check.json").write_text(json.dumps(res, indent=1))
        print(json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
