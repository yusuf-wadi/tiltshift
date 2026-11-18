"""New test: VDA-S + flow smoothing.

For t >= 1: w = warp(d'_{t-1}) along DIS flow t->t-1; valid = fwd/bwd consistent (< 1 px) and both depths
> 1% of their median. Align d_t to w by least-squares scale/shift on valid pixels, then
d'_t = a*d~_t + (1-a)*w on valid pixels, d~_t (aligned raw) elsewhere.
fps counts VDA-S inference (from run_models' json) + DIS flow (both directions) + the smoothing itself.
Writes out/<clip>/vda_s_smooth_a{0.5,0.3}.npy/.json.
"""
import json
import time

import numpy as np

from common import CLIPS, OUT, consistent, dis_flow, load_depth, read_clip, save_depth, warp

ALPHAS = [0.5, 0.3]


def smooth(d, flows, alpha):
    out = np.empty_like(d)
    out[0] = d[0]
    for t in range(1, len(d)):
        fwd, bwd = flows[t - 1]
        w = warp(out[t - 1], fwd)
        dt = d[t]
        m = consistent(fwd, bwd) & (dt > 0.01 * np.median(dt)) & (w > 0.01 * np.median(out[t - 1]))
        if m.sum() > 100:
            x, y = dt[m], w[m]
            xm, ym = x.mean(), y.mean()
            a = ((x - xm) * (y - ym)).sum() / max(((x - xm) ** 2).sum(), 1e-12)
            b = ym - a * xm
            al = a * dt + b
        else:
            al = dt
        o = al.copy()
        o[m] = alpha * al[m] + (1 - alpha) * w[m]
        out[t] = o
    return out


def main():
    for clip in CLIPS:
        frames, _ = read_clip(clip)
        T = len(frames)
        d = load_depth(clip, "vda_s")
        base = json.loads((OUT / clip / "vda_s.json").read_text())
        t0 = time.perf_counter()
        flows = [(dis_flow(frames[t], frames[t - 1]), dis_flow(frames[t - 1], frames[t]))
                 for t in range(1, T)]
        flow_s = time.perf_counter() - t0
        for alpha in ALPHAS:
            t0 = time.perf_counter()
            s = smooth(d, flows, alpha)
            smooth_s = time.perf_counter() - t0
            name = f"vda_s_smooth_a{alpha}"
            save_depth(clip, name, s)
            total = base["seconds"] + flow_s + smooth_s
            rec = {"model": name, "clip": clip, "frames": T, "vda_seconds": base["seconds"],
                   "flow_seconds": flow_s, "smooth_seconds": smooth_s, "seconds": total,
                   "fps": T / total, "peak_vram_gb": base["peak_vram_gb"],
                   "out_size": base["out_size"]}
            (OUT / clip / f"{name}.json").write_text(json.dumps(rec, indent=1))
            print(json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()
