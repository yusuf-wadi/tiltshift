"""Flicker score for every depth output, along DIS flow and along RAFT (torchvision raft_large).

Per frame t: flow t->t-1 and t-1->t, keep fwd/bwd-consistent pixels (< 1 px) where both depths > 1% of their
median, warp d_{t-1} into t, score = mean |d_t - w| / ((d_t + w)/2). Clip score = median over frames.
Writes out/<clip>/scores.json.
"""
import json

import numpy as np

from common import CLIPS, OUT, Raft, consistent, dis_flow, frame_flicker, load_depth, read_clip

MODELS = ["vda_s", "vda_l", "velodepth", "vda_s_smooth_a0.5", "vda_s_smooth_a0.3"]


def main():
    raft = Raft()
    for clip in CLIPS:
        frames, _ = read_clip(clip)
        models = [m for m in MODELS if (OUT / clip / f"{m}.npy").exists()]
        depths = {m: load_depth(clip, m) for m in models}
        series = {f: {m: [] for m in models} for f in ("dis", "raft")}
        valid_frac = {"dis": [], "raft": []}
        for t in range(1, len(frames)):
            r_fwd, r_bwd = raft.pair(frames[t], frames[t - 1])
            flows = {"dis": (dis_flow(frames[t], frames[t - 1]), dis_flow(frames[t - 1], frames[t])),
                     "raft": (r_fwd, r_bwd)}
            for f, (fwd, bwd) in flows.items():
                valid = consistent(fwd, bwd)
                valid_frac[f].append(float(valid.mean()))
                for m in models:
                    series[f][m].append(frame_flicker(depths[m][t], depths[m][t - 1], fwd, valid))
        res = {"clip": clip, "frames": len(frames),
               "valid_frac_median": {f: float(np.median(v)) for f, v in valid_frac.items()},
               "flicker": {f: {m: float(np.nanmedian(s)) for m, s in series[f].items()} for f in series},
               "series": series}
        (OUT / clip / "scores.json").write_text(json.dumps(res))
        print(clip, json.dumps(res["flicker"]), res["valid_frac_median"], flush=True)


if __name__ == "__main__":
    main()
