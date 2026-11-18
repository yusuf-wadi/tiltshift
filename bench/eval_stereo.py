"""Stereo benchmark on video: TMPI vs naive warp, on VDA-S depth (raw and flow-smoothed).

For each clip x method, render the right view (b = 0.04; left = the input frame) for every frame, then:
- speed: ms per frame
- view flicker: RAFT flow between consecutive right views, fwd/bwd-consistent pixels, mean |R_t - warp(R_{t-1})|
  in 0-255 RGB units. Same measure on the input frames = noise floor; report excess over it.
  Also the consistent-pixel fraction (tearing and popping break consistency).
- TMPI plane jitter: per tile, mean |change| of the 3 non-background plane disparities between frames, divided
  by the mean |change| of the input disparity in that tile. 1 = planes move as much as the depth does; >1 = the
  per-frame k-means adds jumps of its own.
Writes out/<clip>/stereo_<method>.json, out/<clip>/sbs_<method>.mp4 and out/<clip>/stereo_sheet.png.
Disparity is normalised with one clip-wide range (per-frame min/max would add flicker of its own).
"""
import json
import sys
import time

import cv2
import imageio.v2 as imageio
import numpy as np

from common import CLIPS, OUT, Raft, consistent, load_depth, read_clip, warp
from stereo_views import TMPIStereo, clip_disp_range, normalize_disp, warp_stereo

B = 0.04
METHODS = [("tmpi", "vda_s"), ("tmpi", "vda_s_smooth_a0.3"), ("warp", "vda_s"), ("warp", "vda_s_smooth_a0.3")]


def view_flicker(raft, views):
    errs, fracs = [], []
    for t in range(1, len(views)):
        fwd, bwd = raft.pair(views[t], views[t - 1])
        m = consistent(fwd, bwd)
        w = warp(views[t - 1].astype(np.float32), fwd)
        e = np.abs(views[t].astype(np.float32) - w).mean(-1)
        errs.append(float(e[m].mean()) if m.sum() > 100 else np.nan)
        fracs.append(float(m.mean()))
    return float(np.nanmedian(errs)), float(np.median(fracs)), errs


def tile_mean(x, tile, step):
    H, W = x.shape
    out = []
    for y in range(0, H, step):
        for xx in range(0, W, step):
            out.append(x[y:y + tile, xx:xx + tile].mean())
    return np.array(out)


def main(clips):
    raft = Raft()
    tm = TMPIStereo()
    for clip in clips:
        frames, fps = read_clip(clip)
        T, H, W = frames.shape[:3]
        floor, floor_frac, _ = view_flicker(raft, frames)
        results = {}
        sheet_rows = []
        for method, depth_name in METHODS:
            d = load_depth(clip, depth_name)
            lo, hi = clip_disp_range(d)
            disps = [normalize_disp(x, lo, hi) for x in d]
            rights, planes, ms = [], [], []
            for t in range(T):
                t0 = time.perf_counter()
                if method == "tmpi":
                    rights.append(tm(frames[t], disps[t], baselines=[B])[0])
                    planes.append(tm.last_planes[:, :3].copy())
                else:
                    rights.append(warp_stereo(frames[t], disps[t], B))
                ms.append((time.perf_counter() - t0) * 1e3)
            flick, frac, series = view_flicker(raft, rights)
            rec = {"clip": clip, "method": method, "depth": depth_name, "baseline": B, "frames": T,
                   "ms_per_frame": float(np.median(ms[1:])), "fps": 1e3 / float(np.median(ms[1:])),
                   "view_flicker": flick, "input_flicker_floor": floor, "excess_flicker": flick - floor,
                   "consistent_frac": frac, "input_consistent_frac": floor_frac}
            if method == "tmpi":
                rec.update({k: v for k, v in tm.timing.items() if k in ("tile_sz", "n_tiles", "work_size")})
                # plane jitter vs input-disparity change, per tile, on TMPI's working grid
                ts, (h, w) = tm.timing["tile_sz"], tm.timing["work_size"]
                step = ts - int(ts * 0.125)
                small = [cv2.resize(x, (w, h), interpolation=cv2.INTER_AREA) for x in disps]
                dp = np.mean([np.abs(planes[t] - planes[t - 1]).mean(1) for t in range(1, T)], 0)
                dd = np.mean([np.abs(tile_mean(small[t], ts, step) - tile_mean(small[t - 1], ts, step))
                              for t in range(1, T)], 0)
                n = min(len(dp), len(dd))
                rec["plane_jitter_ratio"] = float(np.median(dp[:n] / np.maximum(dd[:n], 1e-4)))
                rec["plane_change_mean"] = float(dp.mean())
                rec["input_disp_change_mean"] = float(dd.mean())
            name = f"{method}_{depth_name}"
            (OUT / clip / f"stereo_{name}.json").write_text(json.dumps(rec, indent=1))
            print(json.dumps(rec), flush=True)
            imageio.mimwrite(OUT / clip / f"sbs_{name}.mp4",
                             [np.hstack([frames[t], rights[t]]) for t in range(T)],
                             fps=fps, quality=8, macro_block_size=8)
            t = T // 2
            img = cv2.cvtColor(rights[t], cv2.COLOR_RGB2BGR)
            cv2.putText(img, name, (10, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            sheet_rows.append(img)
            results[name] = rec
        top = cv2.cvtColor(frames[T // 2], cv2.COLOR_RGB2BGR)
        cv2.putText(top, "input (left eye)", (10, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        grid = np.vstack([np.hstack([top, sheet_rows[0]]), np.hstack([sheet_rows[1], sheet_rows[2]]),
                          np.hstack([sheet_rows[3], np.zeros_like(top)])])
        s = 1600 / grid.shape[1]
        cv2.imwrite(str(OUT / clip / "stereo_sheet.png"), cv2.resize(grid, None, fx=s, fy=s))


if __name__ == "__main__":
    main(sys.argv[1:] or list(CLIPS))
