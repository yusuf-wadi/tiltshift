"""Eyeball sheet for lag/ghosting: frames x variants, plus an x-t slice (one image row stacked over time).

Writes out/<clip>/sheet.png and out/<clip>/xt.png.  Usage: python viz.py [clip]   (default rollercoaster)
"""
import sys

import cv2
import numpy as np

from common import OUT, load_depth, read_clip

VARIANTS = ["vda_s", "vda_s_smooth_a0.5", "vda_s_smooth_a0.3", "velodepth"]


def norm(d, inverse):
    x = 1.0 / np.maximum(d, 1e-6) if inverse else d  # show everything as disparity
    lo, hi = np.percentile(x, [2, 98])
    return np.clip((x - lo) / max(hi - lo, 1e-9), 0, 1)


def color(x):
    return cv2.applyColorMap((x * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)


def label(img, text):
    img = img.copy()
    cv2.putText(img, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return img


def main(clip):
    frames, _ = read_clip(clip)
    T, H, W = frames.shape[:3]
    ds = {v: norm(load_depth(clip, v), v == "velodepth") for v in VARIANTS}
    ts = np.linspace(T * 0.25, T - 2, 4).astype(int)
    rows = [np.hstack([label(cv2.cvtColor(frames[t], cv2.COLOR_RGB2BGR), f"t={t}") for t in ts])]
    rows += [np.hstack([label(color(ds[v][t]), v) for t in ts]) for v in VARIANTS]
    sheet = np.vstack(rows)
    s = 1600 / sheet.shape[1]
    cv2.imwrite(str(OUT / clip / "sheet.png"), cv2.resize(sheet, None, fx=s, fy=s))
    y = H // 2
    xt = [label(cv2.resize(cv2.cvtColor(frames[:, y], cv2.COLOR_RGB2BGR), (W, T * 3),
                           interpolation=cv2.INTER_NEAREST), "rgb row")]
    xt += [label(cv2.resize(color(ds[v][:, y]), (W, T * 3), interpolation=cv2.INTER_NEAREST), v)
           for v in VARIANTS]
    cv2.imwrite(str(OUT / clip / "xt.png"), np.vstack(xt))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "rollercoaster")
