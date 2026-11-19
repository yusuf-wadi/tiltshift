"""Sintel subset in iw3's normal range: the 6 sequences whose 99.9th-pct disparity is <= 6.5% of the width
(mountain_1, alley_2, market_2, cave_4, sleeping_2, cave_2; ~4% is a strong consumer 3D setting).
Writes out/sintel/lowparallax.json."""
import json

import numpy as np

from common import OUT

S = {"mountain_1", "alley_2", "market_2", "cave_4", "sleeping_2", "cave_2"}
BASE = ["warp", "splat", "tmpi", "iw3_forward_fill", "iw3_forward_inpaint", "iw3_row_flow_v3", "iw3_mlbw_l2",
        "iw3_mlbw_l4", "iw3_mlbw_l2_inpaint"]
res = {}
for mode in ("oracle", "vda_s"):
    for b in BASE:
        p = OUT / "sintel" / f"{mode}_{b}.json"
        if not p.exists():
            continue
        fr = [r for r in json.loads(p.read_text())["frames"] if r["seq"] in S]
        res[f"{mode}_{b}"] = {k: float(np.nanmean([r.get(k, np.nan) for r in fr]))
                              for k in ("psnr", "psnr_disocc", "psnr_edges", "ssim", "lpips", "temporal_err")}
        res[f"{mode}_{b}"]["frames"] = len(fr)
        x = res[f"{mode}_{b}"]
        print(f"{mode:6s} {b:20s} n{len(fr)} psnr {x['psnr']:.2f} gaps {x['psnr_disocc']:.2f} "
              f"edges {x['psnr_edges']:.2f} ssim {x['ssim']:.3f} lpips {x['lpips']:.3f} temp {x['temporal_err']:.2f}")
(OUT / "sintel" / "lowparallax.json").write_text(json.dumps(res, indent=1))
