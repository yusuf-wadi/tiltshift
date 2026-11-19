"""Break the Sintel results down by how much of the right view is disoccluded (a proxy for parallax size)."""
import json

import numpy as np

from common import OUT

BASE = ["warp", "splat", "tmpi", "iw3_forward_fill", "iw3_forward_inpaint", "iw3_row_flow_v3", "iw3_mlbw_l2",
        "iw3_mlbw_l4", "iw3_mlbw_l2_inpaint"]
KEYS = [f"{m}_{b}" for m in ("oracle", "vda_s") for b in BASE if (OUT / "sintel" / f"{m}_{b}.json").exists()]
BINS = [(0, 0.02, "small (<2% disoccluded)"), (0.02, 0.06, "medium (2-6%)"), (0.06, 1.0, "large (>6%)")]

fr = {k: json.loads((OUT / "sintel" / f"{k}.json").read_text())["frames"] for k in KEYS}
occ = np.array([r["disocc_frac"] for r in fr["oracle_splat"]])
res = {}
for lo, hi, name in BINS:
    m = (occ >= lo) & (occ < hi)
    res[name] = {"frames": int(m.sum())}
    for k in KEYS:
        rs = [r for r, keep in zip(fr[k], m) if keep]
        res[name][k] = {s: float(np.nanmean([r.get(s, np.nan) for r in rs]))
                        for s in ("psnr", "psnr_disocc", "psnr_edges", "ssim", "lpips", "temporal_err")}
for a, b in [("oracle_tmpi", "oracle_splat"), ("vda_s_tmpi", "vda_s_splat")]:
    d = np.array([x["lpips"] - y["lpips"] for x, y in zip(fr[a], fr[b])])
    res[f"{a}_vs_{b}"] = {"lpips_diff_mean": float(d.mean()), "frac_frames_tmpi_lpips_better": float((d < 0).mean())}
(OUT / "sintel" / "breakdown.json").write_text(json.dumps(res, indent=1))
for name, v in res.items():
    if "frames" in v:
        print(f"\n{name}: {v['frames']} frames")
        for k in KEYS:
            x = v[k]
            print(f"  {k:28s} psnr {x['psnr']:.2f} disocc {x['psnr_disocc']:.2f} edges {x['psnr_edges']:.2f} "
                  f"ssim {x['ssim']:.3f} lpips {x['lpips']:.3f} temporal {x['temporal_err']:.2f}")
    else:
        print(name, v)
