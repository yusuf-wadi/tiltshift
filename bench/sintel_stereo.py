"""Ground-truth stereo benchmark on MPI-Sintel stereo (training split, final pass).

Task: given only the LEFT frame, synthesise the RIGHT view (true camera 10 cm to the right) and compare it with the
real right frame. Two depth modes:
- oracle: the true disparity (isolates the renderer and its gap filling)
- vda_s:  VDA-S disparity on the left video, aligned to the true disparity by one scale + shift per sequence
          (least squares over all frames), i.e. the full pipeline with a fair scale
Methods: identity (right = left, floor), warp (naive backward warp), splat (z-buffer forward splat + Telea inpaint),
tmpi (Meta TMPI). All share the camera model in stereo_views.py, with b chosen so pixel disparity matches.

Regions, in right-view coordinates, from the true disparity:
- border: no-hit run touching the right image edge (content outside the left frame), excluded everywhere
- disocc: right-view pixels no visible left pixel lands on (must be hallucinated)
- edges:  true right-view disparity gradient > 1 px, dilated 2 px, outside disocc
Metrics per frame: PSNR (valid / disocc / edges), SSIM, LPIPS-alex (border pixels copied from GT so they don't
count), temporal error = mean |(P_t - P_t-1) - (G_t - G_t-1)| on valid pixels (0-255).
Writes out/sintel/<mode>_<method>.json (per-frame + means) and out/sintel/summary.json.
Usage: python sintel_stereo.py [--seqs alley_1,...] [--modes oracle,vda_s] [--methods ...]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from skimage.metrics import structural_similarity

from common import OUT, ROOT
from stereo_views import TMPIStereo, splat_stereo, warp_stereo

SINTEL = Path.home() / "datasets/sintel-stereo/training"
F_RATIO = 0.58
OUTD = OUT / "sintel"


def read_disp(p):
    a = np.array(Image.open(p)).astype(np.float64)
    return (a[..., 0] * 4 + a[..., 1] / 2 ** 6 + a[..., 2] / 2 ** 14).astype(np.float32)


def load_seq(seq):
    names = sorted(p.name for p in (SINTEL / "final_left" / seq).glob("*.png"))
    rd = lambda sub, n: np.array(Image.open(SINTEL / sub / seq / n))
    L = np.stack([rd("final_left", n)[..., :3] for n in names])
    R = np.stack([rd("final_right", n)[..., :3] for n in names])
    D = np.stack([read_disp(SINTEL / "disparities" / seq / n) for n in names])
    occ = np.stack([rd("occlusions", n) > 0 for n in names])
    oof = np.stack([rd("outofframe", n) > 0 for n in names])
    return L, R, D, occ, oof


def right_regions(D, occ, oof):
    """Splat visible left pixels into the right view with the true disparity -> hit mask and right disparity."""
    H, W = D.shape
    ys, xs = np.nonzero(~occ & ~oof)
    d = D[ys, xs]
    tx = np.floor(xs - d).astype(int)
    ys2, tx2, d2 = np.concatenate([ys, ys]), np.concatenate([tx, tx + 1]), np.concatenate([d, d])
    o = np.argsort(d2, kind="stable")
    ys2, tx2, d2 = ys2[o], tx2[o], d2[o]
    ok = (tx2 >= 0) & (tx2 < W)
    Dr = np.zeros((H, W), np.float32)
    hit = np.zeros((H, W), bool)
    Dr[ys2[ok], tx2[ok]] = d2[ok]
    hit[ys2[ok], tx2[ok]] = True
    # border: no-hit runs that touch the right image edge
    border = np.zeros((H, W), bool)
    run = np.cumprod((~hit)[:, ::-1], axis=1)[:, ::-1].astype(bool)
    border |= run
    disocc = ~hit & ~border
    # fill Dr holes (nearest along row) only to get edge gradients
    scale = max(float(Dr.max()), 1e-6)
    Drf = cv2.inpaint((Dr / scale * 255).astype(np.uint8), (~hit).astype(np.uint8), 2,
                      cv2.INPAINT_NS).astype(np.float32) * scale / 255
    gx = np.abs(np.diff(Drf, axis=1, append=Drf[:, -1:]))
    gy = np.abs(np.diff(Drf, axis=0, append=Drf[-1:, :]))
    edges = cv2.dilate(((gx > 1) | (gy > 1)).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    edges &= ~disocc & ~border
    return {"valid": ~border, "disocc": disocc, "edges": edges}


def psnr(a, b, m):
    if m.sum() < 50:
        return np.nan
    mse = np.mean((a[m].astype(np.float32) - b[m].astype(np.float32)) ** 2)
    return float(10 * np.log10(255 ** 2 / max(mse, 1e-10)))


def vda_disp(L):
    sys.path.insert(0, str(ROOT / "vda"))
    from run_models import load_vda
    global _VDA
    if "_VDA" not in globals():
        _VDA = load_vda("vits")
    return _VDA(L, 24.0)


def align(pred, D, valid):
    x, y = pred[valid].astype(np.float64), D[valid].astype(np.float64)
    A = np.stack([x, np.ones_like(x)], 1)
    (a, c), *_ = np.linalg.lstsq(A, y, rcond=None)
    return np.clip(a * pred + c, 0, None).astype(np.float32), float(a), float(c)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seqs", default="")
    ap.add_argument("--modes", default="oracle,vda_s")
    ap.add_argument("--methods", default="identity,warp,splat,tmpi")
    args = ap.parse_args()
    seqs = args.seqs.split(",") if args.seqs else sorted(p.name for p in (SINTEL / "final_left").iterdir())
    modes, methods = args.modes.split(","), args.methods.split(",")
    OUTD.mkdir(parents=True, exist_ok=True)

    import lpips
    lp = lpips.LPIPS(net="alex", verbose=False).cuda().eval()
    tm = TMPIStereo() if "tmpi" in methods else None
    if any(m.startswith("iw3_") for m in methods):
        from iw3_stereo import IW3Stereo
        iw3 = {m: IW3Stereo(m[4:]) for m in methods if m.startswith("iw3_")}
    to_t = lambda x: torch.from_numpy(x).permute(2, 0, 1)[None].float().cuda() / 127.5 - 1

    recs = {f"{m}_{k}": [] for m in modes for k in methods}
    for seq in seqs:
        L, R, D, occ, oof = load_seq(seq)
        T, H, W = D.shape
        regions = [right_regions(D[t], occ[t], oof[t]) for t in range(T)]
        dmax = float(np.percentile(D, 99.9))
        depth = {"oracle": (D, dmax)}
        if "vda_s" in modes:
            Dp, a, c = align(vda_disp(L), D, ~oof)
            depth["vda_s"] = (Dp, float(np.percentile(Dp, 99.9)))
        for mode in modes:
            Dm, mx = depth[mode]
            b = mx / (F_RATIO * W)
            disp = np.clip(Dm / mx, 0, 1)
            for meth in methods:
                prev_p = None
                if meth.startswith("iw3_"):  # whole sequence at once (iw3 inpaint buffers frames)
                    t0 = time.perf_counter()
                    P_seq = iw3[meth].render_seq(L, disp, mx)
                    iw3_ms = (time.perf_counter() - t0) * 1e3 / T
                for t in range(T):
                    t0 = time.perf_counter()
                    if meth.startswith("iw3_"):
                        P = P_seq[t]
                    elif meth == "identity":
                        P = L[t]
                    elif meth == "warp":
                        P = warp_stereo(L[t], disp[t], b)
                    elif meth == "splat":
                        P, _ = splat_stereo(L[t], disp[t], b)
                    else:
                        P = tm(L[t], disp[t], baselines=[b])[0]
                    ms = iw3_ms if meth.startswith("iw3_") else (time.perf_counter() - t0) * 1e3
                    rg, G = regions[t], R[t]
                    Pm = P.copy()
                    Pm[~rg["valid"]] = G[~rg["valid"]]
                    ssim = structural_similarity(Pm, G, channel_axis=2, data_range=255)
                    with torch.no_grad():
                        lpv = float(lp(to_t(Pm), to_t(G)).item())
                    rec = {"seq": seq, "t": t, "ms": ms,
                           "psnr": psnr(P, G, rg["valid"]), "psnr_disocc": psnr(P, G, rg["disocc"]),
                           "psnr_edges": psnr(P, G, rg["edges"]), "ssim": float(ssim), "lpips": lpv,
                           "disocc_frac": float(rg["disocc"].mean())}
                    if prev_p is not None:
                        dP = P.astype(np.float32) - prev_p.astype(np.float32)
                        dG = G.astype(np.float32) - R[t - 1].astype(np.float32)
                        v = rg["valid"] & regions[t - 1]["valid"]
                        rec["temporal_err"] = float(np.abs(dP - dG).mean(-1)[v].mean())
                    prev_p = P
                    recs[f"{mode}_{meth}"].append(rec)
            print(seq, mode, flush=True)

    sp = OUTD / "summary.json"
    summary = json.loads(sp.read_text()) if sp.exists() and not args.seqs else {}  # merge with earlier runs
    for key, rs in recs.items():
        keys = ["psnr", "psnr_disocc", "psnr_edges", "ssim", "lpips", "temporal_err", "ms"]
        summary[key] = {k: float(np.nanmean([r[k] for r in rs if k in r])) for k in keys}
        summary[key]["frames"] = len(rs)
        (OUTD / f"{key}.json").write_text(json.dumps({"summary": summary[key], "frames": rs}))
    if not args.seqs:
        sp.write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
