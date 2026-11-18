"""Stereo view synthesis backends for TiltShift research: RGB + disparity -> left/right views.

Backends (same camera model, so outputs compare):
- TMPIStereo: Meta's Tiled Multiplane Images (ICCV 2023, CC-BY-NC, research only), with any disparity source
  (we feed VDA-S instead of their DPT). Uses their PyTorch renderer on the GPU (no OpenGL needed).
- warp_stereo: naive backward warp by disparity with border stretch (what the old sbsconvert.py did). Cheap
  baseline for gap filling.

Camera model (TMPI's): K = [[0.58 W, 0, W/2], [0, 0.58 H, H/2]], depth = 1 / disp with disp in [0, 1]
(disp 1 = nearest). An eye shifted by b along x sees pixel disparity 0.58 * W * b * disp.
Default b = 0.04 -> max parallax ~2.3% of image width, a comfortable stereo budget.

Usage:
    s = TMPIStereo()
    out = s(rgb_hw3_uint8, disp_hw_0to1, baselines=[0.0, 0.04])   # list of (H,W,3) uint8 views
    s.last_planes  # per-tile plane disparities of the last frame (n_tiles, 4), for temporal analysis
"""
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

TMPI_DIR = Path(__file__).resolve().parent / "tmpi"


def normalize_disp(disp, lo=None, hi=None):
    """Map disparity to TMPI's [0, 1] range. Pass clip-wide lo/hi for video (per-frame min/max flickers)."""
    lo = np.percentile(disp, 0.5) if lo is None else lo
    hi = np.percentile(disp, 99.5) if hi is None else hi
    return np.clip((disp - lo) / max(hi - lo, 1e-9), 0, 1).astype(np.float32)


def clip_disp_range(disps):
    """Clip-wide (lo, hi) for normalize_disp, robust to outliers."""
    return float(np.percentile(disps, 0.5)), float(np.percentile(disps, 99.5))


def warp_stereo(rgb, disp, baseline=0.04, f_ratio=0.58):
    """Naive backward warp: view(x) = rgb(x + 0.58 W b disp(x)). Holes become stretched texture."""
    H, W = disp.shape
    shift = f_ratio * W * baseline * disp
    gx, gy = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    import cv2
    return cv2.remap(rgb, gx + shift.astype(np.float32), gy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def splat_stereo(rgb, disp, baseline=0.04, f_ratio=0.58, inpaint_radius=3):
    """Classical DIBR: forward-splat each pixel to x - 0.58 W b disp with a z-buffer (nearer wins), then fill the
    disoccluded holes with OpenCV Telea inpainting. Returns (view, hole_mask)."""
    import cv2
    H, W = disp.shape
    shift = f_ratio * W * baseline * disp
    ys, xs = np.mgrid[0:H, 0:W]
    ys, xs, d = ys.ravel(), xs.ravel(), disp.ravel()
    tx = np.floor(xs - shift.ravel()).astype(int)
    # two-pixel splat closes rounding cracks; both copies share the source's depth key
    ys2, xs2 = np.concatenate([ys, ys]), np.concatenate([xs, xs])
    tx2, d2 = np.concatenate([tx, tx + 1]), np.concatenate([d, d])
    order = np.argsort(d2, kind="stable")  # far first, near last -> near overwrites
    ys2, xs2, tx2 = ys2[order], xs2[order], tx2[order]
    ok = (tx2 >= 0) & (tx2 < W)
    out = np.zeros_like(rgb)
    hit = np.zeros((H, W), bool)
    out[ys2[ok], tx2[ok]] = rgb[ys2[ok], xs2[ok]]
    hit[ys2[ok], tx2[ok]] = True
    holes = (~hit).astype(np.uint8)
    view = cv2.inpaint(out, holes, inpaint_radius, cv2.INPAINT_TELEA) if holes.any() else out
    return view, ~hit


class TMPIStereo:
    def __init__(self, device="cuda", max_side=1024):
        sys.path.insert(0, str(TMPI_DIR))
        import config  # noqa: E402  (TMPI's config)
        from tmpi import TMPI
        from tmpi_renderer import TMPIRenderer

        self.config, self.Renderer, self.device, self.max_side = config, TMPIRenderer, device, max_side
        self.model = TMPI(num_planes=config.num_planes)
        sd = torch.load(TMPI_DIR / "weights/mpti_04.pth", map_location="cpu")
        sd = {k.removeprefix("module."): v for k, v in sd.items()}
        self.model.load_state_dict(sd)
        self.model = self.model.to(device).eval()
        self.last_planes = None
        self.timing = {}

    def _tiles(self, disp, rgb, K, tile_sz, pad_sz):
        """Overlapping tiles, as in TMPI's dataset.py."""
        _, _, h, w = disp.shape
        sy = torch.arange(0, h, tile_sz - pad_sz)
        sx = torch.arange(0, w, tile_sz - pad_sz)
        disp = F.pad(disp, (0, tile_sz, 0, tile_sz), "replicate")
        rgb = F.pad(rgb, (0, tile_sz, 0, tile_sz), "replicate")
        Ks, dt, rt, xs, ys = [], [], [], [], []
        for y in sy:
            for x in sx:
                Ki = K.clone()
                Ki[0, 2] -= x
                Ki[1, 2] -= y
                Ks.append(Ki)
                dt.append(disp[:, :, y:y + tile_sz, x:x + tile_sz])
                rt.append(rgb[:, :, y:y + tile_sz, x:x + tile_sz])
                xs.append(int(x))
                ys.append(int(y))
        return (torch.stack(dt, 1), torch.stack(rt, 1), torch.stack(Ks, 0)[None],
                torch.tensor(xs)[None], torch.tensor(ys)[None])

    @torch.no_grad()
    def __call__(self, rgb, disp, baselines=(0.0, 0.04)):
        """rgb (H,W,3) uint8, disp (H,W) in [0,1]. Returns views at x-offsets `baselines`, resized back to (H,W)."""
        cfg, dev = self.config, self.device
        H0, W0 = disp.shape
        s = min(1.0, self.max_side / max(H0, W0))
        h, w = int(round(H0 * s)), int(round(W0 * s))
        r = torch.from_numpy(rgb).permute(2, 0, 1)[None].float().div(255)
        d = torch.from_numpy(disp)[None, None].float()
        if s < 1:
            r = F.interpolate(r, (h, w), mode="bilinear", align_corners=False)
            d = F.interpolate(d, (h, w), mode="bilinear", align_corners=False)
        K = torch.tensor([[0.58 * w, 0, 0.5 * w], [0, 0.58 * h, 0.5 * h], [0, 0, 1.0]])
        tile_sz = int(np.clip(2 ** int(np.ceil(np.log2(max(cfg.tilesz2w_ratio * w - 1, 1)))),
                              cfg.tilesz_min, cfg.tilesz_max))
        pad_sz = int(tile_sz * cfg.padsz2tile_ratio)
        dt, rt, Kt, sx, sy = self._tiles(d, r, K, tile_sz, pad_sz)

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        mpis, planes = self.model(rt.to(dev), dt.to(dev), r.to(dev), d.to(dev))
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        self.last_planes = planes[0].float().cpu().numpy()  # (n_tiles, num_planes) disparities

        renderer = self.Renderer(h, w, dev)
        views = []
        for b in baselines:
            G = torch.eye(4, device=dev)
            G[0, 3] = -b  # target-from-source: eye moved +b along x
            v = renderer.render(mpis, planes, G, Kt.to(dev).clone(), sx.to(dev), sy.to(dev))[0]
            v = v.clamp(0, 1)
            if s < 1:
                v = F.interpolate(v[None], (H0, W0), mode="bilinear", align_corners=False)[0]
            views.append((v.permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8))
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        self.timing = {"predict_ms": (t1 - t0) * 1e3, "render_ms_per_view": (t2 - t1) * 1e3 / len(baselines),
                       "tile_sz": tile_sz, "n_tiles": int(mpis.shape[1]), "work_size": [h, w]}
        return views
