"""iw3 (nagadomi/nunif, MIT) stereo methods as a baseline, driven with OUR depth.

iw3's own camera, with synthetic_view="right" (left = input), convergence 0 and mapper "none":
    right-view pixel disparity = depth * divergence * 0.01 * W        (depth in [0, 1], 1 = near)
so for a disparity map scaled to [0, 1] with max `px_max` pixels: divergence = 100 * px_max / W.
iw3's learned models are trained for divergence ~0-5 (row_flow_v3 auto-splits larger shifts into warp steps;
mlbw picks a d1/d2/d3 model by level). Sintel's ~13 is past that, so learned methods are off-range on large parallax.

Methods: forward_fill (depth-ordered forward splat + shift fill, no ML), forward_inpaint (+ light video inpaint),
row_flow_v3 (learned backward warp, default), mlbw_l2 / mlbw_l4 (learned 2-/4-layer backward warp),
mlbw_l2_inpaint (+ video inpaint).
Usage: IW3Stereo("row_flow_v3").render_seq(L uint8 (T,H,W,3), disp (T,H,W) in [0,1], px_max) -> (T,H,W,3) uint8
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

NUNIF = Path(__file__).resolve().parent / "nunif"
sys.path.insert(0, str(NUNIF))

from iw3 import utils as iw3u  # noqa: E402
from iw3.stereo_model_factory import create_stereo_model  # noqa: E402

INPAINT = {"forward_inpaint", "mlbw_l2_inpaint", "monobw_inpaint"}


class IW3Stereo:
    def __init__(self, method="row_flow_v3", device_id=0):
        self.method = method
        self.device = torch.device(f"cuda:{device_id}")
        self.device_id = device_id
        self._model_div = None
        self.side_model = None

    def _args(self, divergence):
        return SimpleNamespace(
            method=self.method, divergence=divergence, convergence=0.0, mapper="none", synthetic_view="right",
            preserve_screen_border=False, stereo_width=None, disable_amp=False,
            warp_steps=iw3u.calc_auto_warp_steps(self.method, divergence, "right"),
            mask_inner_dilation=0, mask_outer_dilation=0, inpaint_max_width=None,
            state={"convergence_model": None},
        )

    def _load(self, divergence):
        # mlbw picks its weights by divergence level; others are divergence-independent
        key = divergence if self.method.startswith("mlbw") else None
        if self.side_model is None or key != self._model_div:
            # create_stereo_model sees the per-eye divergence doubled for one-sided synthesis
            self.side_model = create_stereo_model(self.method, divergence * 2, self.device_id)
            if self.side_model is not None and hasattr(self.side_model, "eval"):
                self.side_model = self.side_model.eval()
            self._model_div = key

    @torch.inference_mode()
    def render_seq(self, L, disp, px_max):
        T, H, W, _ = L.shape
        divergence = 100.0 * px_max / W
        self._load(divergence)
        args = self._args(divergence)
        if self.method in INPAINT and hasattr(self.side_model, "reset"):
            self.side_model.reset()
        outs = []
        for t in range(T):
            im = torch.from_numpy(L[t]).to(self.device).permute(2, 0, 1).float() / 255
            d = torch.from_numpy(np.ascontiguousarray(disp[t])).to(self.device)[None].float()
            reset = [t == T - 1] if self.method in INPAINT else None
            _, right = iw3u.apply_divergence(d, im, args, self.side_model, reset_pts=reset)
            if right is not None:
                outs.append(right if right.ndim == 4 else right[None])
        R = torch.cat(outs, 0)
        assert R.shape[0] == T, (R.shape, T)
        return (R.clamp(0, 1).permute(0, 2, 3, 1) * 255 + 0.5).byte().cpu().numpy()


if __name__ == "__main__":
    # smoke test on one Sintel sequence, oracle depth
    import time
    from sintel_stereo import load_seq, psnr, right_regions
    meths = sys.argv[1].split(",") if len(sys.argv) > 1 else ["forward_fill", "row_flow_v3"]
    L, R, D, occ, oof = load_seq("alley_1")
    L, R, D, occ, oof = L[:12], R[:12], D[:12], occ[:12], oof[:12]
    mx = float(np.percentile(D, 99.9))
    for m in meths:
        t0 = time.perf_counter()
        P = IW3Stereo(m).render_seq(L, np.clip(D / mx, 0, 1), mx)
        dt = (time.perf_counter() - t0) / len(L) * 1e3
        rg = right_regions(D[10], occ[10], oof[10])
        print(f"{m:16s} f10 psnr {psnr(P[10], R[10], rg['valid']):.2f}  gaps {psnr(P[10], R[10], rg['disocc']):.2f}"
              f"  {dt:.0f} ms/frame (incl. load)", flush=True)
