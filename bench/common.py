"""Shared helpers for the TiltShift depth bench (RTX 5080 run).

Clips, flow (DIS + RAFT), warping and the scale-free flicker score.
"""
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "out"

CLIPS = {
    "rollercoaster": ROOT / "vda/assets/example_videos/davis_rollercoaster.mp4",
    "tokyo": ROOT / "vda/assets/example_videos/Tokyo-Walk_rgb.mp4",
    "bears": ROOT / "velodepth/assets/demo/bears.mp4",
}
MAX_FRAMES = 150
MAX_SIDE = 1280


def read_clip(name):
    """RGB uint8 (T,H,W,3), at most 150 frames, long side <= 1280. Returns frames, fps."""
    cap = cv2.VideoCapture(str(CLIPS[name]))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = []
    while len(frames) < MAX_FRAMES:
        ok, bgr = cap.read()
        if not ok:
            break
        h, w = bgr.shape[:2]
        s = MAX_SIDE / max(h, w)
        if s < 1:
            bgr = cv2.resize(bgr, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    cap.release()
    return np.stack(frames), fps


# ---------------------------------------------------------------- flow
_dis = None


def dis_flow(a_rgb, b_rgb):
    """Flow from a to b: a(x) ~ b(x + f(x)). (H,W,2) float32, (dx, dy)."""
    global _dis
    if _dis is None:
        _dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    ga = cv2.cvtColor(a_rgb, cv2.COLOR_RGB2GRAY)
    gb = cv2.cvtColor(b_rgb, cv2.COLOR_RGB2GRAY)
    return _dis.calc(ga, gb, None)


class Raft:
    def __init__(self, device="cuda"):
        from torchvision.models.optical_flow import Raft_Large_Weights, raft_large

        w = Raft_Large_Weights.DEFAULT
        self.model = raft_large(weights=w).to(device).eval()
        self.tf = w.transforms()
        self.device = device

    @torch.no_grad()
    def pair(self, a_rgb, b_rgb):
        """Returns (flow a->b, flow b->a), each (H,W,2) float32."""
        H, W = a_rgb.shape[:2]
        ph, pw = (-H) % 8, (-W) % 8
        a = torch.from_numpy(a_rgb).permute(2, 0, 1)[None].float() / 255
        b = torch.from_numpy(b_rgb).permute(2, 0, 1)[None].float() / 255
        x1 = torch.cat([a, b]).to(self.device)
        x2 = torch.cat([b, a]).to(self.device)
        x1, x2 = self.tf(x1, x2)
        if ph or pw:
            x1 = F.pad(x1, (0, pw, 0, ph), mode="replicate")
            x2 = F.pad(x2, (0, pw, 0, ph), mode="replicate")
        f = self.model(x1, x2)[-1][:, :, :H, :W]
        f = f.permute(0, 2, 3, 1).float().cpu().numpy()
        return f[0], f[1]


# ---------------------------------------------------------------- warp + score
def warp(img, flow):
    """Sample img at x + flow(x) (bilinear). img (H,W) or (H,W,C)."""
    H, W = flow.shape[:2]
    gx, gy = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    return cv2.remap(img, gx + flow[..., 0], gy + flow[..., 1], cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def consistent(fwd, bwd, thresh=1.0):
    """fwd = t->t-1, bwd = t-1->t. True where |fwd + bwd(x+fwd)| < thresh, in bounds."""
    H, W = fwd.shape[:2]
    gx, gy = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    mx, my = gx + fwd[..., 0], gy + fwd[..., 1]
    inb = (mx >= 0) & (mx <= W - 1) & (my >= 0) & (my <= H - 1)
    wb = cv2.remap(bwd, mx, my, cv2.INTER_LINEAR)
    return inb & (np.linalg.norm(fwd + wb, axis=-1) < thresh)


def frame_flicker(d_t, d_prev, fwd, valid):
    """Mean |d_t - w| / ((d_t + w)/2) over valid pixels, w = d_prev warped into t."""
    w = warp(d_prev, fwd)
    m = valid & (d_t > 0.01 * np.median(d_t)) & (w > 0.01 * np.median(d_prev))
    if m.sum() < 100:
        return np.nan
    return float(np.mean(np.abs(d_t[m] - w[m]) / ((d_t[m] + w[m]) / 2)))


def load_depth(clip, model):
    return np.load(OUT / clip / f"{model}.npy").astype(np.float32)


def save_depth(clip, model, d):
    (OUT / clip).mkdir(parents=True, exist_ok=True)
    # float32: fp16 rounding (~5e-4 relative) is the size of the smallest flicker scores
    np.save(OUT / clip / f"{model}.npy", d.astype(np.float32))
