"""Time VDA-Small, VDA-Large and VeloDepth on the three clips (CUDA, RTX 5080).

Per model: warm up once on 40 frames, then time the full clip with cuda.synchronize().
Writes out/<clip>/<model>.npy (T,H,W float32) and out/<clip>/<model>.json (fps, VRAM, size).
Usage: python run_models.py [model ...]   (default: vda_s vda_l velodepth)
"""
import json
import sys
import time

import cv2
import numpy as np
import torch

from common import CLIPS, OUT, ROOT, read_clip, save_depth

sys.path.insert(0, str(ROOT / "vda"))

DEV = "cuda"
VDA_CFG = {
    "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
}


def load_vda(enc):
    from video_depth_anything.video_depth import VideoDepthAnything

    m = VideoDepthAnything(**VDA_CFG[enc])
    m.load_state_dict(torch.load(ROOT / f"vda/checkpoints/video_depth_anything_{enc}.pth",
                                 map_location="cpu"), strict=True)
    m = m.to(DEV).eval()

    def run(frames, fps):
        d, _ = m.infer_video_depth(frames, fps, input_size=518, device=DEV)  # fp16 default
        return np.asarray(d)

    return run


def load_velodepth():
    from velodepth.models import VeloDepth

    m = VeloDepth.from_pretrained("lpiccinelli/velodepth")
    m.to(DEV)  # returns None, keep on its own line
    m.eval()
    m.resolution_level = 9
    m.interpolation_mode = "bilinear"

    @torch.no_grad()
    def run(frames, fps):
        x = torch.from_numpy(frames).to(DEV).permute(0, 3, 1, 2)
        out = m.infer(rgbs=x, normalize=True)
        d = out["depth"]
        while d.dim() > 3:
            d = d.squeeze(0) if d.shape[0] == 1 else d.squeeze(1)
        return d.float().cpu().numpy()

    return run


LOADERS = {"vda_s": lambda: load_vda("vits"), "vda_l": lambda: load_vda("vitl"),
           "velodepth": load_velodepth}


def main(models):
    clips = {c: read_clip(c) for c in CLIPS}
    for name in models:
        run = LOADERS[name]()
        for clip, (frames, fps) in clips.items():
            T, H, W = frames.shape[:3]
            run(frames[:40], fps)  # warm-up
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            d = run(frames, fps)
            torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            out_size = list(d.shape)
            if d.shape[1:] != (H, W):
                d = np.stack([cv2.resize(x, (W, H), interpolation=cv2.INTER_LINEAR) for x in d])
            save_depth(clip, name, d)
            rec = {"model": name, "clip": clip, "frames": T, "frame_size": [H, W],
                   "out_size": out_size, "seconds": dt, "fps": T / dt,
                   "peak_vram_gb": torch.cuda.max_memory_allocated() / 1e9}
            (OUT / clip / f"{name}.json").write_text(json.dumps(rec, indent=1))
            print(json.dumps(rec), flush=True)
        del run
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main(sys.argv[1:] or ["vda_s", "vda_l", "velodepth"])
