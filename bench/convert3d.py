"""TiltShift convert3d: any-length 2D video -> stereo 3D video for a headset.

Pipeline (all on the GPU, bounded memory, so a full movie works):
  decode (PyAV) -> scene-cut detection -> streaming Video Depth Anything Small (Apache-2.0), same 32-frame windows,
  keyframe context and scale/shift alignment as VDA's own infer_video_depth, reset per shot -> per-shot slow depth-range
  normalisation (no per-frame min/max flicker) -> iw3 forward_fill renderer (nagadomi/nunif, MIT: depth-ordered
  bilinear forward splat + hole fill) for both eyes -> SBS frames -> NVENC HEVC -> remux the source's audio/subtitles.

Usage:
  python convert3d.py in.mkv out.mkv [--layout full|half] [--divergence 2.0] [--convergence 0.5]
                      [--method forward_fill|forward_inpaint] [--start SEC] [--duration SEC] [--max-width 1920]
Output: full SBS (2W x H) by default; --layout half keeps the source size (each eye squeezed to W/2).
"""
import argparse
import heapq
import queue
import threading
import sys
import time
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import av
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "vda"))
sys.path.insert(0, str(ROOT / "nunif"))

from video_depth_anything.video_depth import (INFER_LEN, INTERP_LEN, KEYFRAMES, OVERLAP,  # noqa: E402
                                              VideoDepthAnything)
from iw3 import utils as iw3u  # noqa: E402
from iw3.stereo_model_factory import create_stereo_model  # noqa: E402

DEV = torch.device("cuda")
STEP = INFER_LEN - OVERLAP          # 22 new frames per window
ALIGN_LEN = OVERLAP - INTERP_LEN    # 2 keyframes used for scale/shift alignment
MEAN = torch.tensor([0.485, 0.456, 0.406], device=DEV).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225], device=DEV).view(3, 1, 1)


# ---------------------------------------------------------------- depth: streaming VDA
def load_vda_small():
    m = VideoDepthAnything(encoder="vits", features=64, out_channels=[48, 96, 192, 384])
    m.load_state_dict(torch.load(ROOT / "vda/checkpoints/video_depth_anything_vits.pth", map_location="cpu"))
    return m.to(DEV).eval()


def scale_shift(pred, target):
    """Least-squares a, b with a*pred + b ~= target (VDA's compute_scale_and_shift, in torch)."""
    p, t = pred.flatten().double(), target.flatten().double()
    a00, a01, a11 = (p * p).sum(), p.sum(), torch.tensor(float(p.numel()), dtype=torch.float64, device=p.device)
    b0, b1 = (p * t).sum(), t.sum()
    det = a00 * a11 - a01 * a01
    if det.abs() < 1e-12:
        return 1.0, 0.0
    return float((a11 * b0 - a01 * b1) / det), float((-a01 * b0 + a00 * b1) / det)


class VDAStream:
    """VDA's windowed video inference as a stream. push(frame) -> list of finished depth maps (model resolution).
    Exactly mirrors infer_video_depth: window k>0 reuses the previous window's KEYFRAMES inputs as its first OVERLAP
    slots, is aligned to the reference keyframes by scale/shift, and cross-fades INTERP_LEN frames."""

    def __init__(self, model, frame_hw, input_size=518):
        self.m = model
        H, W = frame_hw
        ratio = max(H, W) / min(H, W)
        if ratio > 1.78:  # same rule as VDA: smaller input for very wide frames (memory)
            input_size = round(int(input_size * 1.777 / ratio) / 14) * 14
        s = input_size / min(H, W)  # lower-bound resize, multiple of 14
        self.size = (max(14, round(H * s / 14) * 14), max(14, round(W * s / 14) * 14))
        self.reset()

    def reset(self):
        self.buf = []            # transformed inputs waiting for the next window
        self.pre_input = None
        self.ref_align = None
        self.pending = []        # last INTERP_LEN aligned depths (may still be cross-faded)
        self.first = True

    def _prep(self, rgb_t):  # rgb_t: 3xHxW float 0..1 on GPU
        x = F.interpolate(rgb_t[None], size=self.size, mode="bicubic", align_corners=False, antialias=True)[0]
        return ((x.clamp(0, 1) - MEAN) / STD)

    def push(self, rgb_t):
        self.buf.append(self._prep(rgb_t))
        need = INFER_LEN if self.first else STEP
        return self._run() if len(self.buf) == need else []

    def flush(self):
        """End of shot: pad with the last frame like VDA does, run, return everything left (trimmed to real length)."""
        out = []
        if self.buf:
            real = len(self.buf)
            need = INFER_LEN if self.first else STEP
            self.buf += [self.buf[-1]] * (need - real)
            out = self._run(real_new=real)
        out += self.pending
        self.pending = []
        self.reset()
        return out

    @torch.no_grad()
    def _run(self, real_new=None):
        if self.first:
            cur = torch.stack(self.buf)[None]
        else:
            cur = torch.cat([self.pre_input[:, KEYFRAMES], torch.stack(self.buf)[None]], 1)
        self.buf = []
        with torch.autocast(device_type="cuda"):
            d = self.m.forward(cur)[0].float()  # T x h x w
        d = d.clamp_min(0)
        self.pre_input = cur
        if self.first:
            self.first = False
            self.ref_align = [d[k] for k in KEYFRAMES[:ALIGN_LEN]]
            aligned = list(d)
            n_new = INFER_LEN if real_new is None else real_new
            aligned = aligned[:n_new]
        else:
            a, b = scale_shift(torch.stack([d[i] for i in range(ALIGN_LEN)]), torch.stack(self.ref_align))
            d = (d * a + b).clamp_min(0)
            post = [d[i] for i in range(ALIGN_LEN, OVERLAP)]  # re-estimates of the last INTERP_LEN frames
            w = torch.linspace(0, 1, INTERP_LEN).tolist()
            self.pending = [p * (1 - wi) + q * wi for p, q, wi in zip(self.pending, post, w)]
            n_new = STEP if real_new is None else real_new
            aligned = self.pending + [d[i] for i in range(OVERLAP, OVERLAP + n_new)]
            self.pending = []
            self.ref_align = self.ref_align[:1] + [d[k] for k in KEYFRAMES[1:ALIGN_LEN]]
        # keep the last INTERP_LEN back: the next window will cross-fade them (unless the shot ends)
        if real_new is None:
            self.pending = aligned[-INTERP_LEN:]
            return aligned[:-INTERP_LEN]
        return aligned


# ---------------------------------------------------------------- shot handling
def cut_signature(rgb_t):
    """48x27 colour thumbnail, zero-mean / unit-std, plus its raw std (to skip near-black frames)."""
    g = F.interpolate(rgb_t[None], size=(27, 48), mode="area")[0].flatten()
    sd = float(g.std())
    return (g - g.mean()) / (sd + 1e-3), sd


class CutDetector:
    """Hard cut = thumbnail correlation drops: 1 - NCC > threshold. Contrast-independent, so dark low-contrast
    cuts are caught (Dune: within-shot 99th pct 0.10, cuts 0.3-1.0). No cut on near-black frames (fades) or
    within `min_gap` frames of the previous cut (flashes, explosions)."""

    def __init__(self, threshold=0.4, min_gap=6, min_std=0.02):
        self.thr, self.min_gap, self.min_std = threshold, min_gap, min_std
        self.prev, self.since = None, 0

    def __call__(self, rgb_t):
        g, sd = cut_signature(rgb_t)
        cut = False
        if self.prev is not None:
            pg, psd = self.prev
            score = 1 - float((g * pg).mean())
            cut = score > self.thr and min(sd, psd) > self.min_std and self.since >= self.min_gap
        self.prev = (g, sd)
        self.since = 0 if cut else self.since + 1
        return cut


class Normaliser:
    """Per-shot disparity range: initialised from the shot's first frames, then a slow EMA of the 2nd/98th
    percentiles, so depth doesn't breathe frame to frame. Reset at cuts."""

    def __init__(self, decay=0.98):
        self.decay, self.lo, self.hi = decay, None, None

    def reset(self):
        self.lo = self.hi = None

    def __call__(self, d):
        s = d[::4, ::4].flatten()
        q = torch.quantile(s, torch.tensor([0.02, 0.98], device=d.device))
        lo, hi = float(q[0]), float(q[1])
        if self.lo is None:
            self.lo, self.hi = lo, hi
        else:
            self.lo = self.decay * self.lo + (1 - self.decay) * lo
            self.hi = self.decay * self.hi + (1 - self.decay) * hi
        return ((d - self.lo) / max(self.hi - self.lo, 1e-6)).clamp(0, 1)


def detect_crop(path, H, W, samples=40):
    """Letterbox bars: sample frames across the file, keep the rows that ever carry picture. Returns (y0, y1) in the
    W x H working frame (even numbers). Movies that switch aspect (IMAX scenes) keep the widest picture."""
    c = av.open(path)
    v = c.streams.video[0]
    dur = float(c.duration / 1e6) if c.duration else None
    top, bot = H, 0
    if not dur:
        return 0, H
    for k in range(1, samples + 1):
        try:
            c.seek(int(dur * k / (samples + 1) * 1e6))
            fr = next(c.decode(v)).to_ndarray(format="gray", width=W, height=H).astype(np.float32)
        except Exception:
            continue
        rows = np.percentile(fr, 98, axis=1) > 28  # picture rows (limited-range black is 16)
        if rows.sum() < H * 0.2:  # fade to black: no information
            continue
        idx = np.nonzero(rows)[0]
        top, bot = min(top, int(idx[0])), max(bot, int(idx[-1]) + 1)
    c.close()
    if bot <= top:
        return 0, H
    y0, y1 = top // 2 * 2, min(H, (bot + 1) // 2 * 2)
    return (y0, y1) if (y0 >= 8 or H - y1 >= 8) else (0, H)


def rgb_to_yuv420(x):
    """3xHxW float RGB 0..1 (GPU) -> (H*3/2) x W uint8 yuv420p, BT.709 limited range."""
    x = x.clamp(0, 1)
    r, g, b = x[0], x[1], x[2]
    y = 0.2126 * r + 0.7152 * g + 0.0722 * b
    cb = (b - y) / 1.8556
    cr = (r - y) / 1.5748
    H, W = y.shape
    Y = (16 + 219 * y + 0.5).clamp(0, 255).byte()
    sub = lambda c: F.avg_pool2d(c[None, None], 2)[0, 0]
    U = (128 + 224 * sub(cb) + 0.5).clamp(0, 255).byte()
    V = (128 + 224 * sub(cr) + 0.5).clamp(0, 255).byte()
    return torch.cat([Y.flatten(), U.flatten(), V.flatten()]).view(H * 3 // 2, W)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("output")
    ap.add_argument("--layout", choices=["full", "half"], default="full")
    ap.add_argument("--divergence", type=float, default=2.0, help="iw3 units: ~%% of width per eye x2 total")
    ap.add_argument("--convergence", type=float, default=0.5, help="0 = far plane at screen, 1 = nearest at screen")
    ap.add_argument("--method", default="forward_fill", choices=["forward_fill", "row_flow_v3"])
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--duration", type=float, default=None)
    ap.add_argument("--max-width", type=int, default=1920, help="downscale the source to this width per eye")
    ap.add_argument("--cq", type=int, default=21, help="NVENC constant quality (lower = better/bigger)")
    ap.add_argument("--cut-threshold", type=float, default=0.4, help="1 - thumbnail correlation for a hard cut")
    ap.add_argument("--no-crop", action="store_true", help="don't detect/crop letterbox bars")
    args = ap.parse_args()

    out_path = Path(args.output)
    tmp_video = out_path.with_name(out_path.stem + ".video-only.mkv")

    inp = av.open(args.input)
    vs = inp.streams.video[0]
    vs.thread_type = "AUTO"
    if args.start:
        inp.seek(int(args.start / vs.time_base), stream=vs)
    W0, H0 = vs.codec_context.width, vs.codec_context.height
    s = min(1.0, args.max_width / W0)
    W, H = (round(W0 * s / 2) * 2, round(H0 * s / 2) * 2)
    rate = vs.average_rate or vs.guessed_rate
    total = None
    if args.duration:
        total = int(args.duration * float(rate))
    elif vs.frames:
        total = vs.frames
    elif vs.duration:
        total = int(float(vs.duration * vs.time_base) * float(rate))

    y0, y1 = detect_crop(args.input, H, W) if not args.no_crop else (0, H)
    Hc = y1 - y0
    if (y0, y1) != (0, H):
        print(f"letterbox: picture rows {y0}-{y1} of {H} (bars restored in the output)", flush=True)
    vda = VDAStream(load_vda_small(), (Hc, W))
    norm = Normaliser()
    side = create_stereo_model(args.method, args.divergence, 0)
    iw3_args = SimpleNamespace(
        method=args.method, divergence=args.divergence, convergence=args.convergence, mapper="none",
        synthetic_view="both", preserve_screen_border=False, stereo_width=None, disable_amp=False,
        warp_steps=iw3u.calc_auto_warp_steps(args.method, args.divergence, "both"),
        mask_inner_dilation=0, mask_outer_dilation=0, inpaint_max_width=None, state={"convergence_model": None})

    OW = 2 * W if args.layout == "full" else W
    stats = {"frames": 0, "shots": 1, "t0": time.perf_counter(), "t_render": 0.0}
    rgb_q = deque()  # (pts, rgb tensor) waiting for depth

    # ---- encode thread: NVENC HEVC from GPU-converted yuv420p
    enc_q = queue.Queue(maxsize=48)

    def encoder():
        out = av.open(str(tmp_video), "w")
        ost = out.add_stream("hevc_nvenc", rate=rate)
        ost.width, ost.height, ost.pix_fmt = OW, H, "yuv420p"
        ost.time_base = vs.time_base
        ost.options = {"preset": "p5", "rc": "vbr", "cq": str(args.cq), "b": "0",
                       "colorspace": "bt709", "color_primaries": "bt709", "color_trc": "bt709"}
        while (item := enc_q.get()) is not None:
            pts, yuv = item
            vf = av.VideoFrame.from_ndarray(yuv, format="yuv420p")
            vf.pts, vf.time_base = pts, vs.time_base
            for pkt in ost.encode(vf):
                out.mux(pkt)
        for pkt in ost.encode():
            out.mux(pkt)
        out.close()

    # ---- decode thread: frames scaled to W x H rgb24
    dec_q = queue.Queue(maxsize=48)

    def decoder():
        n_in, first_pts = 0, None
        for frame in inp.decode(vs):
            if frame.pts is None or (args.start and float(frame.pts * vs.time_base) < args.start):
                continue
            if args.duration and n_in >= total:
                break
            if first_pts is None:
                first_pts = frame.pts
            arr = frame.to_ndarray(format="rgb24", width=W, height=H)
            dec_q.put((frame.pts - first_pts, np.ascontiguousarray(arr[y0:y1])))
            n_in += 1
        dec_q.put(None)

    threads = [threading.Thread(target=encoder, daemon=True), threading.Thread(target=decoder, daemon=True)]
    for t in threads:
        t.start()

    def render(depths):
        if not depths:
            return
        t0 = time.perf_counter()
        for dep in depths:
            pts, im = rgb_q.popleft()
            d = F.interpolate(norm(dep)[None, None], size=(Hc, W), mode="bilinear", align_corners=False)
            with torch.inference_mode():
                left, right = iw3u.apply_divergence(d, im[None], iw3_args, side)
            if Hc != H:  # restore the letterbox bars
                left = F.pad(left, (0, 0, y0, H - y1))
                right = F.pad(right, (0, 0, y0, H - y1))
            if args.layout == "half":
                left = F.interpolate(left, size=(H, W // 2), mode="bilinear", antialias=True, align_corners=False)
                right = F.interpolate(right, size=(H, W // 2), mode="bilinear", antialias=True, align_corners=False)
            enc_q.put((pts, rgb_to_yuv420(torch.cat([left, right], 3)[0]).cpu().numpy()))
            stats["frames"] += 1
            if stats["frames"] % 240 == 0:
                el = time.perf_counter() - stats["t0"]
                fps = stats["frames"] / el
                eta = f", ETA {((total - stats['frames']) / fps) / 60:.0f} min" if total else ""
                print(f"{stats['frames']}{'/' + str(total) if total else ''} frames, {stats['shots']} shots, "
                      f"{fps:.1f} fps{eta}", flush=True)
        stats["t_render"] += time.perf_counter() - t0

    cuts = CutDetector(threshold=args.cut_threshold)
    while (item := dec_q.get()) is not None:
        pts, arr = item
        rgb_t = torch.from_numpy(arr).to(DEV).permute(2, 0, 1).float() / 255
        if cuts(rgb_t):
            render(vda.flush())  # finish the old shot
            norm.reset()
            stats["shots"] += 1
        rgb_q.append((pts, rgb_t))
        render(vda.push(rgb_t))
    render(vda.flush())
    enc_q.put(None)
    for t in threads:
        t.join()
    inp.close()
    el = time.perf_counter() - stats["t0"]
    print(f"video done: {stats['frames']} frames, {stats['shots']} shots, {el / 60:.1f} min, "
          f"{stats['frames'] / el:.1f} fps (render share {stats['t_render'] / el:.0%})", flush=True)

    try:
        remux(tmp_video, args.input, out_path, args.start, args.duration)
        tmp_video.unlink()
        print("wrote", out_path)
    except Exception as e:  # keep the video even if a source stream can't be copied
        print(f"remux failed ({type(e).__name__}: {e}); video-only file kept at {tmp_video}")


def remux(video_path, src_path, out_path, start=0.0, duration=None):
    """Our SBS video + every audio and subtitle stream of the source (stream copy, no re-encode), interleaved by
    time so memory stays flat for a full movie. Tags the video as left-right stereo for players that read it."""
    vin, sin = av.open(str(video_path)), av.open(str(src_path))
    out = av.open(str(out_path), "w")
    v_out = out.add_stream_from_template(vin.streams.video[0])
    v_out.metadata["stereo_mode"] = "left_right"
    smap = {}
    for st in sin.streams:
        if st.type in ("audio", "subtitle"):
            try:
                smap[st.index] = out.add_stream_from_template(st)
                for k, val in st.metadata.items():  # language / title tags, so players can pick tracks
                    if k.upper() not in ("DURATION", "NUMBER_OF_FRAMES", "NUMBER_OF_BYTES", "BPS"):
                        smap[st.index].metadata[k] = val
                try:
                    smap[st.index].disposition = st.disposition  # default / forced flags
                except Exception:
                    pass
            except Exception as e:
                print(f"  skipping {st.type} stream {st.index} ({st.codec_context.name}): {e}")
    end = start + duration if duration else None

    def gen(container, streams, offset, mapping):
        for pkt in container.demux(*streams):
            if pkt.dts is None and pkt.pts is None:
                continue
            ts = pkt.dts if pkt.dts is not None else pkt.pts
            t = float(ts * pkt.time_base)
            if t < offset or (end is not None and t >= end):
                if end is not None and t >= end + 5:
                    return
                continue
            yield t - offset, mapping, pkt

    def tag(it, k):
        for t, mapping, pkt in it:
            yield t, k, mapping, pkt

    vids = tag(gen(vin, [vin.streams.video[0]], 0.0, {vin.streams.video[0].index: v_out}), 0)
    srcs = tag(gen(sin, [sin.streams[i] for i in smap], start, smap), 1) if smap else iter(())
    for _, _, mapping, pkt in heapq.merge(vids, srcs, key=lambda x: (x[0], x[1])):
        dst = mapping[pkt.stream.index]
        if mapping is smap and start:
            shift = int(round(start / pkt.time_base))
            if pkt.pts is not None:
                pkt.pts -= shift
            if pkt.dts is not None:
                pkt.dts -= shift
        pkt.stream = dst
        out.mux(pkt)
    out.close()
    vin.close()
    sin.close()


if __name__ == "__main__":
    main()
