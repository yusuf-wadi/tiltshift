"""List video files with resolution, codec, HDR transfer, fps, length and audio, to pick conversion sources."""
import glob
import os
import sys

import av

root = sys.argv[1]  # folder to scan
for p in sorted(glob.glob(f"{root}/**/*.m*", recursive=True)):
    if os.path.getsize(p) < 500e6:
        continue
    try:
        c = av.open(p)
        v = c.streams.video[0]
        cc = v.codec_context
        aud = [f"{s.codec_context.name}/{s.metadata.get('language', '?')}" for s in c.streams.audio][:3]
        trc = cc.color_trc
        print(f"{os.path.basename(p)[:55]:55s} {cc.width}x{cc.height} {cc.name} {cc.pix_fmt} trc={trc} "
              f"{float(v.average_rate or 0):.3f}fps {c.duration / 6e7:.0f}min audio={aud}")
        c.close()
    except Exception as e:
        print(p, type(e).__name__, e)
