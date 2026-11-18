"""One-frame smoke test for stereo_views: runs TMPI + naive warp, writes out/smoke_tmpi.png."""
import cv2
import numpy as np

from common import OUT, load_depth, read_clip
from stereo_views import TMPIStereo, clip_disp_range, normalize_disp, warp_stereo

frames, _ = read_clip("rollercoaster")
d = load_depth("rollercoaster", "vda_s")
lo, hi = clip_disp_range(d)
t = 30
disp = normalize_disp(d[t], lo, hi)
s = TMPIStereo()
for _ in range(2):  # second call = warm timing
    src, right = s(frames[t], disp, baselines=[0.0, 0.04])
print(s.timing, "planes", s.last_planes.shape, s.last_planes[:3])
naive = warp_stereo(frames[t], disp, 0.04)
row1 = np.hstack([frames[t], src])
row2 = np.hstack([right, naive])
img = cv2.cvtColor(np.vstack([row1, row2]), cv2.COLOR_RGB2BGR)
for (x, y, txt) in [(10, 30, "input"), (970, 30, "TMPI b=0"), (10, 570, "TMPI right b=0.04"),
                    (970, 570, "naive warp b=0.04")]:
    cv2.putText(img, txt, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
cv2.imwrite(str(OUT / "smoke_tmpi.png"), img)
print("diff src vs input (mean abs):", np.abs(src.astype(int) - frames[t].astype(int)).mean())
