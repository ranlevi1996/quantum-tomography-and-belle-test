import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# =========================
# CONFIG
# =========================
DATA_DIR = r"/Users/ranle/Downloads/RanNofit/RanNofit2/partb"
REFERENCE_BIT = "bit1"

PEAK_DISTANCE = 25
threshhold = 0.81
COINCIDENCE_WINDOW = 13

RESIZE_FACTOR = 0.81
RED_START_THRESHOLD = 2000
MAX_START_SEARCH_FRAMES = 3000

alpha_angles = [-45, 0, 45, 90]
beta_angles  = [-22.5, 22.5, 67.5, 112.5]

labels = ["Alice", "Bob"]
colors = ["b", "r"]

# =========================
# BIT → ANGLES (your final mapping)
# =========================
BIT_TO_ANGLES = {
    "bit1":  (45, 112.5),  "bit2":  (45, 67.5),  "bit3":  (45, 22.5),  "bit4":  (-45, -22.5),
    "bit5":  (0, 112.5),   "bit6":  (0, 67.5),   "bit7":  (0, 22.5),   "bit8":  (0, -22.5),
    "bit9":  (-45, 112.5), "bit10": (-45, 67.5), "bit11": (-45, 22.5), "bit12": (45, -22.5),
    "bit13": (90, 112.5),  "bit14": (90, 67.5),  "bit15": (90, 22.5),  "bit16": (90, -22.5),
}
ANGLES_TO_BIT = {(a, b): bit for bit, (a, b) in BIT_TO_ANGLES.items()}

# =========================
# Helpers
# =========================
def normalize_signal(x):
    """Normalizes a signal to [0,1]."""
    x = x.astype(float)
    return (x - x.min()) / (x.max() - x.min()) if x.max() > x.min() else np.zeros_like(x)

def red_sum_in_roi(frame, roi):
    """Returns summed red-mask intensity inside ROI."""
    x, y, w, h = roi
    hsv = cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2HSV)
    m1 = cv2.inRange(hsv, (0, 80, 80), (12, 255, 255))
    m2 = cv2.inRange(hsv, (168, 80, 80), (180, 255, 255))
    return int(np.sum(m1 | m2))

def find_first_red(sig):
    """Finds first index where signal crosses red threshold."""
    for i in range(min(len(sig), MAX_START_SEARCH_FRAMES)):
        if sig[i] >= RED_START_THRESHOLD:
            return i
    return 0

def select_rois(video_path):
    """Lets user select ROIs for Alice and Bob once on a reference frame."""
    cap = cv2.VideoCapture(video_path)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"Cannot read reference video frame: {video_path}")

    small = cv2.resize(frame, None, fx=RESIZE_FACTOR, fy=RESIZE_FACTOR)
    scale = 1 / RESIZE_FACTOR

    rois = []
    for lab in labels:
        r = cv2.selectROI(f"Select ROI for {lab}", small, False)
        rois.append(tuple(int(v * scale) for v in r))
    cv2.destroyAllWindows()
    return rois

def analyze_video(path, rois):
    """Reads video, aligns both cameras to a common start_used=min(startA,startB), returns two signals."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")

    A, B = [], []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        A.append(red_sum_in_roi(frame, rois[0]))
        B.append(red_sum_in_roi(frame, rois[1]))
    cap.release()

    A, B = np.array(A), np.array(B)
    start_used = min(find_first_red(A), find_first_red(B))

    A = A[start_used:]
    B = B[start_used:]
    L = min(len(A), len(B))
    return A[:L], B[:L]

def coincidences(pa, pb, window):
    """Counts peak coincidences within ±window frames."""
    used = np.zeros(len(pb), bool)
    n = 0
    for x in pa:
        d = np.abs(pb - x)
        j = int(np.argmin(d))
        if d[j] <= window and not used[j]:
            used[j] = True
            n += 1
    return int(n)

def canon_alpha(a):
    """Maps alpha into the allowed set (using mod 180 with -90≡90 convention)."""
    a = ((a + 90) % 180) - 90
    if abs(a + 90) < 1e-9:
        a = 90.0
    # snap
    return min(alpha_angles, key=lambda v: abs(v - a))

def canon_beta(b):
    """Maps beta into the allowed representation: -22.5, 22.5, 67.5, 112.5."""
    b = b % 180.0
    # convert 157.5 -> -22.5 representation
    if abs(b - 157.5) < 1e-6:
        return -22.5
    # snap to nearest among {22.5, 67.5, 112.5, 157.5}
    candidates_0_180 = [22.5, 67.5, 112.5, 157.5]
    b0 = min(candidates_0_180, key=lambda v: abs(v - b))
    return -22.5 if abs(b0 - 157.5) < 1e-6 else float(b0)

# =========================
# MAIN: build grid
# =========================
ref_video = os.path.join(DATA_DIR, f"{REFERENCE_BIT}.mp4")
rois = select_rois(ref_video)

counts = {}

fig, axes = plt.subplots(4, 4, figsize=(14, 10), sharey=True)
for i, a in enumerate(alpha_angles):
    for j, b in enumerate(beta_angles):
        bit = ANGLES_TO_BIT[(a, b)]
        path = os.path.join(DATA_DIR, f"{bit}.mp4")

        A, B = analyze_video(path, rois)
        nA, nB = normalize_signal(A), normalize_signal(B)

        pA, _ = find_peaks(nA, height=threshhold, distance=PEAK_DISTANCE)
        pB, _ = find_peaks(nB, height=threshhold, distance=PEAK_DISTANCE)

        N = coincidences(pA, pB, COINCIDENCE_WINDOW)
        counts[(a, b)] = N

        ax = axes[i, j]
        ax.plot(nA, color=colors[0], alpha=0.6)
        ax.plot(nB, color=colors[1], alpha=0.6)
        ax.axhline(threshhold, ls="--", c="k", alpha=0.3)
        ax.set_title(f"α={a}, β={b}\nN={N}")

plt.tight_layout()
plt.show()

print("\n--- Final Grid ---")
for a in alpha_angles:
    for b in beta_angles:
        print(f"N({a},{b}) = {counts[(a,b)]}")

# =========================
# CHSH
# =========================
def getN(a, b):
    """Fetches N for (a,b) after canonicalizing alpha/beta to existing keys."""
    aa = canon_alpha(a)
    bb = canon_beta(b)
    return counts[(aa, bb)]

def E(a, b):
    """Computes correlation E(a,b) from the 4 coincidence counts."""
    aa = canon_alpha(a)
    bb = canon_beta(b)
    ap = canon_alpha(a + 90)
    bp = canon_beta(b + 90)

    N_ab   = getN(aa, bb)
    N_apbp = getN(ap, bp)
    N_abp  = getN(aa, bp)
    N_apb  = getN(ap, bb)

    denom = N_ab + N_apbp + N_abp + N_apb
    return 0.0 if denom == 0 else (N_ab + N_apbp - N_abp - N_apb) / denom

S = E(0, 22.5) - E(0, 67.5) + E(45, 22.5) + E(45, 67.5)
print("\nS =", S)
