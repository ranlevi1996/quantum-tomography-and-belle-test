import os
import re
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# =========================================================
# CONFIG
# =========================================================
SHOW_PLOTS = True  # אם plt.show() תוקע לך – שנה ל-False (הקבצים נשמרים בכל מקרה)

# ---------- Part A ----------
DATA_DIR_A = r"/Users/ranle/Downloads/RanNofit/RanNofit2/part a - exp"
VIDEO_REGEX_A = r"partb(\d+)bits\.mp4"  # partb10bits.mp4, partb25bits.mp4, partb50bits.mp4

RESIZE_FACTOR_A = 0.8
PEAK_DISTANCE_A = 20
THRESHOLD_A = 0.81
COINCIDENCE_WINDOW_A = 12
RED_START_THRESHOLD_A = 2000
MAX_START_SEARCH_FRAMES_A = 3000

# ROI order is IMPORTANT:
# 0: Alice V, 1: Alice H, 2: Bob V, 3: Bob H
ROI_LABELS_A = ["Alice V", "Alice H", "Bob V", "Bob H"]
BASIS_LABELS = ["HH", "HV", "VH", "VV"]  # density matrix basis order

# ---------- Part B ----------
DATA_DIR_B = r"/Users/ranle/Downloads/RanNofit/RanNofit2/partb"
REFERENCE_BIT_B = "bit1"

RESIZE_FACTOR_B = 0.8
PEAK_DISTANCE_B = 20
THRESHOLD_B = 0.81
COINCIDENCE_WINDOW_B = 12
RED_START_THRESHOLD_B = 2000
MAX_START_SEARCH_FRAMES_B = 3000

alpha_angles = [-45, 0, 45, 90]
beta_angles = [-22.5, 22.5, 67.5, 112.5]

LABELS_B = ["Alice", "Bob"]
COLORS_B = ["b", "r"]

# bit -> (alpha, beta) (your final mapping)
BIT_TO_ANGLES_B = {
    "bit1":  (45, 112.5),  "bit2":  (45, 67.5),  "bit3":  (45, 22.5),   "bit4":  (-45, -22.5),
    "bit5":  (0, 112.5),   "bit6":  (0, 67.5),   "bit7":  (0, 22.5),    "bit8":  (0, -22.5),
    "bit9":  (-45, 112.5), "bit10": (-45, 67.5), "bit11": (-45, 22.5),  "bit12": (45, -22.5),
    "bit13": (90, 112.5),  "bit14": (90, 67.5),  "bit15": (90, 22.5),   "bit16": (90, -22.5),
}
ANGLES_TO_BIT_B = {(a, b): bit for bit, (a, b) in BIT_TO_ANGLES_B.items()}


# =========================================================
# Shared helpers (each has a 1-liner)
# =========================================================
def normalize_signal(x: np.ndarray) -> np.ndarray:
    """Normalize a 1D signal to [0,1]."""
    x = x.astype(np.float64)
    d = np.max(x) - np.min(x)
    return np.zeros_like(x) if d == 0 else (x - np.min(x)) / d


def red_sum_in_roi(frame_bgr: np.ndarray, roi: tuple) -> int:
    """Return red-mask pixel sum inside ROI using HSV thresholding."""
    x, y, w, h = roi
    roi_img = frame_bgr[y:y + h, x:x + w]
    hsv = cv2.cvtColor(roi_img, cv2.COLOR_BGR2HSV)
    m1 = cv2.inRange(hsv, (0, 80, 80), (12, 255, 255))
    m2 = cv2.inRange(hsv, (168, 80, 80), (180, 255, 255))
    return int(np.sum(m1 | m2))


def first_red_index(sig: np.ndarray, thr: int, max_frames: int) -> int:
    """Return the first index where sig>=thr within max_frames (else 0)."""
    m = min(len(sig), max_frames)
    for i in range(m):
        if sig[i] >= thr:
            return i
    return 0


def coincidences(peaks_a: np.ndarray, peaks_b: np.ndarray, window: int) -> int:
    """Count 1-to-1 nearest peak matches within ±window frames."""
    if len(peaks_a) == 0 or len(peaks_b) == 0:
        return 0
    pa = np.asarray(peaks_a)
    pb = np.asarray(peaks_b)
    used = np.zeros(len(pb), dtype=bool)
    c = 0
    for x in pa:
        j = int(np.argmin(np.abs(pb - x)))
        if (abs(pb[j] - x) <= window) and (not used[j]):
            used[j] = True
            c += 1
    return int(c)


def select_rois_once(video_path: str, resize_factor: float, labels: list[str]) -> list[tuple]:
    """Let user select ROIs once on a reference frame and return them in full-res coordinates."""
    cap = cv2.VideoCapture(video_path)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"Cannot read reference frame: {video_path}")

    h, w = frame.shape[:2]
    small = cv2.resize(frame, (int(w * resize_factor), int(h * resize_factor)))
    scale = 1.0 / resize_factor

    rois = []
    for lab in labels:
        r = cv2.selectROI(f"Select ROI for {lab} (Draw + ENTER/SPACE)", small, fromCenter=False, showCrosshair=True)
        cv2.destroyWindow(f"Select ROI for {lab} (Draw + ENTER/SPACE)")
        if r == (0, 0, 0, 0):
            raise RuntimeError(f"Empty ROI selected for {lab}")
        rois.append((int(r[0] * scale), int(r[1] * scale), int(r[2] * scale), int(r[3] * scale)))

    cv2.destroyAllWindows()
    return rois


# =========================================================
# Part A helpers (each has a 1-liner)
# =========================================================
def read_signals_4roi(video_path: str, rois: list[tuple], red_thr: int, max_search: int) -> list[np.ndarray]:
    """Read 4 ROI signals from video and cut all from common start_used=min(starts)."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return [np.array([])] * 4

    sig_all = [[] for _ in range(4)]
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        for k in range(4):
            sig_all[k].append(red_sum_in_roi(frame, rois[k]))
    cap.release()

    sig_all = [np.array(s) for s in sig_all]
    starts = [first_red_index(s, red_thr, max_search) for s in sig_all]
    start_used = min(starts)

    sig_cut = [s[start_used:] for s in sig_all]
    if any(len(s) == 0 for s in sig_cut):
        return [np.array([])] * 4

    L = min(len(s) for s in sig_cut)
    return [s[:L] for s in sig_cut]


def peaks_from_signal(sig: np.ndarray, thr: float, dist: int) -> tuple[np.ndarray, np.ndarray]:
    """Return normalized signal and its peak indices using find_peaks."""
    ns = normalize_signal(sig)
    p, _ = find_peaks(ns, height=thr, distance=dist)
    return ns, p


def counts_HV_basis(peaks: list[np.ndarray], window: int, bob_swap: bool = False) -> tuple[int, int, int, int]:
    """Compute (N_HH,N_HV,N_VH,N_VV) from peaks in the fixed order [A_V,A_H,B_V,B_H]."""
    A_V, A_H = peaks[0], peaks[1]
    B_V, B_H = peaks[2], peaks[3]
    if bob_swap:
        B_V, B_H = B_H, B_V
    N_HH = coincidences(A_H, B_H, window)
    N_HV = coincidences(A_H, B_V, window)
    N_VH = coincidences(A_V, B_H, window)
    N_VV = coincidences(A_V, B_V, window)
    return N_HH, N_HV, N_VH, N_VV


def rho_from_counts_phi(N_HH: int, N_HV: int, N_VH: int, N_VV: int) -> np.ndarray:
    """Build a |HH>+|VV> style density matrix (HVVH) from coincidence counts."""
    tot = N_HH + N_HV + N_VH + N_VV
    if tot == 0:
        return np.zeros((4, 4), dtype=complex)
    pHH = N_HH / tot
    pVV = N_VV / tot
    coh = np.sqrt(max(pHH, 0) * max(pVV, 0))
    rho = np.zeros((4, 4), dtype=complex)
    rho[0, 0] = pHH
    rho[3, 3] = pVV
    rho[0, 3] = coh
    rho[3, 0] = coh
    return rho


def rho_from_counts_psi(N_HH: int, N_HV: int, N_VH: int, N_VV: int) -> np.ndarray:
    """Build a |HV>+|VH> style density matrix (HHVV) from coincidence counts."""
    tot = N_HH + N_HV + N_VH + N_VV
    if tot == 0:
        return np.zeros((4, 4), dtype=complex)
    pHV = N_HV / tot
    pVH = N_VH / tot
    coh = np.sqrt(max(pHV, 0) * max(pVH, 0))
    rho = np.zeros((4, 4), dtype=complex)
    rho[1, 1] = pHV
    rho[2, 2] = pVH
    rho[1, 2] = coh
    rho[2, 1] = coh
    return rho


def plot_rho_3d(ax, rho: np.ndarray, title: str) -> None:
    """Plot a 3D bar chart of |rho_ij|."""
    Z = np.abs(rho)
    xs, ys = np.meshgrid(np.arange(4), np.arange(4))
    x = xs.ravel()
    y = ys.ravel()
    z = np.zeros_like(x, dtype=float)
    dz = Z.ravel()

    dx = 0.6 * np.ones_like(dz)
    dy = 0.6 * np.ones_like(dz)

    ax.bar3d(x, y, z, dx, dy, dz, shade=True)
    ax.set_xticks(np.arange(4) + 0.3)
    ax.set_yticks(np.arange(4) + 0.3)
    ax.set_xticklabels(BASIS_LABELS)
    ax.set_yticklabels(BASIS_LABELS)
    ax.set_zlim(0, 0.55)
    ax.set_title(title, pad=10)


# =========================================================
# Part B helpers (each has a 1-liner)
# =========================================================
def read_signals_2roi(video_path: str, rois: list[tuple], red_thr: int, max_search: int) -> tuple[np.ndarray, np.ndarray]:
    """Read 2 ROI signals (Alice,Bob) and cut both from common start_used=min(starts)."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    A, B = [], []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        A.append(red_sum_in_roi(frame, rois[0]))
        B.append(red_sum_in_roi(frame, rois[1]))
    cap.release()

    A = np.array(A)
    B = np.array(B)
    start_used = min(first_red_index(A, red_thr, max_search), first_red_index(B, red_thr, max_search))

    A = A[start_used:]
    B = B[start_used:]
    L = min(len(A), len(B))
    return A[:L], B[:L]


def canon_alpha(a: float) -> float:
    """Canonicalize alpha using mod-180 and snap to the allowed alpha grid."""
    a = ((a + 90) % 180) - 90
    if abs(a + 90) < 1e-9:
        a = 90.0
    return float(min(alpha_angles, key=lambda v: abs(v - a)))


def canon_beta(b: float) -> float:
    """Canonicalize beta to representation in { -22.5, 22.5, 67.5, 112.5 } with -22.5≡157.5."""
    b = b % 180.0
    if abs(b - 157.5) < 1e-6:
        return -22.5
    candidates_0_180 = [22.5, 67.5, 112.5, 157.5]
    b0 = min(candidates_0_180, key=lambda v: abs(v - b))
    return -22.5 if abs(b0 - 157.5) < 1e-6 else float(b0)


# =========================================================
# RUN: Part A
# =========================================================
def run_part_a() -> None:
    """Run Part A: build HVVH/HHVV density matrices for 10/25/50 and save a 2x3 3D plot."""
    filesA = []
    for fn in os.listdir(DATA_DIR_A):
        m = re.match(VIDEO_REGEX_A, fn)
        if m:
            bits = int(m.group(1))
            filesA.append((bits, os.path.join(DATA_DIR_A, fn)))
    filesA.sort(key=lambda t: t[0])

    if len(filesA) == 0:
        raise RuntimeError(f"No videos matching '{VIDEO_REGEX_A}' in: {DATA_DIR_A}")

    # pick ROIs once from the first file
    _, ref_path_A = filesA[0]
    roisA = select_rois_once(ref_path_A, RESIZE_FACTOR_A, ROI_LABELS_A)

    figA = plt.figure(figsize=(14, 9))

    for col, (bits, path) in enumerate(filesA[:3]):  # expect 10/25/50
        sigs = read_signals_4roi(path, roisA, RED_START_THRESHOLD_A, MAX_START_SEARCH_FRAMES_A)
        if any(len(s) == 0 for s in sigs):
            print(f"\n--- PART A ({bits} bits) ---")
            print("WARNING: empty signal(s), skipping.")
            continue

        peaks = []
        for k in range(4):
            _, pk = peaks_from_signal(sigs[k], THRESHOLD_A, PEAK_DISTANCE_A)
            peaks.append(pk)

        # HVVH (no swap)
        N_HH, N_HV, N_VH, N_VV = counts_HV_basis(peaks, COINCIDENCE_WINDOW_A, bob_swap=False)
        rho_hvvh = rho_from_counts_phi(N_HH, N_HV, N_VH, N_VV)

        # HHVV (Bob swap)
        N_HH2, N_HV2, N_VH2, N_VV2 = counts_HV_basis(peaks, COINCIDENCE_WINDOW_A, bob_swap=True)
        rho_hhvv = rho_from_counts_psi(N_HH2, N_HV2, N_VH2, N_VV2)

        print(f"\n--- PART A ({bits} bits) ---")
        print("HVVH rho=\n", np.round(rho_hvvh, 4))
        print("HHVV rho=\n", np.round(rho_hhvv, 4))

        ax1 = figA.add_subplot(2, 3, col + 1, projection="3d")
        plot_rho_3d(ax1, rho_hvvh, title=f"HVVH {bits} bits")

        ax2 = figA.add_subplot(2, 3, 3 + col + 1, projection="3d")
        plot_rho_3d(ax2, rho_hhvv, title=f"HHVV {bits} bits")

    # tighter layout for 3D can be tricky; this avoids the "tight_layout not applied" spam
    figA.subplots_adjust(left=0.04, right=0.98, top=0.93, bottom=0.06, wspace=0.15, hspace=0.25)
    outA = "PartA_density_matrices.png"
    figA.savefig(outA, dpi=300)
    if SHOW_PLOTS:
        plt.show(block=False)
        plt.pause(0.1)
        plt.close(figA)
    plt.close(figA)


# =========================================================
# RUN: Part B
# =========================================================
def run_part_b() -> None:
    """Run Part B: build the 4x4 grid plot, print Final Grid and CHSH S, and save the plot."""
    ref_video_B = os.path.join(DATA_DIR_B, f"{REFERENCE_BIT_B}.mp4")
    roisB = select_rois_once(ref_video_B, RESIZE_FACTOR_B, LABELS_B)

    countsB: dict[tuple[float, float], int] = {}

    figB, axesB = plt.subplots(4, 4, figsize=(14, 10), sharey=True)

    for i, a in enumerate(alpha_angles):
        for j, b in enumerate(beta_angles):
            bit = ANGLES_TO_BIT_B[(a, b)]
            path = os.path.join(DATA_DIR_B, f"{bit}.mp4")

            A, B = read_signals_2roi(path, roisB, RED_START_THRESHOLD_B, MAX_START_SEARCH_FRAMES_B)
            nA, nB = normalize_signal(A), normalize_signal(B)

            pA, _ = find_peaks(nA, height=THRESHOLD_B, distance=PEAK_DISTANCE_B)
            pB, _ = find_peaks(nB, height=THRESHOLD_B, distance=PEAK_DISTANCE_B)

            N = coincidences(pA, pB, COINCIDENCE_WINDOW_B)
            countsB[(a, b)] = N

            ax = axesB[i, j]
            ax.plot(nA, color=COLORS_B[0], alpha=0.6)
            ax.plot(nB, color=COLORS_B[1], alpha=0.6)
            ax.axhline(THRESHOLD_B, ls="--", c="k", alpha=0.3)
            ax.set_title(f"α={a}, β={b}\nN={N}")

            if i == 3:
                ax.set_xlabel("Frame")
            if j == 0:
                ax.set_ylabel("Norm. Intensity")

    figB.tight_layout()
    outB = "PartB_grid.png"
    figB.savefig(outB, dpi=300)
    if SHOW_PLOTS:
        plt.show()
    plt.close(figB)

    print("\n--- Final Grid ---")
    for a in alpha_angles:
        for b in beta_angles:
            print(f"N({a},{b}) = {countsB[(a, b)]}")

    def getN(a: float, b: float) -> int:
        """Fetch N(a,b) after canonicalizing angles to existing keys."""
        return countsB[(canon_alpha(a), canon_beta(b))]

    def E(a: float, b: float) -> float:
        """Compute correlation E(a,b) from the four coincidence counts."""
        aa = canon_alpha(a)
        bb = canon_beta(b)
        ap = canon_alpha(a + 90)
        bp = canon_beta(b + 90)

        N_ab = getN(aa, bb)
        N_apbp = getN(ap, bp)
        N_abp = getN(aa, bp)
        N_apb = getN(ap, bb)

        denom = N_ab + N_apbp + N_abp + N_apb
        return 0.0 if denom == 0 else (N_ab + N_apbp - N_abp - N_apb) / denom

    S = E(0, 22.5) - E(0, 67.5) + E(45, 22.5) + E(45, 67.5)
    print("\nS =", S)


# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":
    run_part_a()
    run_part_b()
