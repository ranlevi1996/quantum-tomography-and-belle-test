import os
import re
import json
from dataclasses import dataclass
from datetime import datetime
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import argparse
from matplotlib import cm
from matplotlib import colors as mcolors


@dataclass
class Config:
    show_plots: bool = True
    resize_factor: float = 0.8
    peak_distance: int = 20
    threshold: float = 0.81
    coincidence_window: int = 12
    red_start_threshold: int = 2000
    max_start_search_frames: int = 3000

    data_dir_a: str = r"/Users/ranle/Downloads/RanNofit/RanNofit2/part a - exp"
    video_regex_a: str = r"partb(\d+)bits\.mp4"
    roi_labels_a: tuple = ("Alice V", "Alice H", "Bob V", "Bob H")
    basis_labels: tuple = ("HH", "HV", "VH", "VV")
    roi_cache_a: str = "roi_cache_partA.json"
    quality_report_a: str = "quality_report_partA.txt"

    data_dir_b: str = r"/Users/ranle/Downloads/RanNofit/RanNofit2/partb"
    reference_bit_b: str = "bit1"
    labels_b: tuple = ("Alice", "Bob")
    roi_cache_b: str = "roi_cache_partB.json"
    quality_report_b: str = "quality_report_partB.txt"
    alpha_angles: tuple = (-45, 0, 45, 90)
    beta_angles: tuple = (-22.5, 22.5, 67.5, 112.5)


CFG = Config()

ANGLES_TO_BIT_B = {
    (45, 112.5): "bit1",   (45, 67.5): "bit2",   (45, 22.5): "bit3",    (-45, -22.5): "bit4",
    (0, 112.5):  "bit5",   (0, 67.5):  "bit6",   (0, 22.5):  "bit7",    (0, -22.5):   "bit8",
    (-45, 112.5): "bit9",  (-45, 67.5): "bit10", (-45, 22.5): "bit11",  (45, -22.5):  "bit12",
    (90, 112.5): "bit13",  (90, 67.5): "bit14",  (90, 22.5): "bit15",   (90, -22.5):  "bit16",
}


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def save_roi_cache(path: str, video_path: str, labels: list[str],
                   rois: list[tuple[int, int, int, int]], cfg: Config) -> None:
    payload = {
        "created_at": _now(),
        "video_path": video_path,
        "labels": labels,
        "rois": [list(map(int, r)) for r in rois],
        "resize_factor": cfg.resize_factor,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def load_roi_cache(path: str, expected_labels: list[str]) -> list[tuple[int, int, int, int]] | None:
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        labels = payload.get("labels", [])
        if list(labels) != list(expected_labels):
            return None
        rois = payload.get("rois", None)
        if rois is None:
            return None
        rois = [tuple(map(int, r)) for r in rois]
        if len(rois) != len(expected_labels):
            return None
        return rois
    except Exception:
        return None


def append_report(path: str, lines: list[str]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        for ln in lines:
            f.write(ln.rstrip() + "\n")


def normalize_signal(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64)
    d = np.max(x) - np.min(x)
    return np.zeros_like(x) if d == 0 else (x - np.min(x)) / d


def red_sum_in_roi(frame_bgr: np.ndarray, roi: tuple[int, int, int, int]) -> int:
    x, y, w, h = roi
    roi_img = frame_bgr[y:y + h, x:x + w]
    hsv = cv2.cvtColor(roi_img, cv2.COLOR_BGR2HSV)
    m1 = cv2.inRange(hsv, (0, 80, 80), (12, 255, 255))
    m2 = cv2.inRange(hsv, (168, 80, 80), (180, 255, 255))
    return int(np.sum(m1 | m2))


def first_red_index(sig: np.ndarray, thr: int, max_frames: int) -> int:
    m = min(len(sig), max_frames)
    for i in range(m):
        if sig[i] >= thr:
            return i
    return 0


def coincidences(peaks_a: np.ndarray, peaks_b: np.ndarray, window: int) -> int:
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


def select_roi_white(win: str, img_bgr: np.ndarray) -> tuple[int, int, int, int]:
    drawing = False
    x0 = y0 = x1 = y1 = 0
    show = img_bgr.copy()

    def on_mouse(event, x, y, flags, param):
        nonlocal drawing, x0, y0, x1, y1, show
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            x0, y0 = x, y
            x1, y1 = x, y
        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            x1, y1 = x, y
            show = img_bgr.copy()
            cv2.rectangle(show, (x0, y0), (x1, y1), (255, 255, 255), 2)
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            x1, y1 = x, y
            show = img_bgr.copy()
            cv2.rectangle(show, (x0, y0), (x1, y1), (255, 255, 255), 2)

    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, on_mouse)

    while True:
        cv2.imshow(win, show)
        k = cv2.waitKey(10) & 0xFF
        if k in (13, 10):
            break
        if k == 27:
            cv2.destroyWindow(win)
            raise RuntimeError("ROI selection cancelled.")
    cv2.destroyWindow(win)

    xa, xb = sorted([x0, x1])
    ya, yb = sorted([y0, y1])
    w = max(1, xb - xa)
    h = max(1, yb - ya)
    return (xa, ya, w, h)


def select_rois_once(video_path: str, labels: list[str], cfg: Config) -> list[tuple[int, int, int, int]]:
    cap = cv2.VideoCapture(video_path)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"Cannot read reference frame: {video_path}")

    h, w = frame.shape[:2]
    small = cv2.resize(frame, (int(w * cfg.resize_factor), int(h * cfg.resize_factor)))
    scale = 1.0 / cfg.resize_factor

    rois = []
    for lab in labels:
        r_small = select_roi_white(f"ROI: {lab} (drag, ENTER=ok, ESC=cancel)", small)
        rois.append((
            int(r_small[0] * scale),
            int(r_small[1] * scale),
            int(r_small[2] * scale),
            int(r_small[3] * scale),
        ))

    cv2.destroyAllWindows()
    return rois


def get_rois_with_cache(cache_path: str, video_path: str, labels: list[str],
                        cfg: Config, force_reselect: bool) -> list[tuple[int, int, int, int]]:
    if not force_reselect:
        cached = load_roi_cache(cache_path, labels)
        if cached is not None:
            return cached
    rois = select_rois_once(video_path, labels, cfg)
    save_roi_cache(cache_path, video_path, labels, rois, cfg)
    return rois


# Part A

def read_signals_nroi(video_path: str, rois: list[tuple[int, int, int, int]], cfg: Config) -> tuple[list[np.ndarray], list[int], int]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return [np.array([])] * len(rois), [0] * len(rois), 0
    sig = [[] for _ in rois]
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        for k, roi in enumerate(rois):
            sig[k].append(red_sum_in_roi(frame, roi))
    cap.release()
    sig = [np.array(s) for s in sig]
    starts = [first_red_index(s, cfg.red_start_threshold, cfg.max_start_search_frames) for s in sig]
    start_used = min(starts)
    sig = [s[start_used:] for s in sig]
    if any(len(s) == 0 for s in sig):
        return [np.array([])] * len(rois), starts, start_used
    L = min(len(s) for s in sig)
    sig = [s[:L] for s in sig]
    return sig, starts, start_used


def peaks_from(sig: np.ndarray, cfg: Config) -> np.ndarray:
    ns = normalize_signal(sig)
    p, _ = find_peaks(ns, height=cfg.threshold, distance=cfg.peak_distance)
    return p


def counts_HV_basis(peaks4: list[np.ndarray], cfg: Config, bob_swap: bool = False) -> tuple[int, int, int, int]:
    A_V, A_H, B_V, B_H = peaks4
    if bob_swap:
        B_V, B_H = B_H, B_V
    N_HH = coincidences(A_H, B_H, cfg.coincidence_window)
    N_HV = coincidences(A_H, B_V, cfg.coincidence_window)
    N_VH = coincidences(A_V, B_H, cfg.coincidence_window)
    N_VV = coincidences(A_V, B_V, cfg.coincidence_window)
    return N_HH, N_HV, N_VH, N_VV


def rho_phi(N_HH: int, N_HV: int, N_VH: int, N_VV: int) -> np.ndarray:
    tot = N_HH + N_HV + N_VH + N_VV
    if tot == 0:
        return np.zeros((4, 4), dtype=complex)
    pHH, pVV = N_HH / tot, N_VV / tot
    coh = np.sqrt(max(pHH, 0) * max(pVV, 0))
    rho = np.zeros((4, 4), dtype=complex)
    rho[0, 0], rho[3, 3] = pHH, pVV
    rho[0, 3] = rho[3, 0] = coh
    return rho


def rho_psi(N_HH: int, N_HV: int, N_VH: int, N_VV: int) -> np.ndarray:
    tot = N_HH + N_HV + N_VH + N_VV
    if tot == 0:
        return np.zeros((4, 4), dtype=complex)
    pHV, pVH = N_HV / tot, N_VH / tot
    coh = np.sqrt(max(pHV, 0) * max(pVH, 0))
    rho = np.zeros((4, 4), dtype=complex)
    rho[1, 1], rho[2, 2] = pHV, pVH
    rho[1, 2] = rho[2, 1] = coh
    return rho


def plot_rho_3d(ax, rho: np.ndarray, title: str, cfg: Config) -> None:
    Z = np.abs(rho)
    xs, ys = np.meshgrid(np.arange(4), np.arange(4))
    x, y = xs.ravel(), ys.ravel()
    z = np.zeros_like(x, dtype=float)
    dz = Z.ravel()
    dx = 0.6 * np.ones_like(dz)
    dy = 0.6 * np.ones_like(dz)
    norm = mcolors.Normalize(
        vmin=float(np.min(dz)),
        vmax=float(np.max(dz)) if np.max(dz) > np.min(dz) else float(np.min(dz)) + 1e-12
    )
    cmap = cm.viridis
    bar_colors = cmap(norm(dz))
    ax.bar3d(x, y, z, dx, dy, dz, shade=True, color=bar_colors)
    ax.set_xticks(np.arange(4) + 0.3)
    ax.set_yticks(np.arange(4) + 0.3)
    ax.set_xticklabels(cfg.basis_labels)
    ax.set_yticklabels(cfg.basis_labels)
    ax.set_zlim(0, 0.55)
    ax.set_title(title, pad=10)
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    ax.figure.colorbar(sm, ax=ax, shrink=0.6, pad=0.08, label=r"$|\rho_{ij}|$")


def simulate_part_a_signals(cfg: Config, n_shots: int = 100, seed: int | None = None) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)

    alice_bits = rng.integers(0, 2, size=n_shots)
    bob_bits = alice_bits.copy()

    pre = 60
    gap = max(cfg.peak_distance + 5, 30)
    T = pre + n_shots * gap + 60

    base = 50
    peak = max(cfg.red_start_threshold + 500, 4000)

    A_V = np.full(T, base, dtype=np.int64)
    A_H = np.full(T, base, dtype=np.int64)
    B_V = np.full(T, base, dtype=np.int64)
    B_H = np.full(T, base, dtype=np.int64)

    for k in range(n_shots):
        t = pre + k * gap
        if alice_bits[k] == 0:
            A_H[t] = peak
        else:
            A_V[t] = peak
        if bob_bits[k] == 0:
            B_H[t] = peak
        else:
            B_V[t] = peak

    return [A_V, A_H, B_V, B_H]


def run_part_a_simulation(cfg: Config, n_shots: int = 100, seed: int | None = None) -> None:
    sigs = simulate_part_a_signals(cfg, n_shots=n_shots, seed=seed)
    pks = [peaks_from(s, cfg) for s in sigs]

    N = counts_HV_basis(pks, cfg, bob_swap=False)
    rhoA = rho_phi(*N)

    N2 = counts_HV_basis(pks, cfg, bob_swap=True)
    rhoB = rho_psi(*N2)

    print(f"\n--- PART A SIMULATION ({n_shots} shots) ---")
    print("N(HH,HV,VH,VV) no-swap =", N)
    print("HVVH rho=\n", np.round(rhoA, 4))
    print("N(HH,HV,VH,VV) swap    =", N2)
    print("HHVV rho=\n", np.round(rhoB, 4))

    fig = plt.figure(figsize=(10, 5))
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    plot_rho_3d(ax1, rhoA, f"SIM HVVH ({n_shots})", cfg)
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    plot_rho_3d(ax2, rhoB, f"SIM HHVV ({n_shots})", cfg)

    fig.tight_layout()
    fig.savefig("PartA_simulation_density_matrices.png", dpi=300)

    if cfg.show_plots:
        plt.show(block=False)
        plt.pause(0.1)
    plt.close(fig)


def run_part_a(cfg: Config, force_reselect_roi: bool) -> None:
    files = []
    for fn in os.listdir(cfg.data_dir_a):
        m = re.match(cfg.video_regex_a, fn)
        if m:
            bits = int(m.group(1))
            files.append((bits, os.path.join(cfg.data_dir_a, fn)))
    files.sort(key=lambda t: t[0])
    if not files:
        raise RuntimeError(f"No videos matching '{cfg.video_regex_a}' in: {cfg.data_dir_a}")

    roisA = get_rois_with_cache(cfg.roi_cache_a, files[0][1], list(cfg.roi_labels_a), cfg, force_reselect_roi)

    append_report(cfg.quality_report_a, [
        "",
        "=================================================",
        f"[{_now()}] Part A run",
        f"data_dir: {cfg.data_dir_a}",
        f"roi_cache: {cfg.roi_cache_a}",
        f"threshold={cfg.threshold}, peak_distance={cfg.peak_distance}, window={cfg.coincidence_window}",
    ])

    fig = plt.figure(figsize=(14, 9))
    for col, (bits, path) in enumerate(files[:3]):
        sigs, starts, start_used = read_signals_nroi(path, roisA, cfg)
        if any(len(s) == 0 for s in sigs):
            append_report(cfg.quality_report_a, [f"video={os.path.basename(path)} bits={bits} -> EMPTY signals, skipped"])
            continue
        pks = [peaks_from(s, cfg) for s in sigs]
        peaks_counts = [len(p) for p in pks]
        N = counts_HV_basis(pks, cfg, bob_swap=False)
        rhoA = rho_phi(*N)
        N2 = counts_HV_basis(pks, cfg, bob_swap=True)
        rhoB = rho_psi(*N2)

        print(f"\n--- PART A ({bits} bits) ---")
        print("HVVH rho=\n", np.round(rhoA, 4))
        print("HHVV rho=\n", np.round(rhoB, 4))

        append_report(cfg.quality_report_a, [
            f"video={os.path.basename(path)} bits={bits}",
            f"  starts_per_roi={starts}, start_used(min)={start_used}",
            f"  peaks_per_roi={peaks_counts}  (order: {list(cfg.roi_labels_a)})",
            f"  N(HH,HV,VH,VV) no-swap={N}  swap={N2}",
        ])

        ax1 = fig.add_subplot(2, 3, col + 1, projection="3d")
        plot_rho_3d(ax1, rhoA, f"HVVH {bits} bits", cfg)
        ax2 = fig.add_subplot(2, 3, 3 + col + 1, projection="3d")
        plot_rho_3d(ax2, rhoB, f"HHVV {bits} bits", cfg)

    fig.subplots_adjust(left=0.04, right=0.98, top=0.93, bottom=0.06, wspace=0.15, hspace=0.25)
    fig.savefig("PartA_density_matrices.png", dpi=300)
    if cfg.show_plots:
        plt.show(block=False)
        plt.pause(0.1)
    plt.close(fig)
    run_part_a_simulation(cfg, n_shots=100, seed=1)



# Part B

def read_signals_2roi(video_path: str, rois2: list[tuple[int, int, int, int]], cfg: Config) -> tuple[np.ndarray, np.ndarray, tuple[int, int], int]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    A, B = [], []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        A.append(red_sum_in_roi(frame, rois2[0]))
        B.append(red_sum_in_roi(frame, rois2[1]))
    cap.release()
    A = np.array(A)
    B = np.array(B)
    startA = first_red_index(A, cfg.red_start_threshold, cfg.max_start_search_frames)
    startB = first_red_index(B, cfg.red_start_threshold, cfg.max_start_search_frames)
    start_used = min(startA, startB)
    A, B = A[start_used:], B[start_used:]
    L = min(len(A), len(B))
    return A[:L], B[:L], (startA, startB), start_used


def canon_alpha(a: float, cfg: Config) -> float:
    a = ((a + 90) % 180) - 90
    if abs(a + 90) < 1e-9:
        a = 90.0
    return float(min(cfg.alpha_angles, key=lambda v: abs(v - a)))


def canon_beta(b: float, cfg: Config) -> float:
    b = b % 180.0
    if abs(b - 157.5) < 1e-6:
        return -22.5
    candidates = [22.5, 67.5, 112.5, 157.5]
    b0 = min(candidates, key=lambda v: abs(v - b))
    return -22.5 if abs(b0 - 157.5) < 1e-6 else float(b0)


def run_part_b(cfg: Config, force_reselect_roi: bool) -> None:
    ref_video = os.path.join(cfg.data_dir_b, f"{cfg.reference_bit_b}.mp4")
    roisB = get_rois_with_cache(cfg.roi_cache_b, ref_video, list(cfg.labels_b), cfg, force_reselect_roi)

    append_report(cfg.quality_report_b, [
        "",
        "=================================================",
        f"[{_now()}] Part B run",
        f"data_dir: {cfg.data_dir_b}",
        f"roi_cache: {cfg.roi_cache_b}",
        f"threshold={cfg.threshold}, peak_distance={cfg.peak_distance}, window={cfg.coincidence_window}",
    ])

    counts: dict[tuple[float, float], int] = {}
    fig, axes = plt.subplots(4, 4, figsize=(14, 10), sharey=True)
    for i, a in enumerate(cfg.alpha_angles):
        for j, b in enumerate(cfg.beta_angles):
            bit = ANGLES_TO_BIT_B[(a, b)]
            path = os.path.join(cfg.data_dir_b, f"{bit}.mp4")
            A, B, starts, start_used = read_signals_2roi(path, roisB, cfg)
            nA, nB = normalize_signal(A), normalize_signal(B)
            pA, _ = find_peaks(nA, height=cfg.threshold, distance=cfg.peak_distance)
            pB, _ = find_peaks(nB, height=cfg.threshold, distance=cfg.peak_distance)
            N = coincidences(pA, pB, cfg.coincidence_window)
            counts[(a, b)] = N
            append_report(cfg.quality_report_b, [
                f"bit={bit} alpha={a} beta={b}",
                f"  starts(A,B)={starts}, start_used(min)={start_used}",
                f"  peaks(A,B)=({len(pA)},{len(pB)}), N={N}",
            ])

            ax = axes[i, j]
            ax.plot(nA, alpha=0.6)
            ax.plot(nB, alpha=0.6)
            ax.axhline(cfg.threshold, ls="--", alpha=0.3)
            ax.set_title(f"α={a}, β={b}\nN={N}")
            if i == 3:
                ax.set_xlabel("Frame")
            if j == 0:
                ax.set_ylabel("Norm. Intensity")

    fig.tight_layout()
    fig.savefig("PartB_grid.png", dpi=300)
    if cfg.show_plots:
        plt.show()
    plt.close(fig)

    def getN(a_: float, b_: float) -> int:
        return counts[(canon_alpha(a_, cfg), canon_beta(b_, cfg))]

    def E(a_: float, b_: float) -> float:
        aa, bb = canon_alpha(a_, cfg), canon_beta(b_, cfg)
        ap, bp = canon_alpha(a_ + 90, cfg), canon_beta(b_ + 90, cfg)
        denom = getN(aa, bb) + getN(ap, bp) + getN(aa, bp) + getN(ap, bb)
        if denom == 0:
            return 0.0
        return (getN(aa, bb) + getN(ap, bp) - getN(aa, bp) - getN(ap, bb)) / denom

    S = E(0, 22.5) - E(0, 67.5) + E(45, 22.5) + E(45, 67.5)
    print("\nS =", S)


# Main

def parse_args():
    p = argparse.ArgumentParser(description="QTM analysis (Part A + Part B) with ROI cache + quality reports.")
    p.add_argument("--no-show", action="store_true", help="Disable plt.show (still saves figures).")
    p.add_argument("--only-a", action="store_true", help="Run only Part A.")
    p.add_argument("--only-b", action="store_true", help="Run only Part B.")
    p.add_argument("--reselect-roi-a", action="store_true", help="Force reselect ROI for Part A (ignore cache).")
    p.add_argument("--reselect-roi-b", action="store_true", help="Force reselect ROI for Part B (ignore cache).")
    p.add_argument("--sim-a", action="store_true", help="Run Part A simulation (synthetic signals).")
    p.add_argument("--sim-shots", type=int, default=100, help="Number of simulated shots for --sim-a.")
    p.add_argument("--sim-seed", type=int, default=None, help="Random seed for --sim-a.")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.no_show:
        CFG.show_plots = False

    if args.sim_a:
        run_part_a_simulation(CFG, n_shots=args.sim_shots, seed=args.sim_seed)

    run_a = True
    run_b = True
    if args.only_a and not args.only_b:
        run_b = False
    if args.only_b and not args.only_a:
        run_a = False

    if run_a:
        run_part_a(CFG, force_reselect_roi=args.reselect_roi_a)
    if run_b:
        run_part_b(CFG, force_reselect_roi=args.reselect_roi_b)
