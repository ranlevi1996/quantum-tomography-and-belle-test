import cv2
import numpy as np
from pathlib import Path

# =========================
# CONFIG
# =========================

# תיקייה שבה נמצאים הסרטונים
VIDEO_DIR = Path(r"C:\path\to\your\folder")  # <-- שנה לזה שב-VS Code

# ROIs של המסכים העליונים (מתאים לסרטון הדוגמה שלך 1280x720)
# פורמט: x1,x2,y1,y2 כולל קצוות (כמו MATLAB)
ROI_LEFT  = dict(x1=293, x2=571, y1=96,  y2=333)  # Cam left
ROI_RIGHT = dict(x1=652, x2=930, y1=98,  y2=335)  # Cam right

# סיומות וידאו
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}

# =========================
# CORE
# =========================

def crop_inclusive(frame_bgr: np.ndarray, roi: dict) -> np.ndarray:
    """Crop with inclusive bounds (MATLAB-like)."""
    x1, x2, y1, y2 = roi["x1"], roi["x2"], roi["y1"], roi["y2"]
    return frame_bgr[y1:y2 + 1, x1:x2 + 1]  # +1 because Python end-exclusive

def sum_red_channel_matlab_equivalent(roi_bgr: np.ndarray) -> float:
    """
    MATLAB: frame(:,:,1) is Red.
    OpenCV: BGR so Red is channel 2.
    """
    return float(np.sum(roi_bgr[:, :, 2]))

def process_one_video(video_path: Path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open: {video_path}")

    # Accumulators per camera
    left_total = 0.0
    right_total = 0.0
    left_max = 0.0
    right_max = 0.0
    n_frames = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # Crop ROIs
        roi_left = crop_inclusive(frame, ROI_LEFT)
        roi_right = crop_inclusive(frame, ROI_RIGHT)

        # MATLAB-like intensity per frame
        left_val = sum_red_channel_matlab_equivalent(roi_left)
        right_val = sum_red_channel_matlab_equivalent(roi_right)

        # Update stats
        left_total += left_val
        right_total += right_val
        if left_val > left_max:
            left_max = left_val
        if right_val > right_max:
            right_max = right_val

        n_frames += 1

    cap.release()

    return {
        "video": video_path.name,
        "frames": n_frames,
        "left_total_sumR": left_total,
        "right_total_sumR": right_total,
        "left_max_frame_sumR": left_max,
        "right_max_frame_sumR": right_max,
    }

def main():
    if not VIDEO_DIR.exists():
        raise FileNotFoundError(f"Folder not found: {VIDEO_DIR}")

    videos = sorted([p for p in VIDEO_DIR.iterdir() if p.suffix.lower() in VIDEO_EXTS])
    if not videos:
        print(f"No videos found in: {VIDEO_DIR}")
        return

    print(f"Found {len(videos)} videos in {VIDEO_DIR}\n")

    for vp in videos:
        try:
            res = process_one_video(vp)

            # הדפסה “שם + עוצמה ימין + עוצמה שמאל”
            # כאן אני מדפיס גם TOTAL וגם MAX כדי שיהיה לך גם אנרגיה כוללת וגם פיק.
            print(
                f"{res['video']}\t"
                f"LEFT total={res['left_total_sumR']:.0f}, max={res['left_max_frame_sumR']:.0f}\t"
                f"RIGHT total={res['right_total_sumR']:.0f}, max={res['right_max_frame_sumR']:.0f}\t"
                f"(frames={res['frames']})"
            )

        except Exception as e:
            print(f"{vp.name}\tERROR: {e}")

if __name__ == "__main__":
    main()
