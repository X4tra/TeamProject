import cv2
import csv
import numpy as np
from datetime import datetime
from pathlib import Path
from tracker import Tracker

# ---------------------------------------------------------------------------
# Tracker
# ---------------------------------------------------------------------------
tracker = Tracker(iou_threshold=0.25, max_age=5, min_hits=2)

# ---------------------------------------------------------------------------
# Tripwire configuration
#
# LINE_X      — horizontal pixel position of the vertical counting line.
#               Set this to somewhere vehicles are clearly visible mid-frame.
#               Default: 480  (centre of a 960-wide frame)
#
# CROSS_MARGIN — how many pixels PAST the line the centroid must travel before
#               the crossing is confirmed. Prevents false triggers from
#               vehicles that briefly touch the line then back away, or from
#               detection jitter. A value of 10–20 px works well.
# ---------------------------------------------------------------------------
LINE_X       = 480
CROSS_MARGIN = 15     # pixels past the line required to confirm a crossing

LINE_START = (LINE_X, 0)
LINE_END   = (LINE_X, 720)   # matches DISPLAY_H in detect.py

# ---------------------------------------------------------------------------
# Per-vehicle crossing state
# ---------------------------------------------------------------------------
counted_ids:  set[int]        = set()   # IDs already counted — never revisited
origin_x:     dict[int, int]  = {}      # {obj_id: cx when first seen}

_prev_total: int = -1

# COCO vehicle class names
VEHICLE_CLASS_NAMES = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

# Colours (BGR)
COL_LINE    = (0, 255, 255)   # cyan
COL_BOX     = (50, 220, 50)   # green  — not yet counted
COL_COUNTED = (80, 80, 255)   # red    — just crossed (flashes one frame)
COL_LABEL   = (255, 255, 0)   # yellow
COL_DOT     = (255, 0, 255)   # magenta centroid
COL_BANNER  = (20, 20, 20)    # banner background
COL_MOTION  = (0, 165, 255)   # orange — motion indicator

# ---------------------------------------------------------------------------
# CSV logging
# ---------------------------------------------------------------------------
CSV_PATH = Path("vehicle_log.csv")
CSV_FIELDNAMES = ["date", "time", "vehicle_type", "object_id", "confidence"]

def _ensure_csv() -> None:
    """Create the CSV file with a header row if it does not already exist."""
    if not CSV_PATH.exists():
        with open(CSV_PATH, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
            writer.writeheader()
        print(f"[CSV] Created log file: {CSV_PATH.resolve()}")

def _log_vehicle(obj_id: int, class_id: int, score: float) -> None:
    """Append one detection event row to the CSV file."""
    now = datetime.now()
    vehicle_type = VEHICLE_CLASS_NAMES.get(class_id, "vehicle")
    row = {
        "date":         now.strftime("%Y-%m-%d"),
        "time":         now.strftime("%H:%M:%S"),
        "vehicle_type": vehicle_type,
        "object_id":    obj_id,
        "confidence":   f"{score:.2f}",
    }
    with open(CSV_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        writer.writerow(row)
    print(f"[CSV] Logged: {row['date']} {row['time']}  {vehicle_type}  "
          f"id={obj_id}  conf={score:.2f}")

# ---------------------------------------------------------------------------
# Motion detection
# ---------------------------------------------------------------------------
# The motion detector keeps a running background model (simple frame-diff).
# When significant pixel change is detected the caller is told to double the
# sampling rate (i.e. halve frame-skip) so no vehicle is missed during
# movement bursts.
#
# MOTION_THRESH   — minimum area (px²) of a change contour to count as motion.
# MOTION_BLUR_K   — Gaussian blur kernel applied before diffing (odd number).
# ---------------------------------------------------------------------------
MOTION_THRESH = 1500   # px² — tune up to reduce noise, down to catch slow cars
MOTION_BLUR_K = 21

_prev_gray: np.ndarray | None = None   # previous frame (grayscale) for diffing
motion_detected: bool = False          # current-frame motion flag (read by detect.py)


def detect_motion(frame: np.ndarray) -> bool:
    """
    Compare *frame* against the previous frame using a simple absolute-diff
    approach. Returns True if meaningful pixel motion is found.

    Side-effect: updates the module-level `motion_detected` flag so that
    detect.py can read it without re-calling this function.
    """
    global _prev_gray, motion_detected

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (MOTION_BLUR_K, MOTION_BLUR_K), 0)

    if _prev_gray is None:
        _prev_gray = gray
        motion_detected = False
        return False

    diff  = cv2.absdiff(_prev_gray, gray)
    _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
    dilated   = cv2.dilate(thresh, None, iterations=2)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    motion_detected = any(cv2.contourArea(c) >= MOTION_THRESH
                          for c in contours)
    _prev_gray = gray
    return motion_detected


# ---------------------------------------------------------------------------
# Banner / HUD helpers
# ---------------------------------------------------------------------------

def _draw_banner(image: np.ndarray, total: int, motion: bool) -> None:
    h, w = image.shape[:2]
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (w, 48), COL_BANNER, -1)
    cv2.addWeighted(overlay, 0.6, image, 0.4, 0, image)

    text = f"VEHICLES COUNTED: {total}"
    (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 1.0, 2)
    cv2.putText(image, text, ((w - tw) // 2, 34),
                cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 2)

    # Small motion-status badge in the top-right corner of the banner
    badge_text = "MOTION" if motion else "STILL"
    badge_col  = COL_MOTION if motion else (100, 100, 100)
    (bw, _), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_PLAIN, 1.2, 1)
    cv2.putText(image, badge_text, (w - bw - 10, 30),
                cv2.FONT_HERSHEY_PLAIN, 1.2, badge_col, 2)


# ---------------------------------------------------------------------------
# Crossing logic
# ---------------------------------------------------------------------------

def _check_crossing(obj_id: int, cx: int) -> bool:
    """
    Return True the first time obj_id's centroid has travelled more than
    CROSS_MARGIN pixels past LINE_X from its origin side.
    """
    if obj_id in counted_ids:
        return False

    if obj_id not in origin_x:
        origin_x[obj_id] = cx
        return False

    ox = origin_x[obj_id]

    if ox < LINE_X and cx > LINE_X + CROSS_MARGIN:
        return True
    if ox > LINE_X and cx < LINE_X - CROSS_MARGIN:
        return True

    return False


# ---------------------------------------------------------------------------
# Public API called by detect.py
# ---------------------------------------------------------------------------

# Initialise CSV on module load
_ensure_csv()


def visualize(image: np.ndarray, detections: list[dict]) -> np.ndarray:
    global _prev_total

    boxes   = [det["box"] for det in detections]
    tracked = tracker.update(boxes)

    det_lookup: dict[tuple, dict] = {
        (d["box"][0], d["box"][1], d["box"][2], d["box"][3]): d
        for d in detections
    }

    newly_counted: set[int] = set()

    for x1, y1, x2, y2, obj_id in tracked:
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        if _check_crossing(obj_id, cx):
            counted_ids.add(obj_id)
            newly_counted.add(obj_id)

            # --- CSV log: find the matching detection for class/score ---
            det = det_lookup.get((x1, y1, x2, y2))
            if det:
                _log_vehicle(obj_id, det["class_id"], det["score"])
            else:
                # Tracker box may not align exactly; log with defaults
                _log_vehicle(obj_id, 2, 0.0)   # class 2 = car, score unknown

        # Draw bounding box
        box_col = COL_COUNTED if obj_id in newly_counted else COL_BOX
        cv2.rectangle(image, (x1, y1), (x2, y2), box_col, 2)
        cv2.circle(image, (cx, cy), 5, COL_DOT, -1)

        det = det_lookup.get((x1, y1, x2, y2))
        if det:
            cls   = VEHICLE_CLASS_NAMES.get(det["class_id"], "vehicle")
            label = f"#{obj_id} {cls} {det['score']:.2f}"
        else:
            label = f"#{obj_id}"

        cv2.putText(image, label, (x1, max(y1 - 5, 54)),
                    cv2.FONT_HERSHEY_PLAIN, 1.1, COL_LABEL, 1)

    # Tripwire
    cv2.line(image, LINE_START, LINE_END, COL_LINE, 2)
    cv2.putText(image, "counting line", (LINE_X + 6, 70),
                cv2.FONT_HERSHEY_PLAIN, 1.0, COL_LINE, 1)

    # Banner (passes current motion flag so it can show the badge)
    total = len(counted_ids)
    _draw_banner(image, total, motion_detected)

    if total != _prev_total:
        print(f"[Count] Total vehicles: {total}")
        _prev_total = total

    return image
