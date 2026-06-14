"""
detect.py  —  Vehicle detector + zone counter for Raspberry Pi 4
================================================================

Detection backend
-----------------
Uses YOLOv8n (Ultralytics).  For best Pi 4 performance export the model to
NCNN once on your desktop:

    pip install ultralytics
    yolo export model=yolov8n.pt format=ncnn imgsz=320

Then copy the resulting `yolov8n_ncnn_model/` folder next to this script and
pass --model yolov8n_ncnn_model  (the folder, not a .pt file).
The NCNN backend avoids PyTorch entirely and runs ~2–3× faster on ARM.

Input sources  (--source)
--------------------------
  picamera              Raspberry Pi camera module via Picamera2  ← DEFAULT
  0                     first USB / CSI camera (OpenCV)
  finalvehicle.mp4      prerecorded file (for offline testing)

Motion-adaptive sampling
-------------------------
utils.detect_motion() runs on every raw frame (it is cheap — just a blurred
frame-diff).  When motion is detected the effective frame-skip is halved
(i.e. inference runs twice as often) so fast-moving vehicles are not missed.
The normal frame-skip is restored once the scene is still again.

CSV logging
-----------
Every time a vehicle crosses the counting line its type, object-id, confidence
score, date and exact time (HH:MM:SS) are appended to vehicle_log.csv next to
the script.  Existing rows are never overwritten — the file grows across runs.

Usage examples
--------------
  python detect.py                                   # Pi camera (default)
  python detect.py --source 0                        # USB / CSI cam via OpenCV
  python detect.py --source finalvehicle.mp4         # prerecorded video
  python detect.py --no-flip                         # disable horizontal flip
  python detect.py --model yolov8n_ncnn_model        # NCNN model (faster)
  python detect.py --frame-skip 3 --conf 0.35        # tuning options
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

import utils  # zone-counting, motion detection, drawing, CSV logging

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# COCO class IDs we care about — everything else is ignored
VEHICLE_CLASS_IDS = {2, 3, 5, 7}  # car, motorcycle, bus, truck

# Display resolution — keep small on Pi to reduce drawing overhead
DISPLAY_W, DISPLAY_H = 960, 720

# YOLO inference resolution — 320 is the sweet spot for Pi 4 speed vs accuracy
YOLO_IMGSZ = 320

# Motion-adaptive sampling:
#   When motion is detected, frame_skip is divided by this factor (floored).
#   E.g. base frame-skip=4 → active frame-skip=2 during motion.
#   A value of 2 means "run inference twice as often when there is motion".
MOTION_SPEEDUP_FACTOR = 2


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_yolo(model_path: str):
    """Load a YOLOv8 model (.pt file or NCNN folder)."""
    try:
        from ultralytics import YOLO
    except ImportError:
        sys.exit(
            "ERROR: 'ultralytics' package not found.\n"
            "Install it with:  pip install ultralytics"
        )

    model = YOLO(model_path)
    print(f"[Model] Loaded: {model_path}")
    return model


def detect_yolo(model, image: np.ndarray, conf: float) -> list[dict]:
    """
    Run YOLOv8 inference on a single BGR frame.

    Returns a list of dicts:
        { 'box': [x1, y1, x2, y2], 'class_id': int, 'score': float }

    Only vehicle classes are returned (VEHICLE_CLASS_IDS).
    """
    results = model(image, imgsz=YOLO_IMGSZ, conf=conf, verbose=False)

    detections = []
    for result in results:
        boxes = result.boxes
        if boxes is None:
            continue
        for box in boxes:
            cls_id = int(box.cls[0])
            if cls_id not in VEHICLE_CLASS_IDS:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            score = float(box.conf[0])
            detections.append({
                "box": [x1, y1, x2, y2],
                "class_id": cls_id,
                "score": score,
            })

    return detections


# ---------------------------------------------------------------------------
# Camera / video source helpers
# ---------------------------------------------------------------------------

def open_picamera(width: int, height: int):
    """
    Open the Raspberry Pi camera module via Picamera2.
    Returns a Picamera2 instance (already started).

    Requires:  sudo apt install python3-picamera2
    """
    try:
        from picamera2 import Picamera2  # type: ignore
    except ImportError:
        sys.exit(
            "ERROR: picamera2 not found.\n"
            "Install it with:  sudo apt install python3-picamera2\n"
            "Or use  --source 0  to use the camera via OpenCV instead."
        )

    picam = Picamera2()
    config = picam.create_video_configuration(
        main={"size": (width, height), "format": "BGR888"}
    )
    picam.configure(config)
    picam.start()
    time.sleep(0.5)  # let the sensor warm up
    print("[Camera] Picamera2 started")
    return picam


def read_picamera_frame(picam) -> tuple[bool, np.ndarray]:
    """Capture one frame from a Picamera2 instance."""
    frame = picam.capture_array()
    return True, frame


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run(args):
    # --- Open source ---
    use_picamera = (args.source.lower() == "picamera")

    if use_picamera:
        picam = open_picamera(DISPLAY_W, DISPLAY_H)
        cap   = None
    else:
        source = args.source
        if source.isdigit():
            source = int(source)

        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            sys.exit(f"ERROR: Cannot open source '{args.source}'")

        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  DISPLAY_W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DISPLAY_H)
        picam = None

    # --- Load model ---
    model = load_yolo(args.model)

    # --- Frame-skip state ---
    # base_skip  — the user-supplied (or default) frame-skip value
    # cur_skip   — the effective skip this frame (may be halved during motion)
    base_skip = args.frame_skip

    # --- FPS bookkeeping ---
    fps         = 0.0
    fps_counter = 0
    fps_timer   = time.time()
    FPS_WINDOW  = 10   # average FPS over this many processed frames

    raw_frame_count = 0     # counts every frame read (for frame-skip logic)
    last_annotated  = None  # cache: last fully processed + annotated frame

    print("[Detector] Starting.  Press ESC to quit.")
    print(f"[Detector] Base frame-skip: {base_skip}  "
          f"(halved to {max(1, base_skip // MOTION_SPEEDUP_FACTOR)} when motion detected)")
    print(f"[CSV] Detections will be logged to: {utils.CSV_PATH.resolve()}")

    while True:
        # --- Read frame ---
        if use_picamera:
            success, image = read_picamera_frame(picam)
        else:
            success, image = cap.read()

        if not success or image is None:
            print("[Info] End of stream or read error — exiting.")
            break

        raw_frame_count += 1

        # -------------------------------------------------------------------
        # Motion detection  (runs on EVERY raw frame — it is very cheap)
        # -------------------------------------------------------------------
        # Resize to display size first so the diff is always on the same scale.
        frame_for_motion = cv2.resize(image, (DISPLAY_W, DISPLAY_H))
        if args.flip:
            frame_for_motion = cv2.flip(frame_for_motion, 1)

        motion = utils.detect_motion(frame_for_motion)

        # Adaptive frame-skip: halve the skip interval when motion is detected
        cur_skip = max(1, base_skip // MOTION_SPEEDUP_FACTOR) if motion else base_skip

        # -------------------------------------------------------------------
        # Frame skip — skipped frames show the last annotated result instead
        # of a blank / raw frame so the window stays visually smooth.
        # -------------------------------------------------------------------
        if raw_frame_count % cur_skip != 0:
            if last_annotated is not None:
                cv2.imshow("Vehicle Detector", last_annotated)
            if cv2.waitKey(1) == 27:
                break
            continue

        # -------------------------------------------------------------------
        # Pre-process the frame chosen for inference
        # -------------------------------------------------------------------
        # Use the already-flipped-and-resized frame we prepared above so we
        # don't need to flip/resize twice.
        image = frame_for_motion

        # --- Detect ---
        detections = detect_yolo(model, image, args.conf)

        # --- Visualize (tracking + zone counting + CSV logging inside utils) ---
        image = utils.visualize(image, detections)

        # --- FPS overlay ---
        fps_counter += 1
        if fps_counter >= FPS_WINDOW:
            elapsed     = time.time() - fps_timer
            fps         = FPS_WINDOW / elapsed
            fps_timer   = time.time()
            fps_counter = 0

        # Show effective skip and motion status alongside FPS
        skip_label = f"skip={cur_skip}" + (" ▲" if motion else "")
        cv2.putText(image, f"FPS {fps:.1f}  {skip_label}", (10, 20),
                    cv2.FONT_HERSHEY_PLAIN, 1.2, (0, 0, 255), 2)

        # Cache this frame
        last_annotated = image

        # --- Display ---
        cv2.imshow("Vehicle Detector", image)
        if cv2.waitKey(1) == 27:   # ESC to quit
            break

    # --- Cleanup ---
    if cap is not None:
        cap.release()
    if use_picamera and picam is not None:
        picam.stop()
    cv2.destroyAllWindows()

    print(f"[Done] Session ended.  Log saved to {utils.CSV_PATH.resolve()}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Vehicle detector + zone counter (YOLOv8, Raspberry Pi)"
    )
    parser.add_argument(
        "--model",
        default="yolov8n.pt",
        help="Path to YOLOv8 model (.pt file or NCNN folder). "
             "Default: yolov8n.pt",
    )
    parser.add_argument(
        "--source",
        default="picamera",            # ← Pi camera is now the default
        help="Video source: 'picamera' (default), camera index (0,1,...), "
             "or path to a video file.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.30,
        help="Detection confidence threshold (0–1). Default: 0.30",
    )
    parser.add_argument(
        "--frame-skip",
        type=int,
        default=4,
        help="Run inference on every Nth frame normally. "
             "Halved automatically when motion is detected. Default: 4",
    )
    parser.add_argument(
        "--no-flip",
        dest="flip",
        action="store_false",
        default=True,
        help="Disable horizontal flip (useful for prerecorded footage or a "
             "correctly-mounted camera).  Default: flip is ON for Pi cam.",
    )

    args = parser.parse_args()

    if args.frame_skip < 1:
        parser.error("--frame-skip must be >= 1")

    run(args)


if __name__ == "__main__":
    main()
