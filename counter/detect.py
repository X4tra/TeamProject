"""
detect.py  —  Vehicle detector + zone counter for Raspberry Pi 4
================================================================

Detection backend
-----------------
Uses YOLOv8n NCNN directly (no ultralytics, no torch required).
Export the model once on your desktop:

    pip install ultralytics
    yolo export model=yolov8n.pt format=ncnn imgsz=320

Then copy the resulting `yolov8n_ncnn_model/` folder next to this script and
pass --model yolov8n_ncnn_model  (the folder, not a .pt file).

Install deps on Pi (no torch needed):
    sudo apt install python3-opencv python3-numpy python3-picamera2
    pip3 install ncnn pillow pyyaml tqdm psutil --break-system-packages

Input sources  (--source)
--------------------------
  picamera              Raspberry Pi camera module via Picamera2  <- DEFAULT
  0                     first USB / CSI camera (OpenCV)
  finalvehicle.mp4      prerecorded file (for offline testing)

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
MOTION_SPEEDUP_FACTOR = 2


# ---------------------------------------------------------------------------
# Model loading  (pure NCNN — no ultralytics, no torch)
# ---------------------------------------------------------------------------

def load_yolo(model_path: str):
    """Load a YOLOv8 NCNN model folder directly."""
    try:
        import ncnn
    except ImportError:
        sys.exit(
            "ERROR: 'ncnn' package not found.\n"
            "Install it with:  pip3 install ncnn --break-system-packages"
        )

    net = ncnn.Net()
    net.opt.use_vulkan_compute = False   # Pi has no Vulkan GPU
    net.opt.num_threads = 4             # use all 4 Pi cores

    param = str(Path(model_path) / "model.ncnn.param")
    bins  = str(Path(model_path) / "model.ncnn.bin")

    if not Path(param).exists():
        sys.exit(f"ERROR: Cannot find {param}\n"
                 f"Make sure the folder '{model_path}' contains "
                 f"model.ncnn.param and model.ncnn.bin")

    net.load_param(param)
    net.load_model(bins)
    print(f"[Model] Loaded NCNN model: {model_path}")
    return net


def detect_yolo(net, image: np.ndarray, conf: float) -> list[dict]:
    """
    Run YOLOv8 NCNN inference on a single BGR frame.

    Returns a list of dicts:
        { 'box': [x1, y1, x2, y2], 'class_id': int, 'score': float }

    Only vehicle classes are returned (VEHICLE_CLASS_IDS).
    """
    import ncnn

    h, w = image.shape[:2]

    # Pre-process: resize + normalise to [0, 1]
    mat_in = ncnn.Mat.from_pixels_resize(
        image,
        ncnn.Mat.PixelType.PIXEL_BGR,
        w, h,
        YOLO_IMGSZ, YOLO_IMGSZ
    )
    mat_in.substract_mean_normalize([0, 0, 0], [1 / 255.0, 1 / 255.0, 1 / 255.0])

    # Inference
    ex = net.create_extractor()
    ex.input("in0", mat_in)
    ret, mat_out = ex.extract("out0")

    if ret != 0 or mat_out is None:
        return []

    # YOLOv8 NCNN output shape: (84, 2100) for imgsz=320
    #   84  = 4 box coords (cx, cy, w, h) + 80 class scores
    #   2100 = 10x10 + 20x20 + 40x40 anchor grid
    out = np.array(mat_out)          # (84, 2100)
    out = out.T                      # (2100, 84)

    boxes_xywh  = out[:, :4]        # cx, cy, w, h  (in YOLO_IMGSZ space)
    class_scores_all = out[:, 4:]   # (2100, 80)

    class_ids    = np.argmax(class_scores_all, axis=1)          # (2100,)
    class_scores = class_scores_all[np.arange(len(out)), class_ids]  # (2100,)

    # Scale factors back to original image size
    scale_x = w / YOLO_IMGSZ
    scale_y = h / YOLO_IMGSZ

    boxes_list, cls_list, score_list = [], [], []

    for i in np.where(class_scores >= conf)[0]:
        cls_id = int(class_ids[i])
        if cls_id not in VEHICLE_CLASS_IDS:
            continue

        cx, cy, bw, bh = boxes_xywh[i]
        x1 = int((cx - bw / 2) * scale_x)
        y1 = int((cy - bh / 2) * scale_y)
        x2 = int((cx + bw / 2) * scale_x)
        y2 = int((cy + bh / 2) * scale_y)

        # Clamp to frame bounds
        x1 = max(0, x1);  y1 = max(0, y1)
        x2 = min(w, x2);  y2 = min(h, y2)

        if x2 <= x1 or y2 <= y1:
            continue

        boxes_list.append([x1, y1, x2, y2])
        cls_list.append(cls_id)
        score_list.append(float(class_scores[i]))

    if not boxes_list:
        return []

    # Non-maximum suppression
    nms_boxes = [[x1, y1, x2 - x1, y2 - y1] for x1, y1, x2, y2 in boxes_list]
    indices   = cv2.dnn.NMSBoxes(nms_boxes, score_list, conf, 0.45)

    if len(indices) == 0:
        return []

    indices = indices.flatten()
    return [
        {
            "box":      boxes_list[i],
            "class_id": cls_list[i],
            "score":    score_list[i],
        }
        for i in indices
    ]


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
    base_skip = args.frame_skip

    # --- FPS bookkeeping ---
    fps         = 0.0
    fps_counter = 0
    fps_timer   = time.time()
    FPS_WINDOW  = 10

    raw_frame_count = 0
    last_annotated  = None

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

        # Motion detection (cheap — runs every frame)
        frame_for_motion = cv2.resize(image, (DISPLAY_W, DISPLAY_H))
        if args.flip:
            frame_for_motion = cv2.flip(frame_for_motion, 1)

        motion = utils.detect_motion(frame_for_motion)

        # Adaptive frame-skip
        cur_skip = max(1, base_skip // MOTION_SPEEDUP_FACTOR) if motion else base_skip

        # Show cached frame on skipped frames
        if raw_frame_count % cur_skip != 0:
            if last_annotated is not None:
                cv2.imshow("Vehicle Detector", last_annotated)
            if cv2.waitKey(1) == 27:
                break
            continue

        image = frame_for_motion

        # --- Detect ---
        detections = detect_yolo(model, image, args.conf)

        # --- Visualize ---
        image = utils.visualize(image, detections)

        # --- FPS overlay ---
        fps_counter += 1
        if fps_counter >= FPS_WINDOW:
            elapsed     = time.time() - fps_timer
            fps         = FPS_WINDOW / elapsed
            fps_timer   = time.time()
            fps_counter = 0

        skip_label = f"skip={cur_skip}" + (" ▲" if motion else "")
        cv2.putText(image, f"FPS {fps:.1f}  {skip_label}", (10, 20),
                    cv2.FONT_HERSHEY_PLAIN, 1.2, (0, 0, 255), 2)

        last_annotated = image

        cv2.imshow("Vehicle Detector", image)
        if cv2.waitKey(1) == 27:
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
        description="Vehicle detector + zone counter (YOLOv8 NCNN, Raspberry Pi)"
    )
    parser.add_argument(
        "--model",
        default="yolov8n_ncnn_model",
        help="Path to NCNN model folder containing model.ncnn.param and model.ncnn.bin. "
             "Default: yolov8n_ncnn_model",
    )
    parser.add_argument(
        "--source",
        default="picamera",
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
        help="Disable horizontal flip. Default: flip is ON for Pi cam.",
    )

    args = parser.parse_args()

    if args.frame_skip < 1:
        parser.error("--frame-skip must be >= 1")

    run(args)


if __name__ == "__main__":
    main()
