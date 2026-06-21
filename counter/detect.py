"""
detect.py  —  Vehicle detector + zone counter for Raspberry Pi 4
================================================================

No ultralytics, no torch required. Uses NCNN directly.

Install deps on Pi:
    sudo apt install python3-opencv python3-numpy python3-picamera2
    pip3 install ncnn --break-system-packages

Usage examples
--------------
  python3 detect.py                                   # Pi camera (default)
  python3 detect.py --source 0                        # USB / CSI cam via OpenCV
  python3 detect.py --source finalvehicle.mp4         # prerecorded video
  python3 detect.py --no-flip                         # disable horizontal flip
  python3 detect.py --model yolov8n_ncnn_model        # NCNN model folder
  python3 detect.py --frame-skip 3 --conf 0.35        # tuning options
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
VEHICLE_CLASS_IDS       = {2, 3, 5, 7}   # car, motorcycle, bus, truck
DISPLAY_W, DISPLAY_H   = 960, 720
YOLO_IMGSZ             = 320
MOTION_SPEEDUP_FACTOR  = 2


# ---------------------------------------------------------------------------
# .param parser — auto-detects input / output blob names
# ---------------------------------------------------------------------------

def _parse_ncnn_layer_names(param_path: str) -> tuple[str, str]:
    """
    Read a .param file and return (input_blob_name, output_blob_name).

    NCNN .param line format:
        LayerType  LayerName  num_inputs  num_outputs  [blobs...]  [key=val...]

    The Input layer's output blob  = the network's input name.
    The last layer's output blob   = the network's output name.
    """
    input_name  = "in0"    # safe fallback
    output_name = "out0"   # safe fallback

    try:
        with open(param_path, "r") as f:
            lines = f.readlines()

        for line in lines[2:]:          # skip magic + count lines
            parts = line.strip().split()
            if len(parts) < 5:
                continue

            layer_type = parts[0]
            try:
                num_in  = int(parts[2])
                num_out = int(parts[3])
            except ValueError:
                continue

            # Collect blob tokens (stop at first key=value param)
            blobs = []
            for tok in parts[4:]:
                if "=" in tok:
                    break
                blobs.append(tok)

            in_blobs  = blobs[:num_in]
            out_blobs = blobs[num_in: num_in + num_out]

            if layer_type == "Input" and out_blobs:
                input_name = out_blobs[0]

            if out_blobs:
                output_name = out_blobs[0]   # keep updating → ends on last layer

    except Exception as e:
        print(f"[Warning] Could not parse {param_path}: {e}  — using defaults")

    return input_name, output_name


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

    param = str(Path(model_path) / "model.ncnn.param")
    bins  = str(Path(model_path) / "model.ncnn.bin")

    if not Path(param).exists():
        sys.exit(
            f"ERROR: Cannot find {param}\n"
            f"Make sure '{model_path}' contains model.ncnn.param and model.ncnn.bin"
        )

    # Auto-detect layer names from the .param file
    input_name, output_name = _parse_ncnn_layer_names(param)
    print(f"[Model] NCNN layer names — input: '{input_name}'  output: '{output_name}'")

    net = ncnn.Net()
    net.opt.use_vulkan_compute = False   # Pi has no Vulkan GPU
    net.opt.num_threads        = 4       # use all 4 Pi cores
    net.load_param(param)
    net.load_model(bins)
    print(f"[Model] Loaded: {model_path}")

    # Return as dict — ncnn.Net doesn't support custom attributes
    return {"net": net, "input": input_name, "output": output_name}


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def detect_yolo(model, image: np.ndarray, conf: float) -> list[dict]:
    """
    Run YOLOv8 NCNN inference on a single BGR frame.

    Returns a list of dicts:
        { 'box': [x1, y1, x2, y2], 'class_id': int, 'score': float }
    Only VEHICLE_CLASS_IDS are returned.
    """
    import ncnn

    # Unpack model dict — ncnn.Net doesn't support custom attributes
    ncnn_net    = model["net"]
    input_name  = model["input"]
    output_name = model["output"]

    h, w = image.shape[:2]

    # Pre-process: resize + normalise to [0, 1]
    mat_in = ncnn.Mat.from_pixels_resize(
        image,
        ncnn.Mat.PixelType.PIXEL_BGR,
        w, h,
        YOLO_IMGSZ, YOLO_IMGSZ,
    )
    mat_in.substract_mean_normalize(
        [0, 0, 0],
        [1 / 255.0, 1 / 255.0, 1 / 255.0],
    )

    # Inference
    ex = ncnn_net.create_extractor()
    ex.input(input_name, mat_in)
    ret, mat_out = ex.extract(output_name)

    if ret != 0 or mat_out is None:
        return []

    # YOLOv8 NCNN output: (84, 2100) for imgsz=320
    #   84   = 4 box coords (cx, cy, w, h in YOLO_IMGSZ space) + 80 class scores
    #   2100 = 10×10 + 20×20 + 40×40 anchor grid
    out = np.array(mat_out).T          # → (2100, 84)

    boxes_xywh       = out[:, :4]      # cx, cy, w, h
    class_scores_all = out[:, 4:]      # (2100, 80)

    class_ids    = np.argmax(class_scores_all, axis=1)
    class_scores = class_scores_all[np.arange(len(out)), class_ids]

    scale_x = w / YOLO_IMGSZ
    scale_y = h / YOLO_IMGSZ

    boxes_list, cls_list, score_list = [], [], []

    for i in np.where(class_scores >= conf)[0]:
        cls_id = int(class_ids[i])
        if cls_id not in VEHICLE_CLASS_IDS:
            continue

        cx, cy, bw, bh = boxes_xywh[i]
        x1 = max(0, int((cx - bw / 2) * scale_x))
        y1 = max(0, int((cy - bh / 2) * scale_y))
        x2 = min(w,  int((cx + bw / 2) * scale_x))
        y2 = min(h,  int((cy + bh / 2) * scale_y))

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

    return [
        {"box": boxes_list[i], "class_id": cls_list[i], "score": score_list[i]}
        for i in indices.flatten()
    ]


# ---------------------------------------------------------------------------
# Camera helpers
# ---------------------------------------------------------------------------

def open_picamera(width: int, height: int):
    try:
        from picamera2 import Picamera2
    except ImportError:
        sys.exit(
            "ERROR: picamera2 not found.\n"
            "Install:  sudo apt install python3-picamera2\n"
            "Or use --source 0 to use the camera via OpenCV instead."
        )
    picam  = Picamera2()
    config = picam.create_video_configuration(
        main={"size": (width, height), "format": "BGR888"}
    )
    picam.configure(config)
    picam.start()
    time.sleep(0.5)
    print("[Camera] Picamera2 started")
    return picam


def read_picamera_frame(picam) -> tuple[bool, np.ndarray]:
    return True, picam.capture_array()


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run(args):
    use_picamera = args.source.lower() == "picamera"

    if use_picamera:
        picam = open_picamera(DISPLAY_W, DISPLAY_H)
        cap   = None
    else:
        source = int(args.source) if args.source.isdigit() else args.source
        cap    = cv2.VideoCapture(source)
        if not cap.isOpened():
            sys.exit(f"ERROR: Cannot open source '{args.source}'")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  DISPLAY_W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DISPLAY_H)
        picam = None

    model = load_yolo(args.model)

    base_skip   = args.frame_skip
    fps         = 0.0
    fps_counter = 0
    fps_timer   = time.time()
    FPS_WINDOW  = 10

    raw_frame_count = 0
    last_annotated  = None

    print("[Detector] Starting.  Press ESC to quit.")
    print(f"[CSV] Logging to: {utils.CSV_PATH.resolve()}")

    while True:
        if use_picamera:
            success, image = read_picamera_frame(picam)
        else:
            success, image = cap.read()

        if not success or image is None:
            print("[Info] End of stream — exiting.")
            break

        raw_frame_count += 1

        frame_disp = cv2.resize(image, (DISPLAY_W, DISPLAY_H))
        if args.flip:
            frame_disp = cv2.flip(frame_disp, 1)

        motion   = utils.detect_motion(frame_disp)
        cur_skip = max(1, base_skip // MOTION_SPEEDUP_FACTOR) if motion else base_skip

        if raw_frame_count % cur_skip != 0:
            if last_annotated is not None:
                cv2.imshow("Vehicle Detector", last_annotated)
            if cv2.waitKey(1) == 27:
                break
            continue

        image      = frame_disp
        detections = detect_yolo(model, image, args.conf)
        image      = utils.visualize(image, detections)

        fps_counter += 1
        if fps_counter >= FPS_WINDOW:
            fps         = FPS_WINDOW / (time.time() - fps_timer)
            fps_timer   = time.time()
            fps_counter = 0

        skip_label = f"skip={cur_skip}" + (" ▲" if motion else "")
        cv2.putText(image, f"FPS {fps:.1f}  {skip_label}",
                    (10, 20), cv2.FONT_HERSHEY_PLAIN, 1.2, (0, 0, 255), 2)

        last_annotated = image
        cv2.imshow("Vehicle Detector", image)
        if cv2.waitKey(1) == 27:
            break

    if cap is not None:
        cap.release()
    if use_picamera and picam is not None:
        picam.stop()
    cv2.destroyAllWindows()
    print(f"[Done] Log saved to {utils.CSV_PATH.resolve()}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Vehicle detector + zone counter (YOLOv8 NCNN, no torch)"
    )
    parser.add_argument("--model",      default="yolov8n_ncnn_model",
                        help="NCNN model folder. Default: yolov8n_ncnn_model")
    parser.add_argument("--source",     default="picamera",
                        help="'picamera', camera index, or video path. Default: picamera")
    parser.add_argument("--conf",       type=float, default=0.30,
                        help="Confidence threshold. Default: 0.30")
    parser.add_argument("--frame-skip", type=int,   default=4,
                        help="Inference every N frames. Default: 4")
    parser.add_argument("--no-flip",    dest="flip", action="store_false", default=True,
                        help="Disable horizontal flip.")

    args = parser.parse_args()
    if args.frame_skip < 1:
        parser.error("--frame-skip must be >= 1")
    run(args)


if __name__ == "__main__":
    main()
