#!/usr/bin/env python3
"""
vehicle_counter_pi.py — lightweight vehicle detector + line counter for Raspberry Pi
=====================================================================================

What this version does
----------------------
- Uses Picamera2 by default on Raspberry Pi.
- Uses pygame for display instead of cv2.imshow().
- Keeps inference lightweight by defaulting to an exported NCNN model folder.
- Counts vehicles crossing a vertical line.
- Logs each counted vehicle to CSV.
- Has a small built-in tracker so you do not need tracker.py.

Recommended model path
----------------------
Export a nano YOLO model to NCNN on your desktop and copy the folder here.
Examples:
    yolo26n_ncnn_model/
    yolo11n_ncnn_model/
    yolov8n_ncnn_model/

On a desktop with Ultralytics installed:
    yolo export model=yolo26n.pt format=ncnn imgsz=320
or:
    yolo export model=yolo11n.pt format=ncnn imgsz=320

Install on Raspberry Pi
-----------------------
    sudo apt update
    sudo apt install python3-opencv python3-numpy python3-picamera2 python3-pygame
    pip3 install ncnn --break-system-packages

Usage
-----
    python3 vehicle_counter_pi.py
    python3 vehicle_counter_pi.py --source 0
    python3 vehicle_counter_pi.py --source video.mp4
    python3 vehicle_counter_pi.py --model yolo11n_ncnn_model
    python3 vehicle_counter_pi.py --no-fullscreen
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

try:
    import pygame
except ImportError:
    pygame = None


# =============================================================================
# Configuration
# =============================================================================

VEHICLE_CLASS_NAMES = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}
VEHICLE_CLASS_IDS = set(VEHICLE_CLASS_NAMES.keys())

DEFAULT_DISPLAY_W = 640
DEFAULT_DISPLAY_H = 480
YOLO_IMGSZ = 320

WINDOW_NAME = "Vehicle Counter"

# CSV
CSV_PATH = Path("vehicle_log.csv")
CSV_FIELDNAMES = ["date", "time", "vehicle_type", "object_id", "confidence", "direction"]

# Detection / tracking
NMS_IOU_THRESH = 0.45
TRACK_IOU_THRESH = 0.25
TRACK_MAX_AGE = 8
TRACK_MIN_HITS = 2
CROSS_MARGIN = 15

# Motion-adaptive inference
MOTION_THRESH = 1500
MOTION_BLUR_K = 21
MOTION_SPEEDUP_FACTOR = 2

# Model defaults
DEFAULT_MODEL = "yolo26n_ncnn_model"


# =============================================================================
# Utility helpers
# =============================================================================

def ensure_csv() -> None:
    if not CSV_PATH.exists():
        with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
            writer.writeheader()
        print(f"[CSV] Created {CSV_PATH.resolve()}")


def log_vehicle(obj_id: int, class_id: int, score: float, direction: str) -> None:
    now = datetime.now()
    row = {
        "date": now.strftime("%Y-%m-%d"),
        "time": now.strftime("%H:%M:%S"),
        "vehicle_type": VEHICLE_CLASS_NAMES.get(class_id, "vehicle"),
        "object_id": obj_id,
        "confidence": f"{score:.2f}",
        "direction": direction,
    }
    with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        writer.writerow(row)
    print(f"[CSV] {row['date']} {row['time']}  {row['vehicle_type']}  id={obj_id}  dir={direction}  conf={score:.2f}")


def make_status_frame(width: int, height: int, lines: list[str]) -> np.ndarray:
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    y = 40
    for line in lines:
        cv2.putText(frame, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
        y += 30
    return frame


def clamp_box(x1: int, y1: int, x2: int, y2: int, w: int, h: int) -> list[int]:
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(0, min(x2, w))
    y2 = max(0, min(y2, h))
    return [x1, y1, x2, y2]


def box_iou(a: list[int], b: list[int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    inter_w = max(0, ix2 - ix1)
    inter_h = max(0, iy2 - iy1)
    inter = inter_w * inter_h
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def detect_motion(frame: np.ndarray, prev_gray: Optional[np.ndarray]) -> tuple[bool, np.ndarray]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (MOTION_BLUR_K, MOTION_BLUR_K), 0)

    if prev_gray is None:
        return False, gray

    diff = cv2.absdiff(prev_gray, gray)
    _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(thresh, None, iterations=2)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    motion = any(cv2.contourArea(c) >= MOTION_THRESH for c in contours)
    return motion, gray


# =============================================================================
# Pygame display
# =============================================================================

class PygameDisplay:
    def __init__(self, fullscreen: bool, logical_size: tuple[int, int]):
        if pygame is None:
            raise RuntimeError(
                "pygame is not installed. Install it with:\n"
                "  sudo apt install python3-pygame\n"
                "or\n"
                "  pip3 install pygame"
            )

        os.environ.setdefault("SDL_VIDEO_CENTERED", "1")
        pygame.init()

        self.logical_w, self.logical_h = logical_size
        if fullscreen:
            info = pygame.display.Info()
            self.window_w = info.current_w or self.logical_w
            self.window_h = info.current_h or self.logical_h
            flags = pygame.FULLSCREEN
        else:
            self.window_w = self.logical_w
            self.window_h = self.logical_h
            flags = 0

        self.screen = pygame.display.set_mode((self.window_w, self.window_h), flags)
        pygame.display.set_caption(WINDOW_NAME)
        self.clock = pygame.time.Clock()

    def show(self, frame_bgr: np.ndarray) -> None:
        if frame_bgr is None:
            return

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        surf = pygame.surfarray.make_surface(np.ascontiguousarray(rgb.swapaxes(0, 1)))

        if surf.get_width() != self.window_w or surf.get_height() != self.window_h:
            surf = pygame.transform.scale(surf, (self.window_w, self.window_h))

        self.screen.blit(surf, (0, 0))
        pygame.display.flip()

    def pump(self) -> bool:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                return False
        return True

    def tick(self, fps: int = 60) -> None:
        self.clock.tick(fps)

    def close(self) -> None:
        pygame.quit()


# =============================================================================
# Camera / video source
# =============================================================================

def open_picamera(width: int, height: int):
    try:
        from picamera2 import Picamera2
    except ImportError as e:
        raise RuntimeError(
            "picamera2 is not installed. Install it with:\n"
            "  sudo apt install python3-picamera2"
        ) from e

    picam = Picamera2()
    config = picam.create_video_configuration(
        main={"size": (width, height), "format": "BGR888"}
    )
    picam.configure(config)
    picam.start()

    # Warm-up: avoid black / empty frames on startup.
    deadline = time.time() + 3.0
    got_frame = False
    while time.time() < deadline:
        try:
            frame = picam.capture_array()
            if frame is not None and frame.size > 0:
                got_frame = True
                break
        except Exception:
            pass
        time.sleep(0.05)

    print("[Camera] Picamera2 started" + (" (warm-up OK)" if got_frame else " (warming up)"))
    return picam


def read_picamera_frame(picam):
    try:
        frame = picam.capture_array()
        if frame is None or frame.size == 0:
            return False, None
        return True, frame
    except Exception as e:
        print(f"[Camera] Picamera2 read error: {e}")
        return False, None


# =============================================================================
# Detector backends
# =============================================================================

def parse_ncnn_layer_names(param_path: str) -> tuple[str, str]:
    input_name = "in0"
    output_name = "out0"
    try:
        with open(param_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        for line in lines[2:]:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            layer_type = parts[0]
            try:
                num_in = int(parts[2])
                num_out = int(parts[3])
            except ValueError:
                continue

            blobs = []
            for tok in parts[4:]:
                if "=" in tok:
                    break
                blobs.append(tok)

            in_blobs = blobs[:num_in]
            out_blobs = blobs[num_in:num_in + num_out]
            if layer_type == "Input" and out_blobs:
                input_name = out_blobs[0]
            if out_blobs:
                output_name = out_blobs[0]
    except Exception as e:
        print(f"[Model] Warning: could not parse {param_path}: {e}")
    return input_name, output_name


class NCNNDetector:
    def __init__(self, model_dir: str):
        try:
            import ncnn
        except ImportError as e:
            raise RuntimeError(
                "ncnn is not installed. Install it with:\n"
                "  pip3 install ncnn --break-system-packages"
            ) from e

        self.ncnn = ncnn
        model_path = Path(model_dir)
        self.param = model_path / "model.ncnn.param"
        self.bin = model_path / "model.ncnn.bin"

        if not self.param.exists():
            raise FileNotFoundError(f"Missing NCNN param file: {self.param}")
        if not self.bin.exists():
            raise FileNotFoundError(f"Missing NCNN bin file: {self.bin}")

        self.input_name, self.output_name = parse_ncnn_layer_names(str(self.param))

        self.net = ncnn.Net()
        self.net.opt.use_vulkan_compute = False
        self.net.opt.num_threads = max(1, min(4, os.cpu_count() or 1))
        self.net.load_param(str(self.param))
        self.net.load_model(str(self.bin))

        print(f"[Model] Loaded NCNN folder: {model_dir}")
        print(f"[Model] Input: '{self.input_name}'  Output: '{self.output_name}'")

    def detect(self, image_bgr: np.ndarray, conf: float) -> list[dict]:
        ncnn = self.ncnn
        h, w = image_bgr.shape[:2]

        mat_in = ncnn.Mat.from_pixels_resize(
            image_bgr, ncnn.Mat.PixelType.PIXEL_BGR,
            w, h, YOLO_IMGSZ, YOLO_IMGSZ
        )
        mat_in.substract_mean_normalize([0, 0, 0], [1 / 255.0, 1 / 255.0, 1 / 255.0])

        ex = self.net.create_extractor()
        ex.input(self.input_name, mat_in)
        ret, mat_out = ex.extract(self.output_name)
        if ret != 0 or mat_out is None:
            return []

        out = np.array(mat_out)
        if out.ndim != 2:
            out = np.array(out).reshape(-1, out.shape[-1] if out.ndim else 0)
        if out.shape[0] < out.shape[1]:
            out = out.T

        if out.shape[1] < 6:
            return []

        boxes_xywh = out[:, :4]
        class_scores_all = out[:, 4:]

        class_ids = np.argmax(class_scores_all, axis=1)
        class_scores = class_scores_all[np.arange(len(out)), class_ids]

        scale_x = w / YOLO_IMGSZ
        scale_y = h / YOLO_IMGSZ

        boxes, classes, scores = [], [], []
        for i in np.where(class_scores >= conf)[0]:
            cls_id = int(class_ids[i])
            if cls_id not in VEHICLE_CLASS_IDS:
                continue

            cx, cy, bw, bh = boxes_xywh[i]
            x1 = int((cx - bw / 2) * scale_x)
            y1 = int((cy - bh / 2) * scale_y)
            x2 = int((cx + bw / 2) * scale_x)
            y2 = int((cy + bh / 2) * scale_y)

            x1, y1, x2, y2 = clamp_box(x1, y1, x2, y2, w, h)
            if x2 <= x1 or y2 <= y1:
                continue

            boxes.append([x1, y1, x2, y2])
            classes.append(cls_id)
            scores.append(float(class_scores[i]))

        if not boxes:
            return []

        nms_boxes = [[x1, y1, x2 - x1, y2 - y1] for x1, y1, x2, y2 in boxes]
        indices = cv2.dnn.NMSBoxes(nms_boxes, scores, conf, NMS_IOU_THRESH)
        if len(indices) == 0:
            return []

        keep = []
        for idx in indices.flatten():
            keep.append({
                "box": boxes[idx],
                "class_id": classes[idx],
                "score": scores[idx],
            })
        return keep


class UltralyticsDetector:
    def __init__(self, model_path: str):
        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise RuntimeError(
                "ultralytics is not installed. Install it with:\n"
                "  pip3 install ultralytics"
            ) from e

        self.YOLO = YOLO
        self.model = YOLO(model_path)
        print(f"[Model] Loaded Ultralytics model: {model_path}")

    def detect(self, image_bgr: np.ndarray, conf: float) -> list[dict]:
        results = self.model(image_bgr, imgsz=YOLO_IMGSZ, conf=conf, verbose=False)
        detections: list[dict] = []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                cls_id = int(box.cls[0])
                if cls_id not in VEHICLE_CLASS_IDS:
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                detections.append({
                    "box": [x1, y1, x2, y2],
                    "class_id": cls_id,
                    "score": float(box.conf[0]),
                })
        return detections


def load_detector(model_path: str):
    p = Path(model_path)
    if p.is_dir():
        param = p / "model.ncnn.param"
        binf = p / "model.ncnn.bin"
        if param.exists() and binf.exists():
            return NCNNDetector(model_path)

    if p.suffix.lower() == ".pt":
        return UltralyticsDetector(model_path)

    # Default to NCNN folder if present, otherwise try Ultralytics on whatever path was given.
    if p.exists():
        try:
            return NCNNDetector(model_path)
        except Exception:
            return UltralyticsDetector(model_path)

    # If the path does not exist, be explicit about the expected NCNN folder.
    raise FileNotFoundError(
        f"Model path '{model_path}' was not found.\n"
        f"Expected an NCNN folder like '{DEFAULT_MODEL}/', or a .pt file if using ultralytics."
    )


# =============================================================================
# Tiny tracker + line counting
# =============================================================================

@dataclass
class Track:
    track_id: int
    box: list[int]
    class_id: int
    score: float
    hits: int = 1
    age: int = 0
    missed: int = 0
    first_cx: Optional[int] = None
    counted: bool = False

    @property
    def centroid(self) -> tuple[int, int]:
        x1, y1, x2, y2 = self.box
        return ((x1 + x2) // 2, (y1 + y2) // 2)


class IoUTracker:
    def __init__(self, iou_threshold: float = TRACK_IOU_THRESH, max_age: int = TRACK_MAX_AGE):
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.tracks: list[Track] = []
        self.next_id = 1

    def update(self, detections: list[dict]) -> list[Track]:
        for tr in self.tracks:
            tr.age += 1

        det_boxes = [d["box"] for d in detections]
        det_class = [d["class_id"] for d in detections]
        det_score = [d["score"] for d in detections]

        matched_track_idxs: set[int] = set()
        matched_det_idxs: set[int] = set()

        # Greedy best-IoU assignment.
        candidates = []
        for ti, tr in enumerate(self.tracks):
            for di, db in enumerate(det_boxes):
                iou = box_iou(tr.box, db)
                if iou >= self.iou_threshold:
                    candidates.append((iou, ti, di))
        candidates.sort(reverse=True)

        for iou, ti, di in candidates:
            if ti in matched_track_idxs or di in matched_det_idxs:
                continue
            matched_track_idxs.add(ti)
            matched_det_idxs.add(di)

            tr = self.tracks[ti]
            tr.box = det_boxes[di]
            tr.class_id = det_class[di]
            tr.score = det_score[di]
            tr.hits += 1
            tr.missed = 0

        # Unmatched tracks age out.
        alive_tracks = []
        for idx, tr in enumerate(self.tracks):
            if idx not in matched_track_idxs:
                tr.missed += 1
            if tr.missed <= self.max_age:
                alive_tracks.append(tr)
        self.tracks = alive_tracks

        # New tracks for unmatched detections.
        for di, db in enumerate(det_boxes):
            if di in matched_det_idxs:
                continue
            self.tracks.append(
                Track(
                    track_id=self.next_id,
                    box=db,
                    class_id=det_class[di],
                    score=det_score[di],
                )
            )
            self.next_id += 1

        # Return only mature tracks.
        return [tr for tr in self.tracks if tr.hits >= TRACK_MIN_HITS or tr.missed == 0]


def count_crossing(track: Track, line_x: int, margin: int) -> Optional[str]:
    cx, _ = track.centroid

    if track.first_cx is None:
        track.first_cx = cx
        return None

    if track.counted:
        return None

    if track.first_cx < line_x and cx > line_x + margin:
        track.counted = True
        return "left->right"
    if track.first_cx > line_x and cx < line_x - margin:
        track.counted = True
        return "right->left"

    return None


# =============================================================================
# Rendering
# =============================================================================

COL_LINE = (0, 255, 255)
COL_BOX = (50, 220, 50)
COL_COUNTED = (80, 80, 255)
COL_LABEL = (255, 255, 0)
COL_DOT = (255, 0, 255)
COL_BANNER = (20, 20, 20)
COL_MOTION = (0, 165, 255)


def draw_banner(image: np.ndarray, total: int, motion: bool) -> None:
    h, w = image.shape[:2]
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (w, 44), COL_BANNER, -1)
    cv2.addWeighted(overlay, 0.65, image, 0.35, 0, image)

    text = f"VEHICLES COUNTED: {total}"
    (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 0.85, 2)
    cv2.putText(image, text, ((w - tw) // 2, 30), cv2.FONT_HERSHEY_DUPLEX, 0.85, (255, 255, 255), 2)

    badge_text = "MOTION" if motion else "STILL"
    badge_col = COL_MOTION if motion else (120, 120, 120)
    (bw, _), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_PLAIN, 1.2, 1)
    cv2.putText(image, badge_text, (w - bw - 12, 28), cv2.FONT_HERSHEY_PLAIN, 1.2, badge_col, 2)


def visualize(frame: np.ndarray, tracks: list[Track], total: int, line_x: int, motion: bool) -> np.ndarray:
    h, w = frame.shape[:2]
    line_start = (line_x, 0)
    line_end = (line_x, h)

    for tr in tracks:
        x1, y1, x2, y2 = tr.box
        cx, cy = tr.centroid

        box_col = COL_COUNTED if tr.counted else COL_BOX
        cv2.rectangle(frame, (x1, y1), (x2, y2), box_col, 2)
        cv2.circle(frame, (cx, cy), 4, COL_DOT, -1)

        cls_name = VEHICLE_CLASS_NAMES.get(tr.class_id, "vehicle")
        label = f"#{tr.track_id} {cls_name} {tr.score:.2f}"
        cv2.putText(frame, label, (x1, max(50, y1 - 5)), cv2.FONT_HERSHEY_PLAIN, 1.05, COL_LABEL, 1)

    cv2.line(frame, line_start, line_end, COL_LINE, 2)
    cv2.putText(frame, "counting line", (line_x + 6, 68), cv2.FONT_HERSHEY_PLAIN, 1.0, COL_LINE, 1)

    draw_banner(frame, total, motion)
    return frame


# =============================================================================
# Main loop
# =============================================================================

def run(args):
    ensure_csv()

    display = PygameDisplay(fullscreen=args.fullscreen, logical_size=(args.width, args.height))

    use_picamera = args.source.lower() == "picamera"
    cap = None
    picam = None

    try:
        if use_picamera:
            picam = open_picamera(args.width, args.height)
        else:
            source = int(args.source) if args.source.isdigit() else args.source
            cap = cv2.VideoCapture(source)
            if not cap.isOpened():
                raise RuntimeError(f"Cannot open source '{args.source}'")
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

        detector = load_detector(args.model)
        tracker = IoUTracker()

        line_x = args.line_x if args.line_x is not None else args.width // 2

        fps = 0.0
        fps_counter = 0
        fps_timer = time.time()
        fps_window = 10

        raw_frame_count = 0
        last_annotated: Optional[np.ndarray] = None
        prev_gray: Optional[np.ndarray] = None
        read_failures = 0

        print("[Detector] Starting. Press ESC to quit.")
        print(f"[Config] frame={args.width}x{args.height}  line_x={line_x}  frame_skip={args.frame_skip}")
        print(f"[Config] model={args.model}")
        print(f"[CSV] Logging to: {CSV_PATH.resolve()}")

        running = True
        while running:
            running = display.pump()
            if not running:
                break

            if use_picamera:
                ok, image = read_picamera_frame(picam)
            else:
                ok, image = cap.read()

            if not ok or image is None:
                read_failures += 1
                if read_failures >= args.max_failures:
                    err = make_status_frame(args.width, args.height, [
                        "Camera read failed too many times.",
                        "Check camera permissions, cable, and that Picamera2 works alone.",
                        "Press ESC to quit.",
                    ])
                    display.show(err)
                    display.tick(10)
                    running = display.pump()
                    if not running:
                        break
                    time.sleep(0.1)
                    continue
                msg = make_status_frame(args.width, args.height, [
                    f"Waiting for camera frame... ({read_failures}/{args.max_failures})",
                    "If this keeps happening, the camera is not delivering frames.",
                    "Press ESC to quit.",
                ])
                display.show(msg)
                display.tick(15)
                continue

            read_failures = 0
            raw_frame_count += 1

            frame = cv2.resize(image, (args.width, args.height))
            if args.flip:
                frame = cv2.flip(frame, 1)

            motion, prev_gray = detect_motion(frame, prev_gray)
            cur_skip = max(1, args.frame_skip // MOTION_SPEEDUP_FACTOR) if motion else args.frame_skip

            # Always keep the screen alive on skipped frames.
            if raw_frame_count % cur_skip != 0:
                if last_annotated is not None:
                    display.show(last_annotated)
                else:
                    display.show(frame)

                running = display.pump()
                display.tick(60)
                continue

            detections = detector.detect(frame, args.conf)
            tracks = tracker.update(detections)

            for tr in tracks:
                direction = count_crossing(tr, line_x, CROSS_MARGIN)
                if direction is not None:
                    log_vehicle(tr.track_id, tr.class_id, tr.score, direction)

            total = sum(1 for tr in tracker.tracks if tr.counted)

            fps_counter += 1
            if fps_counter >= fps_window:
                elapsed = time.time() - fps_timer
                fps = fps_window / max(elapsed, 1e-6)
                fps_timer = time.time()
                fps_counter = 0

            annotated = visualize(frame, tracks, total, line_x, motion)
            skip_label = f"skip={cur_skip}" + (" ▲" if motion else "")
            cv2.putText(annotated, f"FPS {fps:.1f}  {skip_label}", (10, args.height - 12),
                        cv2.FONT_HERSHEY_PLAIN, 1.2, (0, 0, 255), 2)

            last_annotated = annotated
            display.show(annotated)

            running = display.pump()
            display.tick(60)

    finally:
        if cap is not None:
            cap.release()
        if picam is not None:
            picam.stop()
        if pygame is not None:
            display.close()

    print(f"[Done] Log saved to {CSV_PATH.resolve()}")


# =============================================================================
# CLI
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Lightweight vehicle counter for Raspberry Pi")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help="NCNN folder or .pt file. Default: yolo26n_ncnn_model")
    parser.add_argument("--source", default="picamera",
                        help="'picamera', camera index (0,1..), or video path.")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="Confidence threshold. Default: 0.25")
    parser.add_argument("--frame-skip", type=int, default=3,
                        help="Run inference every N frames normally. Default: 3")
    parser.add_argument("--line-x", type=int, default=None,
                        help="Vertical counting line X position. Default: center of the frame")
    parser.add_argument("--width", type=int, default=DEFAULT_DISPLAY_W,
                        help="Capture/display width. Default: 640")
    parser.add_argument("--height", type=int, default=DEFAULT_DISPLAY_H,
                        help="Capture/display height. Default: 480")
    parser.add_argument("--max-failures", type=int, default=60,
                        help="How many read failures before showing an error screen. Default: 60")
    parser.add_argument("--no-flip", dest="flip", action="store_false", default=True,
                        help="Disable horizontal flip.")
    parser.add_argument("--no-fullscreen", dest="fullscreen", action="store_false", default=True,
                        help="Windowed mode instead of fullscreen.")
    args = parser.parse_args()

    if args.frame_skip < 1:
        parser.error("--frame-skip must be >= 1")
    if args.width < 160 or args.height < 120:
        parser.error("--width/--height are too small")
    run(args)


if __name__ == "__main__":
    main()
