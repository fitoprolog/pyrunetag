import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from detector import EllipseDetector, MarkerDetector, load_model


RED = (0, 0, 255)


def build_intrinsics(width, height, fx=None, fy=None, cx=None, cy=None):
    if fx is None:
        fx = float(width)
    if fy is None:
        fy = float(height)
    if cx is None:
        cx = width / 2.0
    if cy is None:
        cy = height / 2.0
    return np.array(
        [
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def marker_center(marker):
    points = []
    for slot in marker.code:
        if slot.payload is not None:
            points.append(slot.payload.center())
    if not points:
        return None
    pts = np.array(points, dtype=np.float64)
    return tuple(np.mean(pts, axis=0))


def marker_bbox(marker):
    xs = []
    ys = []
    for slot in marker.code:
        if slot.payload is None:
            continue
        (cx, cy), (w, h), _ = slot.payload.rr
        rx = w / 2.0
        ry = h / 2.0
        xs.extend((cx - rx, cx + rx))
        ys.extend((cy - ry, cy + ry))
    if not xs:
        return None
    x0 = int(round(min(xs)))
    y0 = int(round(min(ys)))
    x1 = int(round(max(xs)))
    y1 = int(round(max(ys)))
    return x0, y0, x1, y1


def draw_marker(frame, marker, origin):
    ox, oy = origin
    for slot in marker.code:
        if slot.payload is None:
            continue
        (cx, cy), (w, h), angle = slot.payload.rr
        center = (int(round(cx + ox)), int(round(cy + oy)))
        axes = (max(1, int(round(w / 2.0))), max(1, int(round(h / 2.0))))
        cv2.ellipse(frame, center, axes, angle, 0.0, 360.0, RED, 2, lineType=cv2.LINE_AA)
    bbox = marker_bbox(marker)
    if bbox is not None:
        x0, y0, x1, y1 = bbox
        cv2.rectangle(frame, (x0 + int(round(ox)), y0 + int(round(oy))), (x1 + int(round(ox)), y1 + int(round(oy))), RED, 2, lineType=cv2.LINE_AA)
    center = marker_center(marker)
    if center is None:
        return
    label = marker.model.name
    label_origin = (int(round(center[0] + ox)), int(round(center[1] + oy)))
    cv2.putText(frame, label, label_origin, cv2.FONT_HERSHEY_SIMPLEX, 1.0, RED, 3, cv2.LINE_AA)


def draw_status(frame, text):
    cv2.putText(frame, text, (16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, RED, 2, cv2.LINE_AA)


def load_models(paths):
    models = []
    for path in paths:
        path_obj = Path(path)
        if path_obj.is_dir():
            for child in sorted(path_obj.glob("*.txt")):
                try:
                    models.append(load_model(child))
                except ValueError:
                    continue
        else:
            models.append(load_model(path_obj))
    if not models:
        raise ValueError("no RUNE_direct model descriptors loaded")
    return models


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("models", nargs="+")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fx", type=float)
    parser.add_argument("--fy", type=float)
    parser.add_argument("--cx", type=float)
    parser.add_argument("--cy", type=float)
    parser.add_argument("--process-width", type=int, default=640)
    parser.add_argument("--max-detected", type=int, default=96)
    parser.add_argument("--min-filled-slots", type=int, default=24)
    parser.add_argument("--max-observed-errors", type=int, default=4)
    parser.add_argument("--min-observed-match-ratio", type=float, default=0.85)
    args = parser.parse_args()

    models = load_models(args.models)
    capture = cv2.VideoCapture(args.camera)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    ok, frame = capture.read()
    if not ok or frame is None:
        raise RuntimeError("unable to read from webcam")

    base_intrinsics = build_intrinsics(frame.shape[1], frame.shape[0], args.fx, args.fy, args.cx, args.cy)
    process_scale = min(1.0, args.process_width / float(frame.shape[1]))
    process_size = (int(round(frame.shape[1] * process_scale)), int(round(frame.shape[0] * process_scale)))
    intrinsics = base_intrinsics.copy()
    intrinsics[0, 0] *= process_scale
    intrinsics[1, 1] *= process_scale
    intrinsics[0, 2] *= process_scale
    intrinsics[1, 2] *= process_scale
    marker_detector = MarkerDetector(
        intrinsics,
        models,
        min_filled_slots=args.min_filled_slots,
        max_observed_errors=args.max_observed_errors,
        min_observed_match_ratio=args.min_observed_match_ratio,
    )
    ellipse_detector = EllipseDetector(max_detected=args.max_detected)
    last_markers = []
    last_dt = 0.0
    origin = (float(intrinsics[0, 2]), float(intrinsics[1, 2]))

    while True:
        ok, frame = capture.read()
        if not ok or frame is None:
            break
        display = frame
        if process_scale != 1.0:
            display = cv2.resize(frame, process_size, interpolation=cv2.INTER_AREA)
        start = time.perf_counter()
        last_markers, _ = marker_detector.detect_image(display, ellipse_detector)
        last_dt = time.perf_counter() - start
        for marker in last_markers:
            draw_marker(display, marker, origin)
        if last_dt > 0.0:
            draw_status(display, f"{1.0 / last_dt:.1f} fps")
        cv2.imshow("RUNETag Webcam PoC", display)
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord("q"):
            break

    capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
