from ultralytics import YOLO
from core.pid import PID
from core.seeker_state import SeekerState
from core.kalman_tracker import update
from core.class_filter import ALLOWED_CLASSES
from core.performance import calculate_fps
from core.trajectory import add_point
from core.trajectory import get_points
from core.trajectory import clear_points
from core.predictor import predict
from core.target_manager import TargetManager
import cv2
import time
import os
import json
import psutil
import datetime
import numpy as np

# ============================================
# KONFIGURASI
# ============================================
VIDEO_INPUT  = "videos/test4.mp4"
VIDEO_OUTPUT = "outputs/test_result4latest.mp4"

FRAME_WIDTH  = 640
FRAME_HEIGHT = 360

# ============================================
# COLORS & FONT
# ============================================
GREEN     = (0, 255, 80)
GREEN_DIM = (0, 140, 40)
WHITE     = (220, 220, 220)
RED       = (0, 60, 255)
CYAN      = (0, 220, 220)
ORANGE    = (0, 165, 255)
FONT      = cv2.FONT_HERSHEY_SIMPLEX

# ============================================
# OUTPUT FOLDER
# ============================================
os.makedirs("outputs", exist_ok=True)

# ============================================
# MODEL
# ============================================
model = YOLO("models/best.pt")

print("Warming up model...")
dummy = np.zeros((416, 416, 3), dtype=np.uint8)
model.predict(dummy, imgsz=416, verbose=False)
print("Model ready.")
print(f"Model classes: {model.names}")

# ============================================
# PID
# ============================================
yaw_pid   = PID(kp=0.01, ki=0.0001, kd=0.003)
pitch_pid = PID(kp=0.01, ki=0.0001, kd=0.003)

# ============================================
# TARGET MANAGER
# ============================================
manager = TargetManager()

# ============================================
# ALLOWED CLASSES (case-insensitive)
# ============================================
allowed_lower = [c.lower() for c in ALLOWED_CLASSES]

# ============================================
# CPU TEMPERATURE
# ============================================
def get_cpu_temp():
    try:
        with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
            temp = int(f.read().strip()) / 1000.0
        return round(temp, 1)
    except:
        return 0.0

# ============================================
# DRAW HELPERS
# ============================================
def draw_text(img, text, x, y, color=GREEN, scale=0.45, thick=1):
    cv2.putText(img, text, (x, y), FONT, scale, color, thick, cv2.LINE_AA)

def draw_panel(img, x, y, w, h, alpha=0.50):
    overlay = img.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

# ============================================
# CROSSHAIR BERGERAK (saat LOCK)
# ============================================
def draw_tracking_crosshair(img, tx, ty, W, H):
    cx = int(tx)
    cy = int(ty)
    cv2.line(img, (0, cy), (W, cy), GREEN, 1, cv2.LINE_AA)
    cv2.line(img, (cx, 0), (cx, H), GREEN, 1, cv2.LINE_AA)
    for x in range(0, W, 80):
        cv2.line(img, (x, cy - 4), (x, cy + 4), GREEN, 1)
    for y in range(0, H, 80):
        cv2.line(img, (cx - 4, y), (cx + 4, y), GREEN, 1)
    sq = 12
    cv2.rectangle(img, (cx - sq, cy - sq), (cx + sq, cy + sq), GREEN, 1, cv2.LINE_AA)
    cv2.circle(img, (cx, cy), 2, GREEN, -1)

# ============================================
# CROSSHAIR STATIS (saat SEARCH)
# ============================================
def draw_static_crosshair(img, W, H):
    cx = W // 2
    cy = H // 2
    cv2.line(img, (0, cy), (W, cy), GREEN_DIM, 1, cv2.LINE_AA)
    cv2.line(img, (cx, 0), (cx, H), GREEN_DIM, 1, cv2.LINE_AA)
    sq = 12
    cv2.rectangle(img, (cx - sq, cy - sq), (cx + sq, cy + sq), GREEN_DIM, 1)

# ============================================
# SEEKER VIEW PIP
# ============================================
def draw_seeker_view(img, raw_frame, tx, ty, bw, bh, W, H):
    pip_w, pip_h = 240, 160
    margin = 14
    px = W - pip_w - margin
    py = H - pip_h - margin - 6

    pad    = 50
    cx_    = int(tx)
    cy_    = int(ty)
    half_w = max(bw // 2 + pad, 70)
    half_h = max(bh // 2 + pad, 70)

    x1c = max(cx_ - half_w, 0)
    y1c = max(cy_ - half_h, 0)
    x2c = min(cx_ + half_w, W)
    y2c = min(cy_ + half_h, H)

    crop = raw_frame[y1c:y2c, x1c:x2c]
    if crop.size == 0:
        return

    crop_resized = cv2.resize(crop, (pip_w, pip_h))
    draw_panel(img, px - 2, py - 20, pip_w + 4, pip_h + 22, alpha=0.6)
    cv2.rectangle(img, (px - 2, py - 20), (px + pip_w + 2, py - 2), (0, 80, 30), -1)
    draw_text(img, "SEEKER VIEW", px + 4, py - 6, GREEN, 0.40)
    img[py:py + pip_h, px:px + pip_w] = crop_resized
    cv2.rectangle(img, (px, py), (px + pip_w, py + pip_h), GREEN, 1)
    mx = px + pip_w // 2
    my = py + pip_h // 2
    cv2.line(img, (px, my), (px + pip_w, my), GREEN_DIM, 1)
    cv2.line(img, (mx, py), (mx, py + pip_h), GREEN_DIM, 1)
    sq = 6
    cv2.rectangle(img, (mx - sq, my - sq), (mx + sq, my + sq), GREEN, 1)

# ============================================
# OPEN VIDEO
# ============================================
cap = cv2.VideoCapture(VIDEO_INPUT, cv2.CAP_FFMPEG)

if not cap.isOpened():
    print(f"[ERROR] Tidak bisa buka video: {VIDEO_INPUT}")
    print("Pastikan file ada dan tidak rusak: ls -lh videos/test.mp4")
    print("Coba repair: ffmpeg -i videos/test.mp4 -c copy videos/test_fixed.mp4")
    exit()

fps_src = cap.get(cv2.CAP_PROP_FPS)
delay   = 1.0 / fps_src if fps_src > 0 else 0.033

print(f"Video: {VIDEO_INPUT}")
print(f"FPS source: {fps_src:.1f}")

# ============================================
# VIDEO WRITER
# ============================================
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
writer = cv2.VideoWriter(VIDEO_OUTPUT, fourcc, 25.0, (FRAME_WIDTH, FRAME_HEIGHT))
print(f"Recording ke: {VIDEO_OUTPUT}")

print("FOOTAGE TEST STARTED")

# ============================================
# MAIN LOOP
# ============================================
while True:

    ret, frame = cap.read()

    if not ret:
        print("VIDEO END")
        break

    # resize ke resolusi kerja
    frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

    raw_frame = frame.copy()

    W = FRAME_WIDTH
    H = FRAME_HEIGHT

    # ============================================
    # YOLO TRACKING
    # ============================================
    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        imgsz=640,
        conf=0.25,
        iou=0.45,
        agnostic_nms=True,
        verbose=False
    )

    detections = []

    for box in results[0].boxes:
        cls        = int(box.cls[0])
        class_name = model.names[cls]
        if class_name.lower() not in allowed_lower:
            continue
        detections.append((box, class_name))

    # ============================================
    # TARGET PRIORITY
    # ============================================
    boxes_only     = [d[0] for d in detections]
    selected       = manager.select_target(boxes_only)
    selected_class = "TARGET"

    if selected is not None:
        for box, cname in detections:
            if box is selected:
                selected_class = cname.upper()
                break

    fps         = calculate_fps()
    cpu         = psutil.cpu_percent()
    cpu_temp    = get_cpu_temp()
    now_utc     = datetime.datetime.utcnow().strftime("%H:%M:%S UTC")
    num_targets = len(detections)

    # ============================================
    # TARGET TERDETEKSI
    # ============================================
    if selected is not None:

        x1, y1, x2, y2 = selected.xyxy[0]
        conf = float(selected.conf[0])

        track_id = 1
        if selected.id is not None:
            track_id = int(selected.id[0])

        bw = int(x2 - x1)
        bh = int(y2 - y1)

        measured_x = (x1 + x2) / 2
        measured_y = (y1 + y2) / 2

        kalman_x, kalman_y = update(measured_x, measured_y)
        target_x, target_y = predict(kalman_x, kalman_y)

        yaw_error    = target_x - W / 2
        pitch_error  = target_y - H / 2
        yaw_output   = yaw_pid.compute(yaw_error)
        pitch_output = pitch_pid.compute(pitch_error)

        add_point(target_x, target_y)
        points = get_points()

        # trajectory fade
        total = len(points)
        for i in range(1, total):
            alpha = i / total
            color = (
                int(GREEN_DIM[0] + (GREEN[0] - GREEN_DIM[0]) * alpha),
                int(GREEN_DIM[1] + (GREEN[1] - GREEN_DIM[1]) * alpha),
                int(GREEN_DIM[2] + (GREEN[2] - GREEN_DIM[2]) * alpha),
            )
            thickness = 1 if i < total - 3 else 2
            cv2.line(frame, points[i-1], points[i], color, thickness)

        draw_tracking_crosshair(frame, target_x, target_y, W, H)

        cv2.rectangle(
            frame,
            (int(x1), int(y1)),
            (int(x2), int(y2)),
            GREEN, 2
        )

        clen = 14
        corners = [
            ((int(x1), int(y1)), (+clen, 0), (0, +clen)),
            ((int(x2), int(y1)), (-clen, 0), (0, +clen)),
            ((int(x1), int(y2)), (+clen, 0), (0, -clen)),
            ((int(x2), int(y2)), (-clen, 0), (0, -clen)),
        ]
        for (ox, oy), (dx1, dy1), (dx2, dy2) in corners:
            cv2.line(frame, (ox, oy), (ox+dx1, oy+dy1), WHITE, 2)
            cv2.line(frame, (ox, oy), (ox+dx2, oy+dy2), WHITE, 2)

        label = f"{selected_class}  {conf*100:.0f}%"
        lx = int(x1)
        ly = int(y1) - 8
        if ly < 16:
            ly = int(y2) + 16
        draw_text(frame, label, lx, ly, GREEN, 0.48, 1)
        draw_text(frame, "TRACKING", lx, int(y2) + 16, GREEN_DIM, 0.40, 1)

        draw_seeker_view(frame, raw_frame, target_x, target_y, bw, bh, W, H)

        # panel kiri atas
        draw_panel(frame, 8, 8, 190, 138)
        tl = [
            ("TARGET SELECTOR",                             GREEN, 0.42),
            (f"Targets  : {num_targets:02d}",               WHITE, 0.44),
            ("",                                            GREEN, 0.40),
            ("TARGET TRACKING",                             GREEN, 0.42),
            (f"ID       : {track_id:02d}",                  WHITE, 0.44),
            (f"Conf     : {conf*100:.0f}%",                 WHITE, 0.44),
            (f"Size     : {bw}x{bh}px",                     WHITE, 0.44),
            (f"Pos XY   : {int(target_x)},{int(target_y)}", WHITE, 0.44),
        ]
        ty_ = 24
        for text, color, scale in tl:
            draw_text(frame, text, 14, ty_, color, scale)
            ty_ += 16

        # panel kanan atas
        draw_panel(frame, W - 210, 8, 202, 160)
        temp_color = ORANGE if cpu_temp >= 70 else WHITE
        tr = [
            ("TRACK AUTOMATION",           GREEN,      0.42),
            ("Source  : VIDEO",            WHITE,      0.44),
            ("Seeker  : TEST_FOOTAGE",     WHITE,      0.44),
            ("Tracker : ByteTrack",        WHITE,      0.44),
            ("Model   : YOLOv8",           WHITE,      0.44),
            ("",                           GREEN,      0.40),
            ("TRACKER DATA",               GREEN,      0.42),
            (f"FPS     : {fps}",           WHITE,      0.44),
            (f"CPU     : {cpu}%",          WHITE,      0.44),
            (f"Temp    : {cpu_temp}\xb0C", temp_color, 0.44),
            (f"Time    : {now_utc}",       WHITE,      0.44),
        ]
        ty_ = 24
        for text, color, scale in tr:
            draw_text(frame, text, W - 206, ty_, color, scale)
            ty_ += 16

        state_str = "[ LOCK ]"
        sw = cv2.getTextSize(state_str, FONT, 0.55, 1)[0][0]
        draw_text(frame, state_str, W // 2 - sw // 2, 22, GREEN, 0.55, 1)

        telemetry = {
            "state":        "LOCK",
            "target_id":    track_id,
            "target_x":     float(target_x),
            "target_y":     float(target_y),
            "yaw_error":    float(yaw_error),
            "pitch_error":  float(pitch_error),
            "yaw_output":   float(yaw_output),
            "pitch_output": float(pitch_output),
            "confidence":   conf,
            "fps":          fps,
            "cpu":          cpu,
            "cpu_temp":     cpu_temp
        }

    # ============================================
    # TIDAK ADA TARGET
    # ============================================
    else:

        clear_points()

        draw_static_crosshair(frame, W, H)

        draw_panel(frame, 8, 8, 190, 56)
        draw_text(frame, "TARGET SELECTOR", 14, 24, GREEN, 0.42)
        draw_text(frame, "Targets  : 00",   14, 42, WHITE, 0.44)

        draw_panel(frame, W - 210, 8, 202, 96)
        draw_text(frame, "TRACKER DATA",            W - 206, 24, GREEN, 0.42)
        draw_text(frame, f"FPS  : {fps}",           W - 206, 42, WHITE, 0.44)
        draw_text(frame, f"CPU  : {cpu}%",          W - 206, 58, WHITE, 0.44)
        temp_color_s = ORANGE if cpu_temp >= 70 else WHITE
        draw_text(frame, f"Temp : {cpu_temp}\xb0C", W - 206, 74, temp_color_s, 0.44)
        draw_text(frame, f"Time : {now_utc}",       W - 206, 90, WHITE, 0.44)

        state_str = "[ SEARCHING ]"
        sw = cv2.getTextSize(state_str, FONT, 0.55, 1)[0][0]
        draw_text(frame, state_str, W // 2 - sw // 2, 22, CYAN, 0.55, 1)

        telemetry = {
            "state":      "SEARCH",
            "confidence": 0,
            "fps":        fps,
            "cpu":        cpu,
            "cpu_temp":   cpu_temp
        }

    if cpu > 80:
        draw_text(frame, f"!! CPU HIGH: {cpu}% !!",
            14, H - 14, RED, 0.50, 1)

    # ============================================
    # SAVE TELEMETRY
    # ============================================
    with open("outputs/telemetry.json", "w") as f:
        json.dump(telemetry, f, indent=4)

    # ============================================
    # SAVE FRAME (untuk dashboard web)
    # ============================================
    cv2.imwrite("outputs/realtime.jpg", frame)

    # ============================================
    # SAVE TO VIDEO OUTPUT
    # ============================================
    writer.write(frame)

    time.sleep(delay)

# ============================================
# CLEANUP
# ============================================
cap.release()
writer.release()

print(f"[INFO] Video tersimpan: {VIDEO_OUTPUT}")
print("TEST FINISHED")
