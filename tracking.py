from ultralytics import YOLO
from picamera2 import Picamera2
import cv2
import time
import os

os.makedirs("outputs", exist_ok=True)

# LOAD MODEL
model = YOLO("models/best.pt")

# INIT CAMERA
picam2 = Picamera2()

config = picam2.create_preview_configuration(
    main={"size": (640, 480)}
)

picam2.configure(config)
picam2.start()

time.sleep(2)

print("TRACKING STARTED")

while True:

    frame = picam2.capture_array()

    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    results = model.track(
        frame,
        persist=True,
        verbose=False
    )

    annotated = results[0].plot()

    cv2.imwrite("outputs/tracking.jpg", annotated)

    print("TRACKING RUNNING")
