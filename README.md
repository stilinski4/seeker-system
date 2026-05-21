# Seeker System — Drone Target Tracker

Low-cost autonomous target detection, tracking, and locking system for drone platforms. Built on Raspberry Pi 5 with YOLOv8/YOLOv11 + ByteTrack + Kalman Filter, communicating via MAVLink to a Pixhawk flight controller.

---

## Features

- Real-time object detection using YOLOv8/YOLOv11n (ONNX/PT)
- Multi-object tracking with ByteTrack
- Kalman Filter for smooth target position estimation
- PID controller for yaw and pitch correction
- Finite State Machine: `SEARCH → LOCK → LOST`
- Target priority management (closest to center)
- Trajectory prediction and fade trail visualization
- Seeker View (PiP crop of target area)
- Live telemetry output (`outputs/telemetry.json`)
- Flask web dashboard for remote monitoring
- MAVLink integration for Pixhawk yaw command

---

## Hardware

| Component | Detail |
|---|---|
| Compute | Raspberry Pi 5 (4GB RAM) |
| Camera | Arducam 16MP (CSI interface) |
| Flight Controller | Pixhawk (MAVLink via `/dev/ttyAMA0`) |
| Power | LiPo + XY-3606 Buck Converter |
| Cooling | Active cooling kit |
| Enclosure | 3D printed PLA casing |

---

## Project Structure

```
seeker-system/
├── core/
│   ├── class_filter.py      # Allowed detection classes
│   ├── kalman_tracker.py    # Kalman Filter position estimator
│   ├── performance.py       # FPS calculator
│   ├── pid.py               # PID controller (yaw & pitch)
│   ├── predictor.py         # Target position predictor
│   ├── seeker_state.py      # FSM states (SEARCH, ACQUIRE, LOCK, LOST)
│   ├── target_manager.py    # Target selection & priority
│   └── trajectory.py        # Trajectory trail points
├── models/
│   ├── best.pt              # Vehicle detection model
│   ├── best_tank.pt         # Military vehicle model
│   └── yolov8n.pt           # Base YOLOv8n model
├── videos/                  # Test video files
├── outputs/                 # Detection results & telemetry
├── realtime_seeker.py       # Main program (video input mode)
├── tracking.py              # Tracking pipeline
├── dashboard.py             # Flask web dashboard
├── mavlink.py               # MAVLink / Pixhawk interface
└── test_footage.py          # Offline testing with video
```

---

## Installation

### 1. Clone repository

```bash
git clone https://github.com/stilinski4/seeker-system
cd seeker-system
```

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 3. Install Picamera2 (Raspberry Pi only)

```bash
sudo apt install -y python3-picamera2
```

### 4. Add model weights

Copy your `.pt` model files into the `models/` folder manually (not tracked by Git due to file size).

---

## Usage

### Run with video file (testing)

Edit `VIDEO_INPUT` path in `realtime_seeker.py`, then:

```bash
python realtime_seeker.py
```

Output video saved to `outputs/`. Open browser at `http://<IP_RASPY>:5000` to view dashboard.

### Run offline test

```bash
python test_footage.py
```

### View web dashboard

```bash
python dashboard.py
```

Then open: `http://<IP_RASPY>:5000`

---

## Target Classes

Defined in `core/class_filter.py`:

```
car, tank, military_truck, military_vehicle, Tank
```

To add or remove classes, edit `ALLOWED_CLASSES` in that file.

---

## Telemetry Output

Live telemetry is written to `outputs/telemetry.json` on every frame:

```json
{
    "state": "LOCK",
    "target_id": 1,
    "target_x": 320.0,
    "target_y": 180.0,
    "yaw_error": 12.5,
    "pitch_error": -4.2,
    "yaw_output": 0.125,
    "pitch_output": -0.042,
    "confidence": 0.87,
    "fps": 24,
    "cpu": 61.0,
    "cpu_temp": 58.3
}
```

---

## FSM States

| State | Description |
|---|---|
| `SEARCH` | No target detected, static crosshair active |
| `ACQUIRE` | Target candidate found, beginning lock |
| `LOCK` | Target confirmed and tracked, PID active |
| `LOST` | Target disappeared, transitioning back to SEARCH |

---

## License

For academic / research use only.
