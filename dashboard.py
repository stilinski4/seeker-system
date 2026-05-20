from flask import Flask
from flask import render_template_string
from flask import Response
from flask import jsonify

import cv2
import json
import time
import os

app = Flask(__name__)

HTML = """
<!DOCTYPE html>
<html>
<head>
<title>SEEKER SYSTEM</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { background: #050505; color: #00ff66; font-family: monospace; }
header { text-align: center; padding: 16px; font-size: 20px; letter-spacing: 4px; border-bottom: 1px solid #00ff6633; }
.container { display: flex; flex-direction: row; align-items: flex-start; gap: 24px; padding: 24px; }
.video-box { flex: 0 0 50%; display: flex; flex-direction: column; gap: 8px; }
.video-box img { width: 100%; border: 2px solid #00ff66; display: block; }
.video-label { font-size: 11px; letter-spacing: 2px; color: #00ff6688; text-align: center; }
.data-box { flex: 1; display: flex; flex-direction: column; gap: 16px; }
.state-badge { font-size: 22px; font-weight: bold; letter-spacing: 4px; padding: 12px 20px; border: 2px solid #00ff66; display: inline-block; margin-bottom: 4px; }
.state-LOCK   { color: #00ff66; border-color: #00ff66; }
.state-LOST   { color: #ff3333; border-color: #ff3333; }
.state-SEARCH { color: #ffff00; border-color: #ffff00; }
.section-title { font-size: 11px; letter-spacing: 3px; color: #00ff6666; margin-bottom: 6px; border-bottom: 1px solid #00ff6622; padding-bottom: 4px; }
.row { display: flex; justify-content: space-between; align-items: center; padding: 6px 0; border-bottom: 1px solid #00ff6611; font-size: 14px; }
.row .label { color: #00ff6699; letter-spacing: 1px; }
.row .value { color: #00ff66; font-size: 15px; }
.row .value.cyan   { color: #00ffff; }
.row .value.yellow { color: #ffff00; }
.card { background: #0a0a0a; border: 1px solid #00ff6622; padding: 14px 18px; border-radius: 4px; }
</style>
</head>
<body>
<header>&#9632; AI SEEKER SYSTEM</header>
<div class="container">
    <div class="video-box">
        <img src="/video_feed">
        <div class="video-label">LIVE FEED</div>
    </div>
    <div class="data-box">
        <div class="card">
            <div class="section-title">STATUS</div>
            <div id="state" class="state-badge state-SEARCH">SEARCH</div>
        </div>
        <div class="card">
            <div class="section-title">TARGET</div>
            <div class="row"><span class="label">CONFIDENCE</span><span class="value" id="confidence">--</span></div>
            <div class="row"><span class="label">TARGET X</span><span class="value" id="target_x">--</span></div>
            <div class="row"><span class="label">TARGET Y</span><span class="value" id="target_y">--</span></div>
        </div>
        <div class="card">
            <div class="section-title">GUIDANCE (PID)</div>
            <div class="row"><span class="label">YAW ERROR</span><span class="value cyan" id="yaw_error">--</span></div>
            <div class="row"><span class="label">PITCH ERROR</span><span class="value cyan" id="pitch_error">--</span></div>
            <div class="row"><span class="label">YAW OUTPUT</span><span class="value cyan" id="yaw_output">--</span></div>
            <div class="row"><span class="label">PITCH OUTPUT</span><span class="value cyan" id="pitch_output">--</span></div>
        </div>
        <div class="card">
            <div class="section-title">SYSTEM</div>
            <div class="row"><span class="label">FPS</span><span class="value yellow" id="fps">--</span></div>
            <div class="row"><span class="label">CPU</span><span class="value yellow" id="cpu">--</span></div>
            <div class="row"><span class="label">CPU TEMP</span><span class="value" id="cpu_temp">--</span></div>
        </div>
    </div>
</div>
<script>
async function updateTelemetry() {
    try {
        const response = await fetch('/telemetry')
        const data = await response.json()

        const stateEl = document.getElementById("state")
        stateEl.innerText = data.state
        stateEl.className = "state-badge state-" + data.state

        document.getElementById("confidence").innerText =
            data.confidence !== undefined ? (parseFloat(data.confidence) * 100).toFixed(1) + "%" : "--"
        document.getElementById("target_x").innerText =
            data.target_x !== undefined ? parseFloat(data.target_x).toFixed(1) : "--"
        document.getElementById("target_y").innerText =
            data.target_y !== undefined ? parseFloat(data.target_y).toFixed(1) : "--"
        document.getElementById("yaw_error").innerText =
            data.yaw_error !== undefined ? parseFloat(data.yaw_error).toFixed(2) : "--"
        document.getElementById("pitch_error").innerText =
            data.pitch_error !== undefined ? parseFloat(data.pitch_error).toFixed(2) : "--"
        document.getElementById("yaw_output").innerText =
            data.yaw_output !== undefined ? parseFloat(data.yaw_output).toFixed(4) : "--"
        document.getElementById("pitch_output").innerText =
            data.pitch_output !== undefined ? parseFloat(data.pitch_output).toFixed(4) : "--"
        document.getElementById("fps").innerText =
            data.fps !== undefined ? data.fps : "--"
        document.getElementById("cpu").innerText =
            data.cpu !== undefined ? data.cpu + "%" : "--"

        const tempEl = document.getElementById("cpu_temp")
        if (data.cpu_temp !== undefined) {
            tempEl.innerText = data.cpu_temp + "°C"
            if (data.cpu_temp >= 80) {
                tempEl.style.color = "#ff3333"
            } else if (data.cpu_temp >= 70) {
                tempEl.style.color = "#ff9900"
            } else {
                tempEl.style.color = "#00ff66"
            }
        } else {
            tempEl.innerText = "--"
        }

    } catch(e) {
        console.log("Telemetry error:", e)
    }
}
setInterval(updateTelemetry, 300)
</script>
</body>
</html>
"""

def generate_frames():
    while True:
        frame = cv2.imread("outputs/realtime.jpg")
        if frame is None:
            continue
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n'
            + frame_bytes +
            b'\r\n'
        )
        time.sleep(0.03)

@app.route('/')
def index():
    return render_template_string(HTML)

@app.route('/video_feed')
def video_feed():
    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

@app.route('/telemetry')
def telemetry():
    try:
        with open("outputs/telemetry.json", "r") as f:
            data = json.load(f)
    except:
        data = {
            "state": "SEARCH",
            "confidence": 0,
            "fps": 0,
            "cpu": 0,
            "cpu_temp": 0
        }
    return jsonify(data)

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False
    )
