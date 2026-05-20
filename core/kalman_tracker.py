import cv2
import numpy as np

# =========================
# CREATE KALMAN FILTER
# =========================
kalman = cv2.KalmanFilter(4, 2)

kalman.measurementMatrix = np.array(
    [[1,0,0,0],
     [0,1,0,0]],
    np.float32
)

kalman.transitionMatrix = np.array(
    [[1,0,1,0],
     [0,1,0,1],
     [0,0,1,0],
     [0,0,0,1]],
    np.float32
)

kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03

initialized = False

# =========================
# UPDATE — dipanggil saat target terdeteksi
# =========================
def update(measured_x, measured_y):

    global initialized

    measurement = np.array(
        [[np.float32(measured_x)],
         [np.float32(measured_y)]]
    )

    if not initialized:

        kalman.statePre = np.array(
            [[measured_x],
             [measured_y],
             [0],
             [0]],
            np.float32
        )

        initialized = True

    kalman.correct(measurement)

    prediction = kalman.predict()

    pred_x = float(prediction[0][0])
    pred_y = float(prediction[1][0])

    return pred_x, pred_y


# =========================
# PREDICT — alias update()
# Dipakai di seeker_lock.py
# =========================
def predict(measured_x, measured_y):
    return update(measured_x, measured_y)


# =========================
# PREDICT ONLY — tanpa measurement baru
# Dipakai saat target hilang sementara
# =========================
def predict_only():

    if not initialized:
        return None, None

    prediction = kalman.predict()

    pred_x = float(prediction[0][0])
    pred_y = float(prediction[1][0])

    return pred_x, pred_y


# =========================
# RESET
# =========================
def reset():

    global initialized
    initialized = False
