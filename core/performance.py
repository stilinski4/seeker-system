import time

last_time = time.time()

def calculate_fps():

    global last_time

    now = time.time()

    fps = 1 / (now - last_time)

    last_time = now

    return round(fps, 2)

