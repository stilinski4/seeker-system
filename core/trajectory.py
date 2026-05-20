import math

trajectory_points = []

MAX_POINTS   = 12     # hanya simpan 12 titik terakhir (dari 40)
MAX_JUMP_PX  = 80     # abaikan titik yang loncat lebih dari 80px

def add_point(x, y):

    global trajectory_points

    new_point = (int(x), int(y))

    # filter: abaikan jika posisi loncat terlalu jauh dari titik terakhir
    if len(trajectory_points) > 0:

        last = trajectory_points[-1]

        dist = math.hypot(new_point[0] - last[0], new_point[1] - last[1])

        if dist > MAX_JUMP_PX:
            # target loncat terlalu jauh — reset trajectory
            trajectory_points.clear()
            return

    trajectory_points.append(new_point)

    if len(trajectory_points) > MAX_POINTS:
        trajectory_points.pop(0)

def get_points():

    return trajectory_points

def clear_points():

    global trajectory_points
    trajectory_points.clear()
