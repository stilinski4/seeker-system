from pymavlink import mavutil

# CONNECT PIXHAWK
master = mavutil.mavlink_connection(
    '/dev/ttyAMA0',
    baud=57600
)

print("WAITING HEARTBEAT...")

master.wait_heartbeat()

print("PIXHAWK CONNECTED")

def send_yaw_command(yaw_speed):

    master.mav.command_long_send(

        master.target_system,
        master.target_component,

        mavutil.mavlink.MAV_CMD_CONDITION_YAW,

        0,

        abs(yaw_speed),  # speed
        0,               # angle
        1 if yaw_speed > 0 else -1,
        0,
        0,
        0,
        0
    )

    print("YAW COMMAND SENT")
