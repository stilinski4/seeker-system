import time

class PID:

    def __init__(self, kp, ki, kd, integral_limit=500):

        self.kp = kp
        self.ki = ki
        self.kd = kd

        # Anti-windup: batas maksimum nilai integral
        self.integral_limit = integral_limit

        self.previous_error = 0
        self.integral = 0

        # Untuk delta time
        self._last_time = time.time()

    def compute(self, error):

        now = time.time()
        dt = now - self._last_time

        # Hindari dt = 0
        if dt <= 0:
            dt = 1e-6

        self._last_time = now

        # PROPORTIONAL
        p = self.kp * error

        # INTEGRAL dengan anti-windup clamp
        self.integral += error * dt
        self.integral = max(
            -self.integral_limit,
            min(self.integral_limit, self.integral)
        )
        i = self.ki * self.integral

        # DERIVATIVE berbasis dt
        derivative = (error - self.previous_error) / dt
        d = self.kd * derivative

        self.previous_error = error

        return p + i + d

    def reset(self):
        """Reset state PID saat target hilang."""
        self.integral = 0
        self.previous_error = 0
        self._last_time = time.time()
