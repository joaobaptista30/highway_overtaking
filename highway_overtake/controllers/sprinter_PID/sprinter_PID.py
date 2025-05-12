from vehicle import Driver
import math
import time

driver = Driver()
timestep = int(driver.getBasicTimeStep())

# Constants
TARGET_SPEED = 60.0  # m/s
LEFT_LANE_CENTER_Y = 3.1
SAFE_DISTANCE = 10.0
OVERTAKE_DONE_DISTANCE = 3.0
LANE_WIDTH = 3.5
MAX_STEERING_ANGLE = 0.5  # radians

# Initialize LiDAR
lidar = driver.getDevice("lidar")
lidar.enable(timestep)

# Initialize GPS
gps = driver.getDevice("gps")
gps.enable(timestep)

# State definitions
CRUISE = 0
OVERTAKE = 1
FOLLOW = 2
state = CRUISE
print("[INFO]: cruising")


driver.setCruisingSpeed(TARGET_SPEED)

# PID controller variables
Kp = 0.1
Ki = 0.0
Kd = 0.3
integral = 0.0
previous_error = 0.0

def safe_steering(angle):
    return max(-MAX_STEERING_ANGLE, min(MAX_STEERING_ANGLE, angle))

lane_centers = [
    (-43.6912, 3.20022),
    (-68.3553, 3.20078),
    (-89.5582, 3.20117),
    (-144.177, 3.20101),
    (-207.044, 3.19843),
    (-279.784, 3.19225)
]

def get_lane_center_y(x_current):
    # If before first or after last point
    if x_current <= lane_centers[0][0]:
        return lane_centers[0][1]
    elif x_current >= lane_centers[-1][0]:
        return lane_centers[-1][1]
    
    # Linear interpolation between surrounding points
    for i in range(len(lane_centers) - 1):
        x0, y0 = lane_centers[i]
        x1, y1 = lane_centers[i + 1]
        if x0 <= x_current <= x1:
            ratio = (x_current - x0) / (x1 - x0)
            return y0 + ratio * (y1 - y0)
    
    # Fallback — shouldn't reach here
    return lane_centers[-1][1]



while driver.step() != -1:

    ranges = lidar.getRangeImage()
    front_distance = ranges[len(ranges) // 2]

    if state == CRUISE:
        driver.setSteeringAngle(0.0)
        if front_distance < SAFE_DISTANCE:
            state = OVERTAKE
            print("[INFO]: overtaking")


    elif state == OVERTAKE:
        driver.setSteeringAngle(-0.1)  # steer left
        left_indices = range(0, len(ranges) // 3)
        left_distances = [r for i, r in enumerate(ranges) if i in left_indices and math.isfinite(r)]
        left_distance = sum(left_distances) / len(left_distances) if left_distances else SAFE_DISTANCE
        if left_distance < SAFE_DISTANCE:
            time_start = time.time()
            while time.time() - time_start < 0.15:
                driver.step()
            state = FOLLOW
            integral = 0.0  # Reset PID state
            previous_error = 0.0
            print("[INFO]: following")


    elif state == FOLLOW:
        current_x = gps.getValues()[0]
        current_y = gps.getValues()[1]
        desired_y = get_lane_center_y(current_x)
        error = desired_y - current_y


        # PID calculations
        integral += error * (timestep / 1000.0)
        derivative = (error - previous_error) / (timestep / 1000.0)
        steering_correction = Kp * error + Ki * integral + Kd * derivative

        driver.setSteeringAngle(safe_steering(steering_correction))
        previous_error = error
