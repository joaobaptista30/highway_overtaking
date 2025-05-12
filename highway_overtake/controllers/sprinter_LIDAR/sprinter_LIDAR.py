from vehicle import Driver
import math
import time

driver = Driver()
timestep = int(driver.getBasicTimeStep())

# Constants
TARGET_SPEED = 60.0  # m/s
LEFT_LANE_ANGLE = -0.1  # radians, adjust as needed
ALIGNMENT_THRESHOLD = 0.01  # radians
# Vehicle control
MAX_SPEED = 60.0  # Max cruising speed
SAFE_DISTANCE = 5.0  # Trigger distance to start overtake (m)
OVERTAKE_DONE_DISTANCE = 3.0  # Distance when overtake is considered done
LANE_WIDTH = 3.5  # Approx lane width (meters)

# Initialize LiDAR
lidar = driver.getDevice("lidar")
lidar.enable(timestep)

# State definitions
CRUISE = 0
OVERTAKE = 1
FOLLOW = 2
state = CRUISE
print("[INFO]: cruising")

driver.setCruisingSpeed(TARGET_SPEED)

def get_front_distance(ranges):
    return ranges[len(ranges) // 2]

def get_left_distance(ranges):
    # Assuming LiDAR FOV is 180 degrees and ranges are ordered left to right
    left_indices = range(0, len(ranges) // 3)
    left_distances = [ranges[i] for i in left_indices]
    return sum(left_distances) / len(left_distances)

def get_alignment_error(ranges):
    # Simple method: compare distances at slight angles to estimate alignment
    left = ranges[len(ranges) // 4]
    right = ranges[3 * len(ranges) // 4]
    return left - right

while driver.step() != -1:
    ranges = lidar.getRangeImage()
    front_distance = get_front_distance(ranges)

    if state == CRUISE:
        driver.setSteeringAngle(0.0)
        if front_distance < SAFE_DISTANCE:
            state = OVERTAKE
            print("[INFO]: overtaking")


    elif state == OVERTAKE:
        # Begin lane change to the left
        driver.setSteeringAngle(LEFT_LANE_ANGLE)
        left_distance = get_left_distance(ranges)
        if left_distance < SAFE_DISTANCE:
            time_start = time.time()
            while time.time() - time_start < 0.15:
                driver.step()
            state = FOLLOW
            print("[INFO]: following")

    elif state == FOLLOW:
        # Align with the front car's trajectory
        alignment_error = get_alignment_error(ranges)
        if abs(alignment_error) > ALIGNMENT_THRESHOLD:
            steering_correction = -alignment_error * 0.1  # Proportional control
            driver.setSteeringAngle(steering_correction)
        else:
            driver.setSteeringAngle(0.0)
