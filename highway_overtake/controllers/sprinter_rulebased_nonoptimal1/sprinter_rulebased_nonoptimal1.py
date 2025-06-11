from vehicle import Driver
import math
import time
import random

driver = Driver()
timestep = int(driver.getBasicTimeStep())

# Constants
TARGET_SPEED = 50.0  # m/s
LEFT_LANE_CENTER_Y = 3.1
SAFE_DISTANCE = 7.0
OVERTAKE_DONE_DISTANCE = 3.0
LANE_WIDTH = 3.5
MAX_STEERING_ANGLE = 0.5  # radians
SUCCESS_TOLERANCE = 0.3
SUCCESS_TIME_REQUIRED = 5.0  # seconds


# Initialize LiDAR
lidar = driver.getDevice("lidar")
lidar.enable(timestep)


# Initialize GPS
gps = driver.getDevice("gps")
gps.enable(timestep)
# Noise settings
mean = 1           # mean of noise
std_dev = 1       # standard deviation in meters (adjust as needed)
def add_noise(value):
    noisy_value = value + random.gauss(mean, std_dev)
    return max(0.0, noisy_value)  # avoid invalid negative distances



# State definitions
CRUISE = 0
OVERTAKE = 1
FOLLOW = 2
VERIFY_OVERTAKE = 3
state = CRUISE
print("[CAR1]: Cruising!")


#PID controller variables
Kp = 0.05
Ki = 0.05           #VARIAR
Kd = 0.1
integral = 0.0
previous_error = 0.0


#coordinates from the center of the left lane (it does not properly align with the y axis)
lane_centers = [
    (-43.6912, 3.20022),
    (-68.3553, 3.20078),
    (-89.5582, 3.20117),
    (-144.177, 3.20101),
    (-207.044, 3.19843),
    (-279.784, 3.19225)
]


#calculate the expected left lane center, given the current x
def get_lane_center_y(x_current):
    if x_current <= lane_centers[0][0]:
        return lane_centers[0][1]
    elif x_current >= lane_centers[-1][0]:
        return lane_centers[-1][1]
    
    for i in range(len(lane_centers) - 1):
        x0, y0 = lane_centers[i]
        x1, y1 = lane_centers[i + 1]
        if x0 <= x_current <= x1:
            ratio = (x_current - x0) / (x1 - x0)
            return y0 + ratio * (y1 - y0)
    
    # Fallback — shouldn't reach here
    return lane_centers[-1][1]


#does not allow for too much steering!
def safe_steering(angle):
    return max(-MAX_STEERING_ANGLE, min(MAX_STEERING_ANGLE, angle))


#metrics
initial_y = gps.getValues()[1]      
maneuver_start_time = float('inf')
maneuver_end_time = float('inf')
maneuver_start_y = float('inf')
maneuver_start_velocity = float('inf')
maneuver_start_distance = float('inf')
min_distance = float('inf')
min_distance_raw = float('inf')
success = False

verify_start_time = None


#start simulation
start_time=None
driver.setCruisingSpeed(TARGET_SPEED)
while driver.step() != -1:
    

    if start_time is None:
        start_time = driver.getTime()
    if  driver.getTime() - start_time > 60:
        print("")
        print("[INFO]: OVERTAKE FAILED!")
        break

    if success == True:
        print("")
        print("[INFO]: OVERTAKE SUCCESSFUL!")
        print("[INFO]: Metrics:")
        print("[INFO]: Success = ", success)
        print("[INFO]: Maneuver Time = ", maneuver_end_time - maneuver_start_time)
        print("[INFO]: Minimum Distance = ", min_distance_raw)
        print("[INFO]: Maneuver Start Distance = ", maneuver_start_distance)
        print("[INFO]: Maneuver Start Velocity = ", maneuver_start_velocity)
        break


    ranges = lidar.getRangeImage()
    raw_ranges = ranges
    ranges = [add_noise(r) for r in ranges]
    front_distance = ranges[len(ranges) // 2]
    raw_front_distance = raw_ranges[len(ranges) // 2]


    if state == CRUISE:

        driver.setSteeringAngle(0.0)
        if front_distance < SAFE_DISTANCE:
            maneuver_start_time = driver.getTime()
            maneuver_start_y = gps.getValues()[1]
            maneuver_start_velocity = gps.getSpeed()
            maneuver_start_distance = front_distance
            state = OVERTAKE
            print("[CAR1]: Overtaking!")


    elif state == OVERTAKE:


        current_min = min(ranges)
        min_index = ranges.index(min(ranges))
        current_min_raw = raw_ranges[min_index]
        if current_min < min_distance:
            min_distance = current_min
        if current_min_raw < min_distance_raw:
            min_distance_raw = current_min_raw


        current_y = gps.getValues()[1]
        current_x = gps.getValues()[0]
        desired_y = get_lane_center_y(current_x)

        lateral_error = desired_y - current_y

        if abs(lateral_error) > 3:
            driver.setSteeringAngle(safe_steering(-0.25))
        else:
            driver.setSteeringAngle(0.1)
            for _ in range(10):
                driver.step()
            integral = 0.0
            previous_error = 0.0
            state = FOLLOW
            print("[CAR1]: Following!")



    elif state == FOLLOW:


        current_min = min(ranges)
        min_index = ranges.index(min(ranges))
        current_min_raw = raw_ranges[min_index]
        if current_min < min_distance:
            min_distance = current_min
        if current_min_raw < min_distance_raw:
            min_distance_raw = current_min_raw

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


        if abs(error) <= SUCCESS_TOLERANCE:
            if verify_start_time is None:
                verify_start_time = driver.getTime()
            elif driver.getTime() - verify_start_time >= SUCCESS_TIME_REQUIRED:
                maneuver_end_time = driver.getTime()
                success = True
        else:
            verify_start_time = None




with open("../supervisor_rulebased_nonoptimal1/results_rulebased_nonoptimal1.txt", "a") as file:
    file.write(f"{success},{maneuver_end_time - maneuver_start_time},{min_distance_raw},{maneuver_start_distance},{maneuver_start_velocity},\n")


with open("../supervisor_rulebased_nonoptimal1/results_rulebased_nonoptimal1.csv", "a") as file:
    file.write(f"{success},{maneuver_end_time - maneuver_start_time},{min_distance_raw},{maneuver_start_distance},{maneuver_start_velocity},\n")
