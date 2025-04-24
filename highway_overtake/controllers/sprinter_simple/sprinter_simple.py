'''
    from vehicle import Driver

    driver = Driver()
    timestep = int(driver.getBasicTimeStep())

    # Set target speed (in m/s)
    target_speed = 30.0  # Feel free to adjust this

    # Main loop
    while driver.step() != -1:
        driver.setCruisingSpeed(target_speed)
'''

from vehicle import Driver
import math

# Initialize driver and sensors
driver = Driver()
timestep = int(driver.getBasicTimeStep())


# Setup Lidar
lidar = driver.getDevice('lidar')  # Make sure your Lidar is named 'lidar'
lidar.enable(timestep)
lidar.enablePointCloud()

# Vehicle control
MAX_SPEED = 60.0  # Max cruising speed
SAFE_DISTANCE = 5.0  # Trigger distance to start overtake (m)
OVERTAKE_DONE_DISTANCE = 3.0  # Distance when overtake is considered done
LANE_WIDTH = 3.5  # Approx lane width (meters)

# PID constants for lane change
Kp = 0.5
Ki = 0.01
Kd = 0.1

# PID control variables
integral = 0
previous_error = 0

# Flags
overtaking = False

i = -1

# Main loop
while driver.step() != -1:
    i += 1
    # Set default speed
    driver.setCruisingSpeed(MAX_SPEED)

    # Read Lidar
    lidar_values = lidar.getRangeImage()
    lidar_fov = lidar.getFov()
    lidar_res = lidar.getHorizontalResolution()
    num_rays = len(lidar_values)



    angle_increment = lidar_fov / num_rays

    # Find distance in front (middle rays)
    center_index = num_rays // 2
    front_distance = min(lidar_values[center_index - 10:center_index + 10])

    if i % 10 == 0:
        print(num_rays)
        print(front_distance)

    if not overtaking and front_distance < SAFE_DISTANCE:
        # Begin overtake
        overtaking = True

    if overtaking:
        # Simple PID lateral controller to steer to the left lane
        target_lateral_offset = LANE_WIDTH  # Want to be in left lane
        current_lateral_offset = 0  # Assuming car starts centered in middle lane
        error = target_lateral_offset - current_lateral_offset

        # PID calculations
        integral += error
        derivative = error - previous_error
        steering_angle = Kp * error + Ki * integral + Kd * derivative
        previous_error = error

        # Clamp steering angle
        steering_angle = max(-0.5, min(0.5, steering_angle))
        driver.setSteeringAngle(-steering_angle)

        # Check if overtaking is done
        if front_distance > OVERTAKE_DONE_DISTANCE:
            driver.setSteeringAngle(0.0)  # Hold in left lane

            overtaking = False
