from vehicle import Driver
import random

driver = Driver()
timestep = int(driver.getBasicTimeStep())  # in ms

# Set constant cruising speed
constant_speed = 45.0  # m/s
driver.setCruisingSpeed(constant_speed)

# Brake setup
brake_intensity = 1.0  # full brake (range: 0.0 to 1.0)
brake_duration = 1000   # ms
brake_timer = 0        # counter for braking time left

while driver.step() != -1:

    if brake_timer > 0:
        # Apply brakes
        driver.setBrakeIntensity(brake_intensity)
        brake_timer -= timestep

    else:
        # Release brakes
        driver.setBrakeIntensity(0.0)
        # Random chance to brake (e.g., 2% chance per timestep)
        if random.random() < 0.01:
            print("[CAR2]: Braking!")
            brake_timer = brake_duration
