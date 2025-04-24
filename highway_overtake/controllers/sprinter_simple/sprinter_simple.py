from vehicle import Driver

driver = Driver()
timestep = int(driver.getBasicTimeStep())

# Set target speed (in m/s)
target_speed = 30.0  # Feel free to adjust this

# Main loop
while driver.step() != -1:
    driver.setCruisingSpeed(target_speed)