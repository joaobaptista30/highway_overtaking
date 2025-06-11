from controller import Supervisor
import os


supervisor = Supervisor()
timestep = int(supervisor.getBasicTimeStep())

#start position and rotations
start_pos_car1 = [-11.3445, 6.56001, 0.371196]
start_rot_car1 = [-0.00107476, -2.50665e-10, 0.999999, 3.14159]

car2_positions = [
    [-18.9263, 6.72982, 0.0877538],     #muito perto do car1 (dentro do lidar)
    [-29.1854, 6.73005, 0.173927],      #perto do car1 (na berma do lidar, mas fora do lidar)
    [-42.0248, 6.73016, 0.308511],      #distancia razoavel do car1
    [-68.6439, 6.73032, 0.587531],      #longe do car1
    ]
start_rot_car2 = [0.00524083, 8.04038e-05, 0.999986, 3.14158]


# Run 25 times for each CAR2 position
runs_per_position = 25
num_runs = len(car2_positions) * runs_per_position
run_number = 0


# Read current run number if it exists
if os.path.exists("run_number.txt"):
    with open("run_number.txt", "r") as f:
        run_number = int(f.read())

while run_number < num_runs:
    print(f"\n[SUPERVISOR]: Starting run {run_number+1}/{num_runs}")


    car2_index = run_number // runs_per_position
    start_pos_car2 = car2_positions[car2_index]
    print(start_pos_car2)

    # Retrieve car nodes by DEF name
    car1 = supervisor.getFromDef("CAR1")
    car2 = supervisor.getFromDef("CAR2")

    # Reset positions and rotations
    car1.getField("translation").setSFVec3f(start_pos_car1)
    car2.getField("translation").setSFVec3f(start_pos_car2)
    car1.getField("rotation").setSFRotation(start_rot_car1)
    car2.getField("rotation").setSFRotation(start_rot_car2)

    # Wait for run completion
    while supervisor.step(timestep) != -1:
        try:
            with open("results_rulebased_baseline.txt", "r") as file:
                lines = file.readlines()
                if len(lines) >= run_number + 1:
                    print(f"[SUPERVISOR]: Run {run_number+1} complete.")
                    break
        except FileNotFoundError:
            pass

    # Increment run number and save
    run_number += 1
    with open("run_number.txt", "w") as f:
        f.write(str(run_number))

    if run_number >= num_runs:
        print("[SUPERVISOR]: All runs completed.")
        supervisor.simulationSetMode(Supervisor.SIMULATION_MODE_PAUSE)
        break

    # Restart simulation and controllers
    #supervisor.simulationReset()
    # Reset physics
    supervisor.simulationResetPhysics()
    car1.restartController()
    car2.restartController()
