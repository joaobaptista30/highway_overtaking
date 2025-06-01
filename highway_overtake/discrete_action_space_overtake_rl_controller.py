import sys
import os

from torch.onnx.symbolic_opset9 import contiguous

from controller import Supervisor

try:
    import numpy as np
    import gymnasium as gym
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_checker import check_env
    import torch
except ImportError:
    sys.exit(
        'Please make sure you have all dependencies installed. '
        'Run: "pip3 install numpy gymnasium stable_baselines3"'
    )

'''
Found MercedesBenzSprinter, ID: 175
Found CitroenCZero, ID: 1492 | DEF: FRONT_CAR
'''


class OvertakeEnv(Supervisor, gym.Env):
    """Overtaking environment for Reinforcement Learning."""

    def __init__(self, max_episode_steps=5000000):
        """Initialize the environment.

        Args:
            max_episode_steps: Maximum number of steps per episode
        """
        super().__init__()

        # Environment parameters
        self.max_speed = 80.0  # Maximum speed in m/s
        self.target_speed = 60.0  # Target cruising speed in m/s
        self.front_safe_distance = 5.0  # Safe distance to front vehicle in m
        self.timestep = int(self.getBasicTimeStep())
        self.overtake_reward = 100.0  # Reward for successful overtaking
        self.crash_penalty = -100.0  # Penalty for crashing
        self.speed_reward_factor = 0.1  # Reward factor for maintaining speed

        # Define action and observation spaces
        # Actions: [steer_left, maintain_course, steer_right]
        self.action_space = gym.spaces.Discrete(3)

        # Observations:
        # We'll use 10 LiDAR segments + relative speed + lane position
        self.n_lidar_segments = 10
        self.observation_space = gym.spaces.Box(
            low=np.array([0.0] * self.n_lidar_segments + [-200.0, -1.0]),
            high=np.array([7.0] * self.n_lidar_segments + [200.0, 1.0]),
            dtype=np.float32
        )

        # Environment spec for compatibility with gym
        self.spec = gym.envs.registration.EnvSpec(
            id='OvertakeEnv-v0',
            max_episode_steps=max_episode_steps
        )

        # State variables
        self.state = None
        self.steps_overtaking = 0
        self.has_overtaken = False
        self.episode_steps = 0

        # Get self node for position tracking
        self.self_car = self.getSelf()

        # Get the front car node for position tracking
        self.front_car = self.getFromDef("FRONT_CAR")  # CitroenCZero DEF

        # Initialize devices
        self._init_devices()

        # Tools
        self.keyboard = self.getKeyboard()
        self.keyboard.enable(self.timestep)

    def _init_devices(self):
        """Initialize robot devices: sensors, etc."""

        # Initialize LiDAR
        self.lidar = self.getDevice("lidar")

        self.lidar.enable(self.timestep)
        self.lidar_resolution = self.lidar.getHorizontalResolution()
        #print(f"LiDAR initialized with resolution: {self.lidar_resolution}")

        # Get GPS (position)
        self.gps = self.getDevice("gps")
        self.gps.enable(self.timestep)
        #print("GPS initialized successfully")

    def reset(self,seed = 0):
        """Reset the environment to initial state."""
        # Reset the simulation
        self.simulationResetPhysics()
        self.simulationReset()
        super().step(self.timestep)

        # TODO
        # podemos fazer um codigo para o carro da frente ser posicionado num local aleatorio
        # com base na posição do mapa

        # Reset state variables
        self.steps_overtaking = 0
        self.has_overtaken = False
        self.episode_steps = 0

        # Initialize devices
        self._init_devices()

        # Run one step to get initial sensor readings
        super().step(self.timestep)

        info = {}
        info['timeout'] = False
        info['overtake'] = False
        info['crash'] = False
        info['error'] = False

        # Get initial observation
        return self._get_observation(), info

    def _get_observation(self):
        """Get the current observation from sensors.
           # Observations:
           # We'll use 10 LiDAR segments + relative speed + lane position
        """

        # Process LiDAR data
        lidar_segments = []
        lidar_data = self.lidar.getRangeImage()

        # Process the 180-degree field of view into segments
        segment_size = len(lidar_data) // self.n_lidar_segments

        for i in range(self.n_lidar_segments):
            start_idx = i * segment_size
            end_idx = (i + 1) * segment_size
            segment_data = lidar_data[start_idx:end_idx]
            # Use minimum distance in each segment (most conservative)
            min_distance = min(segment_data) if segment_data else 7.0
            lidar_segments.append(min(min_distance,7.0)) # 7.0 is the range of Lidar


        # Get lane position (-1: right lane, 0: center, 1: left lane)
        # self_pos[1]: (y values) 12.5 - 8.5  |  8.4 - 5 |  5 - 2
        # This is a simplified measure - in a real system you would use more sophisticated lane detection
        lane_pos = 0.0
        rel_speed = 0.0

        if self.self_car and self.front_car:
            try:
                self_pos = self.self_car.getPosition()
                if 12.5 <= self_pos[1] < 8.5: lane_pos = -1.0 # right lane
                elif 8.5 <= self_pos[1] < 5: lane_pos = 0.0 # center lane
                elif 5 <= self_pos[1] < 2: lane_pos = 1.0 # left lane

                # Calculate relative speed (simplified)
                self_vel = self.self_car.getVelocity()
                front_car_vel = self.front_car.getVelocity()

                # Relative speed in the X direction (forward)
                rel_speed = -(self_vel[0] - front_car_vel[0]) # due to map config, car are moving on the -X axis
            except Exception as e:
                print(f"Error calculating position/velocity: {e}")

        # Combine observations
        observation = np.array(lidar_segments + [rel_speed, lane_pos], dtype=np.float32)

        return  observation

    def _get_reward(self):
        """Calculate the reward based on current state."""
        reward = 0
        done = False
        info = {}

        if self.self_car and self.front_car:
            try:
                self_pos = self.self_car.getPosition()
                front_car_pos = self.front_car.getPosition()


                # in the left lane during overtaking
                if self_pos[1] < 4.3:
                    # Check if we have overtaken
                    if abs(self_pos[0]) > abs(front_car_pos[0]) + 5.0: # Passed by 5 meter
                        if not self.has_overtaken:
                            reward += self.overtake_reward
                            self.has_overtaken = True
                            info['overtake'] = True
                            done = True

                    if not self.has_overtaken:
                        reward += 30
                        self.steps_overtaking += 1

                # Check for collision using LiDAR
                if self.lidar:
                    try:
                        lidar_data = self.lidar.getRangeImage()
                        min_distance = min(lidar_data)

                        if min_distance < 0.5:  # Too close to an obstacle
                            reward += self.crash_penalty
                            done = True
                            info['crash'] = True

                    except Exception as e:
                        print(f"Error reading LiDAR: {e}")


                # end simulation for being in the right lane
                if self_pos[1] > 8.0:
                    reward += self.crash_penalty
                    info['overtake'] = False
                    info['crash'] = True
                    done = True
                        
                # Penalize for taking too long to overtake
                if self.steps_overtaking > 200 and not self.has_overtaken:
                    reward -= 50.0
                    info['timeout'] = True
                    done = True


            except Exception as e:
                print(f"Error calculating reward: {e}")
                reward = 0
                done = True
                info['error'] = True

        # End episode if max steps reached
        self.episode_steps += 1
        if self.episode_steps >= self.spec.max_episode_steps:
            done = True
            info['timeout'] = True

        return reward, done, info

    def step(self, action):
        """Execute action and return new state, reward, done, and info."""
        ''' MOVE OVERTAKING '''
        # Convert action to steering angle
        if action == 0:  # Steer left
            steering_angle = -0.2
        elif action == 1:  # Maintain course
            steering_angle = 0.0
        else:  # Steer right
            steering_angle = 0.2

        # Get devices
        # speed control
        left_drive = self.getDevice("left_front_wheel")
        right_drive = self.getDevice("right_front_wheel")
        # wheel ang control
        left_steer = self.getDevice("left_steer")
        right_steer = self.getDevice("right_steer")

        # Set drive wheels to velocity mode
        left_drive.setPosition(float('inf'))
        right_drive.setPosition(float('inf'))

        try:
            # Set cruising speed
            left_drive.setVelocity(self.target_speed)
            right_drive.setVelocity(self.target_speed)

            #self.self_car.setCruisingSpeed(self.target_speed)

            # Set steering angle
            left_steer.setPosition(steering_angle)
            right_steer.setPosition(steering_angle)
            # self.self_car.setSteeringAngle(steering_angle)
        except Exception as e:
            print(f"Error applying vehicle controls: {e}")

        ''' MOVE FRONT CAR '''
        # angular_velocity (rad/s) = linear_velocity (m/s) / wheel_radius (m)
        wheel_radius = 0.28475
        speed_desired = (self.target_speed - 15) / wheel_radius
        # self.front_car.setVelocity([speed_desired,0.0])
        self.front_car.setVelocity([-(self.target_speed-20)/10, 0.0, 0.0])

        # Execute simulation step
        super().step(self.timestep)

        # Get new observation
        new_obs = self._get_observation()

        # Calculate reward
        reward, done, info = self._get_reward()

        return new_obs, reward, done, False, info  # The False is for truncated (gymnasium API)

    def wait_keyboard(self):
        """Wait for keyboard input."""
        print("Press 'Y' to start...")
        while self.keyboard.getKey() != ord('Y'):
            super().step(self.timestep)



def train_model(env: OvertakeEnv):
    """Train the reinforcement learning model."""

    # Check for GPU and set device to CUDA if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    try:
        algorithm_name = "PPO"
        policy = "MlpPolicy"
        print(f"Starting training for {algorithm_name} with {policy}")

        # Create model directory
        model_dir = f"models/{algorithm_name}/{policy}"
        os.makedirs(model_dir, exist_ok=True)

        check_env(env)
        env.reset()

        model = PPO(policy, env, verbose=1, device=device)

        # Training parameters
        timesteps = 100000
        num_iterations = 10

        for i in range(1, num_iterations + 1):
            print(f"Training iteration {i}/{num_iterations} for {algorithm_name} with {policy}")
            model.learn(total_timesteps=timesteps, reset_num_timesteps=False, progress_bar=True)

            # Save model
            model.save(f"{model_dir}/{timesteps * i}")

        print(f"Completed training for {algorithm_name} with {policy}")

    except Exception as e:
        print(f"Error during training: {e}")


def evaluate_model(model, env, num_episodes=10):
    """Evaluate the trained model."""
    print("\nEvaluating model performance...")

    if model is None:
        print("No model to evaluate.")
        return

    total_rewards = 0
    successes = 0

    for ep in range(num_episodes):
        obs, _ = env.reset()
        done = False
        ep_reward = 0

        while not done:
            action, _ = model.predict(obs)

            obs, reward, done, _, info = env.step(action)
            ep_reward += reward

            if info.get('overtake', False):
                successes += 1

        total_rewards += ep_reward
        print(f"Episode {ep + 1}: Reward = {ep_reward}")

    avg_reward = total_rewards / num_episodes
    success_rate = successes / num_episodes * 100

    print(f"\nEvaluation Results:")
    print(f"Average Reward: {avg_reward}")
    print(f"Success Rate: {success_rate}%")


def demo_model(model, env):
    """Run a demo of the trained model."""
    print("\nRunning model demo. Press 'Y' to start...")

    if model is None:
        print("No model to demo.")
        return

    env.wait_keyboard()

    obs = env.reset()
    done = False
    total_reward = 0

    print("Demo started! Watch the simulation...")

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, info = env.step(action)
        total_reward += reward

        if info.get('crash', False):
            print("Crash detected!")
        elif info.get('overtake', False):
            print("Successful overtake!")
        elif info.get('timeout', False):
            print("Episode timeout.")

    print(f"Demo finished with total reward: {total_reward}")


def main():
    """Main function."""

    print("Overtaking RL controller initialized")

    # Create environment
    env = OvertakeEnv()

    train_new = True

    if train_new:
        train_model(env)

    else:
        model = None
        try:
            # steps = 100000 # replace with model version
            steps = "primeiro_treino" # replace with model version

            model_dir = f"models/PPO/MlpPolicy/{steps}.zip"
            print("Looking for existing model...")
            model = PPO.load(model_dir, device="cpu")
            print("Existing model loaded!")
        except Exception as e:
            print(f"No existing model found ({e}). Will train a new one.")


        # Evaluate model performance
        if model:
            evaluate_model(model, env)

            # Run demo
            demo_model(model, env)
        else:
            print("No model to evaluate or demo.")


if __name__ == "__main__":
    main()
