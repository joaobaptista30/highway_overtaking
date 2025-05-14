import sys
import os
import numpy as np
from controller import Supervisor

try:
    import gymnasium as gym  # Using the newer gymnasium API
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_checker import check_env
except ImportError:
    sys.exit(
        'Please make sure you have all dependencies installed. '
        'Run: "pip3 install numpy gymnasium stable_baselines3"'
    )

# Check if Webots is in PATH
webots_home = os.environ.get('WEBOTS_HOME', 'C:/Program Files/Webots')
if not os.path.exists(f"{webots_home}/lib/controller"):
    print(f"Warning: WEBOTS_HOME directory structure doesn't seem right")
    print(f"Current WEBOTS_HOME: {webots_home}")
    print("Try setting the WEBOTS_HOME environment variable correctly")

'''
Found MercedesBenzSprinter, ID: 175
Found CitroenCZero, ID: 1492
'''


class OvertakeEnv(Supervisor, gym.Env):
    """Overtaking environment for Reinforcement Learning."""

    def __init__(self, max_episode_steps=500):
        """Initialize the environment.

        Args:
            max_episode_steps: Maximum number of steps per episode
        """
        super().__init__()

        # Environment parameters
        self.max_speed = 80.0  # Maximum speed in m/s
        self.target_speed = 60.0  # Target cruising speed in m/s
        self.front_safe_distance = 5.0  # Safe distance to front vehicle in m
        self.side_safe_distance = 3.0  # Safe distance to side obstacles in m
        self.lane_width = 3.5  # Approximate lane width in meters
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
            low=np.array([0.0] * self.n_lidar_segments + [-20.0, -1.0]),
            high=np.array([7.0] * self.n_lidar_segments + [20.0, 1.0]),
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

        # Get the front car node for position tracking
        self.front_car = self.getFromId(1492)  # CitroenCZero ID

        # Get self node for position tracking
        self.self_car = self.getSelf()

        # Print car information for debugging
        if self.self_car:
            print(f"Self car found: {self.self_car.getTypeName()}")
        if self.front_car:
            print(f"Front car found: {self.front_car.getTypeName()}")

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
        print(f"LiDAR initialized with resolution: {self.lidar_resolution}")

        # Get GPS (position) if available
        self.gps = self.getDevice("gps")
        self.gps.enable(self.timestep)
        print("GPS initialized successfully")

    def reset(self,seed = 0):
        """Reset the environment to initial state."""
        # Reset the simulation
        self.simulationResetPhysics()
        self.simulationReset()
        super().step(self.timestep)

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
        """Get the current observation from sensors."""
        # Process LiDAR data
        lidar_segments = []

        if self.lidar:
            try:
                lidar_data = self.lidar.getRangeImage()

                # Process the 180-degree field of view into segments
                segment_size = len(lidar_data) // self.n_lidar_segments

                for i in range(self.n_lidar_segments):
                    start_idx = i * segment_size
                    end_idx = (i + 1) * segment_size
                    segment_data = lidar_data[start_idx:end_idx]
                    # Use minimum distance in each segment (most conservative)
                    min_distance = min(segment_data) if segment_data else 7.0
                    lidar_segments.append(min_distance)
            except Exception as e:
                print(f"Error getting LiDAR data: {e}")
                # Use dummy data if LiDAR fails
                lidar_segments = [7.0] * self.n_lidar_segments
        else:
            # Use dummy data if no LiDAR
            lidar_segments = [7.0] * self.n_lidar_segments

        # Get lane position (-1: right lane, 0: center, 1: left lane)
        # This is a simplified measure - in a real system you would use more sophisticated lane detection
        lane_pos = 0.0
        rel_speed = 0.0

        if self.self_car and self.front_car:
            try:
                self_pos = self.self_car.getPosition()
                lane_pos = self_pos[1] / self.lane_width

                # Calculate relative speed (simplified)
                self_vel = self.self_car.getVelocity()
                front_car_pos = self.front_car.getPosition()
                front_car_vel = self.front_car.getVelocity()

                # Relative speed in the X direction (forward)
                rel_speed = self_vel[0] - front_car_vel[0]
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
                self_vel = self.self_car.getVelocity()

                # Check if we have overtaken
                if (self_pos[0] > front_car_pos[0] + 5.0 and  # Passed by 5 meters
                        self_pos[1] < 0.0):  # In the left lane
                    if not self.has_overtaken:
                        reward += self.overtake_reward
                        self.has_overtaken = True
                        info['overtake'] = True
                        done = True

                # Check for collision using LiDAR
                min_distance = 7.0  # Default value
                if self.lidar:
                    try:
                        lidar_data = self.lidar.getRangeImage()
                        min_distance = min(lidar_data) if lidar_data else 7.0
                    except Exception as e:
                        print(f"Error reading LiDAR: {e}")

                if min_distance < 0.5:  # Too close to an obstacle
                    reward += self.crash_penalty
                    done = True
                    info['crash'] = True

                # Reward for maintaining speed
                speed = np.linalg.norm(self_vel)
                speed_reward = -abs(speed - self.target_speed) * self.speed_reward_factor
                reward += speed_reward

                # Reward for being in the left lane during overtaking
                if self_pos[1] < 0.0 and not self.has_overtaken:
                    lane_reward = 0.5
                    reward += lane_reward
                    self.steps_overtaking += 1

                # Penalize for taking too long to overtake
                if self.steps_overtaking > 200 and not self.has_overtaken:
                    reward -= 50.0
                    done = True
                    info['timeout'] = True
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
        # Convert action to steering angle
        if action == 0:  # Steer left
            steering_angle = -0.2
        elif action == 1:  # Maintain course
            steering_angle = 0.0
        else:  # Steer right
            steering_angle = 0.2

        # Apply controls using the driver API
        try:
            # Set cruising speed
            self.self_car.setCruisingSpeed(self.target_speed)

            # Set steering angle
            self.self_car.setSteeringAngle(steering_angle)
        except Exception as e:
            print(f"Error applying vehicle controls: {e}")

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

    try:
        # Check environment compatibility
        check_env(env)

        # Create and train the model
        model = PPO("MlpPolicy", env, verbose=1,
                    learning_rate=0.0003,
                    n_steps=2048,
                    batch_size=64,
                    n_epochs=10,
                    gamma=0.99)

        # Train for 100,000 timesteps
        model.learn(total_timesteps=100000)

        # Save the model
        model.save("overtake_ppo_model")

        return model
    except Exception as e:
        print(f"Error during training: {e}")
        return None


def evaluate_model(model, env, num_episodes=10):
    """Evaluate the trained model."""
    print("\nEvaluating model performance...")

    if model is None:
        print("No model to evaluate.")
        return

    total_rewards = 0
    successes = 0

    for ep in range(num_episodes):
        obs = env.reset()
        done = False
        ep_reward = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
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

    # Check if we should load an existing model
    model = None
    try:
        print("Looking for existing model...")
        model = PPO.load("overtake_ppo_model")
        print("Existing model loaded!")
        train_new = False
    except Exception as e:
        print(f"No existing model found ({e}). Will train a new one.")
        train_new = True

    if train_new:
        print("Starting training...")
        model = train_model(env)
        if model:
            print("Training complete!")
        else:
            print("Training failed.")

    # Evaluate model performance
    if model:
        evaluate_model(model, env)

        # Run demo
        demo_model(model, env)
    else:
        print("No model to evaluate or demo.")


if __name__ == "__main__":
    main()
