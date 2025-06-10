import sys
from controller import Supervisor # Supervisor object from webots software

try:
    import numpy as np
    import gymnasium as gym
    
except ImportError:
    sys.exit(
        'Please make sure you have all dependencies installed. '
        'Run: "pip3 install numpy gymnasium"'
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
        self.last_action = None
        # Actions: [steer_left, maintain_course, steer_right]
        self.action_space = gym.spaces.Discrete(3)

        # Enhanced observations:
        # We'll use 10 LiDAR segments + relative speed + lane position
        self.n_lidar_segments = 10
        self.observation_space = gym.spaces.Box(
            low=np.array([0.0] * self.n_lidar_segments + [-200.0, -1.0]),
            high=np.array([12.0] * self.n_lidar_segments + [200.0, 1.0]),
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
        """Initialize robot devices: Lidar, gps"""

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
            min_distance = min(segment_data) if segment_data else 12.0
            lidar_segments.append(min(min_distance,12.0)) # 12.0 is the range of Lidar


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
        """Calculate the reward based on current state, only allowing overtaking
           when front distance < 4m, and shaping rewards to straighten out, staying in the most left lane."""

        reward = 0.0
        done = False
        info = {'overtake': False, 'crash': False, 'timeout': False}

        contacts = self.self_car.getContactPoints()
        if contacts:
            reward += self.crash_penalty
            done = True
            info['crash'] = True

            return float(reward), done, info

        self_pos = self.self_car.getPosition()
        front_pos = self.front_car.getPosition()

        # get lidar front distance (middle segment)
        lidar_data = self.lidar.getRangeImage()
        segs = len(lidar_data) // self.n_lidar_segments
        mid_idx = self.n_lidar_segments // 2
        front_segment = lidar_data[(mid_idx-1)*segs:(mid_idx+1)*segs]
        front_dist = min(front_segment) if front_segment else 7.0
        front_dist = min(front_dist, 7.0)

        # current lane_pos: -1 right, 0 center, +1 left
        lane_pos = self._get_observation()[-1]

        # a) speed reward for staying at target speed
        reward += self.speed_reward_factor * self.target_speed

        # b) crash penalty
        if front_dist < 0.3:
            reward += self.crash_penalty
            done = True
            info['crash'] = True

            return float(reward), done, info

        if self_pos[1] > 8.0 or self_pos[1] < 2.0:
            # hit barrier (out of three lanes)
            reward += self.crash_penalty
            done = True
            info['crash'] = True

            return float(reward), done, info

        # only start rewarding lane change if we’re close enough
        if front_dist < 4:
            # encourage move toward left lane
            reward += 20.0 * lane_pos
            # if we've passed the front car by 5m
            if abs(self_pos[0]) > abs(front_pos[0]) + 5.0 and not self.has_overtaken:
                reward += self.overtake_reward
                self.has_overtaken = True
                info['overtake'] = True
                done = True

                return float(reward), done, info
        else:
            # if not in center lane when no overtake needed, penalize
            reward -= 5.0 * abs(lane_pos)
            # if steering left/right unnecessarily, penalize more
            last_action = getattr(self, 'last_action', 1)
            if last_action != 1:
                reward -= 2.0

        # 4) once in left lane, encourage straightening
        if lane_pos > 0.5:
            # reward being near the ideal left line
            reward += 10.0 * (1.0 - abs(1.0 - lane_pos))

        # 5) timeout penalty
        self.steps_overtaking = self.steps_overtaking + 1 if front_dist < 4 else 0
        if self.steps_overtaking > 500 and not self.has_overtaken:
            reward += -50.0
            info['timeout'] = True
            done = True

            return float(reward), done, info

        if abs(self_pos[0]) > (abs(front_pos[0]) + 20.0):
            reward += self.overtake_reward
            self.has_overtaken = True
            info['overtake'] = True
            done = True

            return float(reward), done, info

        # episode step limit
        self.episode_steps += 1
        if self.episode_steps >= self.spec.max_episode_steps:
            info['timeout'] = True
            done = True

        return float(reward), done, info

    def step(self, action):
        """Execute action and return new state, reward, done, and info."""
        ''' MOVE OVERTAKING '''
        # remember action for reward-shaping
        self.last_action = action

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

            # Set steering angle
            left_steer.setPosition(steering_angle)
            right_steer.setPosition(steering_angle)
        except Exception as e:
            print(f"Error applying vehicle controls: {e}")

        ''' MOVE FRONT CAR '''
        # angular_velocity (rad/s) = linear_velocity (m/s) / wheel_radius (m)
        self.front_car.setVelocity([-(self.target_speed-20)/10, 0.0, 0.0])

        # Execute simulation step
        super().step(self.timestep)

        # Get new observation
        new_obs = self._get_observation()

        # Calculate reward
        reward, done, info = self._get_reward()

        return new_obs, reward, done, False, info  # The False is for truncated (gymnasium API)
