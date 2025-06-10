import sys
import os
from discrete_env import OvertakeEnv

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_checker import check_env
    import torch
except ImportError:
    sys.exit(
        'Please make sure you have all dependencies installed. '
        'Run: "pip3 install stable_baselines3"'
    )

"""Train the reinforcement learning model
   Saving a model every 100k steps"""

# Check for GPU and set device to CUDA if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


try:
    env = OvertakeEnv()


    algorithm_name = "PPO"
    policy = "MlpPolicy"
    print(f"Starting training for {algorithm_name} with {policy}")


    # Create model directory
    model_dir = f"discrete_models"
    os.makedirs(model_dir, exist_ok=True)

    check_env(env)
    env.reset()

    model = PPO(policy, env, verbose=1, device=device)

    # Training parameters (1 million total steps)
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