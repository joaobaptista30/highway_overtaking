import sys
from continuous_env import OvertakeEnv

try:
    from stable_baselines3 import PPO
except ImportError:
    sys.exit(
        'Please make sure you have all dependencies installed. '
        'Run: "pip3 install stable_baselines3"'
    )

"""Evaluate the trained model."""

try:
    print("\nEvaluating model performance...")
    env = OvertakeEnv()

    steps = 700000 # model number steps to evaluate
    model_dir = f"/continuous_models/{steps}.zip"
    print("Looking for existing model...")

    model = PPO.load(model_dir, device="cpu")
    print("Existing model loaded!")


    num_episodes = 25
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



except Exception as e:
    print(f"No existing model found ({e}).")