import gymnasium as gym

env = gym.make("Pong", render_mode="human")
observation, info = env.reset()

for i in range(1000):
    print(f"Step {i}")
    action = env.action_space.sample()  # This line chooses a random action
    observation, reward, terminated, truncated, info = env.step(action)
    
    if terminated or truncated:
        print("Episode finished")
        observation, info = env.reset()

env.close()