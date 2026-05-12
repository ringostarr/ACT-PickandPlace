import pybullet
import gymnasium as gym
import panda_gym

print(gym.__version__)
print("pybullet imported properly!\n")
env = gym.make("PandaPickAndPlace-v3", render_mode="human", control_type="joints")
obs, info = env.reset()
print("obs keys:", list(obs.keys()))
print("OK")

import time
for _ in range(200):
    a = env.action_space.sample()
    obs, r, term, trunc, info = env.step(a)
    if term or trunc:
        env.reset()
    time.sleep(0.01)
env.close()