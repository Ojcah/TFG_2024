import gymnasium as gym
from learned_pahm import LearnedPAHM

env = learned_pahm(render_mode="human")
observation,info = env.reset(seed=123)
done=False
while not done:
    action = env.action_space.sample()
    observation,reward,terminated,truncated,info = env.step(action)
    env.render()
    done=terminated or truncated
env.close()
