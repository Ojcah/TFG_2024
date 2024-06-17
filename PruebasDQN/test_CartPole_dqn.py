import gymnasium as gym
import math
import matplotlib
import matplotlib.pyplot as plt
from itertools import count

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# *******************************************************************************************
# *******************************************************************************************
# *******************************************************************************************

#env = gym.make("CartPole-v1")
env = gym.make("CartPole-v1", render_mode="human")

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

# if GPU is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(torch.cuda.is_available())

# *******************************************************************************************
# *******************************************************************************************
# *******************************************************************************************


class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)
    

# *******************************************************************************************
# *******************************************************************************************
# *******************************************************************************************

# BATCH_SIZE is the number of transitions sampled from the replay buffer
# GAMMA is the discount factor as mentioned in the previous section
# EPS_START is the starting value of epsilon
# EPS_END is the final value of epsilon
# EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
# TAU is the update rate of the target network
# LR is the learning rate of the ``AdamW`` optimizer
BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4

# Get number of actions from gym action space
n_actions = env.action_space.n
# Get the number of state observations
state, info = env.reset()
n_observations = len(state)

policy_net = DQN(n_observations, n_actions).to(device)
policy_net.load_state_dict(torch.load("./models/CartPole_1000eps.pth"))

steps_done = 0
episode_durations = []


def plot_durations(show_result=False):
    plt.figure(1)
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)  
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())

# *******************************************************************************************
# *******************************************************************************************
# *******************************************************************************************

if torch.cuda.is_available():
    #num_episodes = 600
    num_episodes = 30
else:
    #num_episodes = 50
    num_episodes = 20
    
for i_episode in range(num_episodes):
    # Initialize the environment and get its state
    state, info = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    print("Episode >> ", i_episode)
    for t in count():
    # for t in range(1000):
        with torch.no_grad():
            action = policy_net(state).max(1).indices.view(1, 1)
        state, reward, terminated, truncated, _ = env.step(action.item())
        reward = torch.tensor([reward], device=device)
        done = terminated or truncated

        ## *********************************************************************************
        ## *********************************************************************************
        pole_angle = math.degrees(state[2])

        print(np.array([pole_angle, state[0], action.item(), reward.item()]),end="\r")

        ## *********************************************************************************
        ## *********************************************************************************
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        if done:
            episode_durations.append(t + 1)
            #plot_durations()
            break

print('Complete')
#plot_durations(show_result=True)
#plt.ioff()
#plt.show()
env.close()


