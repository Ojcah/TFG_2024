
import argparse
import gymnasium as gym
from PAHM.learned_pahm import LearnedPAHM
import numpy as np
import sys

import torch
import PPO.test_ppo_pahm as testModel
from torch import nn
import torch.nn.functional as F

parser = argparse.ArgumentParser()

parser.add_argument('--actor_name', type=str, default='./actor-critic/ppo_actor.pth', dest='actor_name')
parser.add_argument('--target_angle', type=int, default=60, dest='target_angle')

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu" if torch.cuda.is_available() else "cuda")


## **************************************************************************************
## **************************************************************************************
"""
	This file contains a neural network module for us to
	define our actor and critic networks in PPO.
"""
class FeedForwardNN(nn.Module):
	"""
		A standard in_dim-64-64-out_dim Feed Forward Neural Network.
	"""
	def __init__(self, in_dim, out_dim):
		"""
			Initialize the network and set up the layers.

			Parameters:
				in_dim - input dimensions as an int
				out_dim - output dimensions as an int

			Return:
				None
		"""
		super(FeedForwardNN, self).__init__()

		self.layer1 = nn.Linear(in_dim, 64)
		self.layer2 = nn.Linear(64, 128)
		self.layer3 = nn.Linear(128, 128)
		self.layer4 = nn.Linear(128, 64)
		self.layer5 = nn.Linear(64, out_dim)
		

	def forward(self, obs):
		"""
			Runs a forward pass on the neural network.

			Parameters:
				obs - observation to pass as input

			Return:
				output - the output of our forward pass
		"""
		# Convert observation to tensor if it's a numpy array
		if isinstance(obs, np.ndarray):
			obs = torch.tensor(obs, dtype=torch.float, device=device)

		activation1 = F.relu(self.layer1(obs))
		activation2 = F.relu(self.layer2(activation1))
		activation3 = F.relu(self.layer3(activation2))
		activation4 = F.relu(self.layer4(activation3))
		output = self.layer5(activation4)

		return output
## **************************************************************************************
## **************************************************************************************
def test(env, target_angle, actor_model):
	"""
		Tests the model.

		Parameters:
			env - the environment to test the policy on
			actor_model - the actor model to load in

		Return:
			None
	"""
	print(f"Testing {actor_model}", flush=True)

	# If the actor model is not specified, then exit
	if actor_model == '':
		print(f"Didn't specify model file. Exiting.", flush=True)
		sys.exit(0)

	# Extract out dimensions of observation and action spaces
	obs_dim = env.observation_space.shape[0] + 3
	act_dim = env.action_space.shape[0]

	# Build our policy the same way we build our actor model in PPO
	policy = FeedForwardNN(obs_dim, act_dim).to(device)

	# Load in the actor model saved by the PPO algorithm
	policy.load_state_dict(torch.load(actor_model))

	# Evaluate our policy with a separate module, eval_policy, to demonstrate
	# that once we are done training the model/policy with ppo.py, we no longer need
	# ppo.py since it only contains the training algorithm. The model/policy itself exists
	# independently as a binary file that can be loaded in with torch.
	testModel.eval_policy(policy=policy, env=env, render=True, target_angle=target_angle)
	
## **************************************************************************************
## **************************************************************************************

env = LearnedPAHM(render_mode="human")

test(env=env, target_angle=args.target_angle, actor_model=args.actor_name)


env.close()
