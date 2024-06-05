
import argparse
import gymnasium as gym
from PAHM.learned_pahm import LearnedPAHM
import numpy as np
import sys

import torch
from PPO.ppo_pahm import PPO
from torch import nn
from datetime import datetime
import torch.nn.functional as F

import wandb

try:
  from google.colab import drive
  plataform = "_google"
except:
  plataform = ""

today = datetime.today()
today = today.strftime("%y%m%d_%H%M")

parser = argparse.ArgumentParser()

parser.add_argument('--run_name', type=str, default=('run_'+today+plataform), dest='run_name')
parser.add_argument('--run_id', type=str, default=('run_v1_'+today+plataform), dest='run_id')
parser.add_argument('--description', type=str, default='', dest='description')	

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
def train(env, hyperparameters, actor_model, critic_model):
	"""
		Trains the model.

		Parameters:
			env - the environment to train on
			hyperparameters - a dict of hyperparameters to use, defined in main
			actor_model - the actor model to load in if we want to continue training
			critic_model - the critic model to load in if we want to continue training

		Return:
			None
	"""	
	print(f"Training", flush=True)

	# Create a model for PPO.
	model = PPO(policy_class=FeedForwardNN, env=env, **hyperparameters)

	# Tries to load in an existing actor/critic model to continue training on
	if actor_model != '' and critic_model != '':
		print(f"Loading in {actor_model} and {critic_model}...", flush=True)
		model.actor.load_state_dict(torch.load(actor_model))
		model.critic.load_state_dict(torch.load(critic_model))
		print(f"Successfully loaded.", flush=True)
	elif actor_model != '' or critic_model != '': # Don't train from scratch if user accidentally forgets actor/critic model
		print(f"Error: Either specify both actor/critic models or none at all. We don't want to accidentally override anything!")
		sys.exit(0)
	else:
		print(f"Training from scratch.", flush=True)

	# Train the PPO model with a specified total timesteps
	# NOTE: You can change the total timesteps here, I put a big number just because
	# you can kill the process whenever you feel like PPO is converging
	#model.learn(total_timesteps=200_000_000, target_angle=target_angle)
	#model.learn(total_timesteps=1_000_000, target_angle=target_angle)
	model.learn(total_timesteps=wandb.config['total_timesteps'])
	
## **************************************************************************************
## **************************************************************************************

wandb.login(key="0005da299924ab3d8473fa6a5f120b46a82a6a7a")


wandb.init(project = "PAMH_PPO", 
           name = args.run_name,
           #resume = 'Allow',
		   #monitor_gym=True,
		   reinit=True,
           id = args.run_id,
		   notes="""Timesteps: ### || Target angle: ### || Change angle: ###
		   || """
		   )

wandb.config = {
    'timesteps_per_batch': 2048, 
	'max_timesteps_per_episode': 200, 
	'gamma': 0.99, 
	'n_updates_per_iteration': 10,
	'lr': 1e-4, 
	'clip': 0.2,
	'render': True,
	'render_every_i': 10,
	# *****************
	'total_timesteps': 1_000_000,
	'target_angle': 45,
	'change_angle': False,
	'change_dev_std': False
}

wandb.run.notes = f"""Timesteps: {wandb.config['total_timesteps']} || Target angle: {wandb.config['target_angle']} || Change angle: {wandb.config['change_angle']} || """ + args.description

env = LearnedPAHM(render_mode="human")
#env = LearnedPAHM(render_mode="rgb_array")

train(env=env, hyperparameters=wandb.config, actor_model='', critic_model='')



wandb.finish()
env.close()
