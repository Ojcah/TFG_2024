
import gymnasium as gym
import random
import sys
import wandb
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

from datetime import datetime
from PAHM.learned_pahm import LearnedPAHM
from DQN.dqn_discrete_pahm import DQN


today = datetime.today()
today = today.strftime("%y%m%d_%H%M")

parser = argparse.ArgumentParser()

parser.add_argument('--run_name', type=str, default=('run_'+today), dest='run_name')
parser.add_argument('--run_id', type=str, default=('run_v1_'+today), dest='run_id')
parser.add_argument('--description', type=str, default='', dest='description')	
	

args = parser.parse_args()


wandb.login(key="0005da299924ab3d8473fa6a5f120b46a82a6a7a")


wandb.init(project = "PAMH_DQN", 
           name = args.run_name,
           #resume = 'Allow',
		   #monitor_gym=True,
           reinit=True,
           id = args.run_id,
           notes="""Timesteps: ### || Target angle: ### || Change angle: ###
		   || """
           )

wandb.config = {
    'batch_size': 128,  # number of transitions sampled from the replay buffer
	'gamma': 0.99,      # discount factor as mentioned in the previous section
    'eps_start': 0.9,   # starting value of epsilon
    'eps_end': 0.05,    # final value of epsilon
    'eps_decay': 1000,  # controls the rate of exponential decay of epsilon, higher means a slower decay
    'tau': 0.005,       # update rate of the target network
    'lr': 1e-4,         # learning rate of the ``AdamW`` optimizer
	# *****************
	'num_episodes': 10,
	'target_angle': 45,
	'change_angle': False,
    'num_intervals': 10
}
wandb.run.notes = f"""Timesteps: {wandb.config['num_episodes']} || Target angle: {wandb.config['target_angle']} || Change angle: {wandb.config['change_angle']} || """ + args.description


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FeedForwardNN(nn.Module):
    """
		A standard in_dim-128-128-out_dim Feed Forward Neural Network.
	"""
    def __init__(self, n_observations, n_actions):
        super(FeedForwardNN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.active1 = nn.PReLU(128, random.uniform(0.1, 0.5))    ##
        self.layer2 = nn.Linear(128, 128)
        self.active2 = nn.PReLU(128, random.uniform(0.1, 0.5))    ##
        self.output_layer = nn.Linear(128, n_actions)
    def forward(self, x):
        #x = F.prelu(self.norm1(self.layer1(x)), 0.5*torch.rand(1, dtype=torch.float32, device=device))
        #x = F.prelu(self.norm2(self.layer2(x)), 0.5*torch.rand(1, dtype=torch.float32, device=device))
        #x = F.relu(self.layer1(x))
        #x = F.relu(self.layer2(x))
        x = self.layer1(x)
        x = self.active1(x)
        x = self.layer2(x)
        x = self.active2(x)
        return self.output_layer(x)
    

def train(env, hyperparameters, policy_model, target_model):
    """
        Trains the model.
    """
    print(f"Training", flush=True)

	# Create a model for PPO.
    model = DQN(policy_class=FeedForwardNN, env=env, **hyperparameters)

	# Tries to load in an existing policy/target model to continue training on
    if policy_model != '' and target_model != '':
        print(f"Loading in {policy_model} and {target_model}...", flush=True)
        model.policy.load_state_dict(torch.load(policy_model))
        model.target.load_state_dict(torch.load(target_model))
        print(f"Successfully loaded.", flush=True)
    elif policy_model != '' or target_model != '': # Don't train from scratch if user accidentally forgets policy/target model
        print(f"Error: Either specify both policy/target models or none at all. We don't want to accidentally override anything!")
        sys.exit(0)
    else:
        print(f"Training from scratch.", flush=True)

	# Train the DQN model with a specified total timesteps
	# NOTE: You can change the total timesteps here, I put a big number just because
	# you can kill the process whenever you feel like DQN is converging
    model.learn()


## **************************************************************************************
## **************************************************************************************


#env = LearnedPAHM(render_mode="human")
env = LearnedPAHM(render_mode="rgb_array")

train(env=env, hyperparameters=wandb.config, policy_model='', target_model='')

wandb.finish()
env.close()



