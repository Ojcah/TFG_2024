"""
	The file contains the PPO class to train with.
	NOTE: All "ALG STEP"s are following the numbers from the original PPO pseudocode.
			It can be found here: https://spinningup.openai.com/en/latest/_images/math/e62a8971472597f4b014c2da064f636ffe365ba3.svg
"""

import gymnasium as gym

import numpy as np
import time
import torch
import torch.nn as nn
import matplotlib
import matplotlib.pyplot as plt
import math
from gymnasium.wrappers.monitoring import video_recorder
from torch.optim import Adam
from torch.distributions import MultivariateNormal
from torch.distributions.normal import Normal

import wandb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

average_len = []
average_reward = []
average_loss = []
timesteps_sofar = []
time_for_iteration = []

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display
plt.ion()


class PPO:
	"""
		This is the PPO class we will use as our model in main.py
	"""
	def __init__(self, policy_class, env, **hyperparameters):
		"""
			Initializes the PPO model, including hyperparameters.

			Parameters:
				policy_class - the policy class to use for our actor/critic networks.
				env - the environment to train on.
				hyperparameters - all extra arguments passed into PPO that should be hyperparameters.

			Returns:
				None
		"""
		# Make sure the environment is compatible with our code
		assert(type(env.observation_space) == gym.spaces.Box)
		assert(type(env.action_space) == gym.spaces.Box)

		# Initialize hyperparameters for training with PPO
		self._init_hyperparameters(hyperparameters)

		# Extract environment information
		self.env = env
		self.obs_dim = env.observation_space.shape[0] + 3
		self.act_dim = env.action_space.shape[0] 

		 # Initialize actor and critic networks
		self.actor = policy_class(self.obs_dim, self.act_dim).to(device)                                                   # ALG STEP 1
		self.critic = policy_class(self.obs_dim, 1).to(device) 

		# Initialize optimizers for actor and critic
		self.actor_optim = Adam(self.actor.parameters(), lr=self.lr)
		self.critic_optim = Adam(self.critic.parameters(), lr=self.lr)

		# Initialize the covariance matrix used to query the actor for actions
		self.cov_var = torch.full(size=(self.act_dim,), fill_value=0.5, device=device)
		#self.cov_var = torch.full(size=(self.act_dim,), fill_value=0.1, device=device)
		self.cov_mat = torch.diag(self.cov_var)

		self.dev_std = 0.7071
		self.last_noise = np.array([0.1])
		self.noise_suppressor = False
		self.sigma_x0_noise = 0.7629
		self.noise_threshold = 4
		self.theta_good = 0.0

		# This logger will help us with printing out summaries of each iteration
		self.logger = {
			'delta_t': time.time_ns(),
			't_so_far': 0,          # timesteps so far
			'i_so_far': 0,          # iterations so far
			'batch_lens': [],       # episodic lengths in batch
			'batch_rews': [],       # episodic returns in batch
			'actor_losses': [],     # losses of actor network in current iteration
			'critic_losses': [],     # losses of critic network in current iteration
			'pwm_signal': [],

			'theta_error_cost': [],
			'velocity_cost': [],
			'extra_cost': [],
			'theta_good': [],
		}

	def calculate_reward(self, observ, pwm, target_angle, batch_len): # Todos los valores estan en radianes
		theta = observ[0]
		theta_dot = observ[1]
		
		theta_n = ((theta + np.pi) % (2*np.pi)) - np.pi

		theta_error = np.abs(theta_n - target_angle)
		# theta_error_cost = theta_error ** 2
		theta_error_cost = 5 * theta_error

		velocity_cost = 0.1 * (theta_dot ** 2)
		
		if pwm < 0.0 or pwm > 0.5:
			costs = theta_error_cost + velocity_cost + (10**(np.absolute(pwm - 0.5)))
		else:
			costs = theta_error_cost + velocity_cost

		
		if theta_error <= 0.0873: # 0.1745 ~ 10° # 0.0873 ~ 5°
			#reward_n = -costs + math.exp(-(6*theta_error)**2)
			reward_n = -costs + 1.8*math.exp(-(10*theta_error)**2)
		else:
			reward_n = -costs 

		if batch_len >= 100:
			batch_len_cost = 1.05 ** (batch_len - 200)
			reward_n = reward_n + batch_len_cost

		return reward_n.item()
	
	def calculate_rewardV2(self, observ, pwm, target_angle): # Todos los valores estan en radianes
		theta = observ[0]
		theta_dot = observ[1]
		
		theta_n = ((theta + np.pi) % (2*np.pi)) - np.pi

		theta_error = np.abs(theta_n - target_angle)

		theta_error_cost = (theta_error ** 2)
		velocity_cost = 100 * (theta_dot ** 2)

		if theta_error <= 0.1745: # 0.1745 ~ 10° # 0.0873 ~ 5°
			if self.theta_good < 0.0:
				self.theta_good = 0.0
			else:
				self.theta_good += 0.15
		else:
			if self.theta_good > 0.0:
				self.theta_good = 0.0
			else:
				self.theta_good -= 0.15

		if pwm < 0.0 or pwm > 0.25:
			extra_cost = 10 ** np.absolute(pwm - 0.15)
		else:
			extra_cost = 0.0
  
		self.logger['theta_error_cost'].append(-theta_error_cost)
		self.logger['velocity_cost'].append(-velocity_cost)
		self.logger['extra_cost'].append(-extra_cost)
		self.logger['theta_good'].append(self.theta_good)

		reward_n = np.min([-velocity_cost, -theta_error_cost, -extra_cost]) + self.theta_good

		return reward_n.item()

	def learn(self, total_timesteps):
		"""
			Train the actor and critic networks. Here is where the main PPO algorithm resides.

			Parameters:
				total_timesteps - the total number of timesteps to train for

			Return:
				None
		"""
		print(f"Learning... Running {self.max_timesteps_per_episode} timesteps per episode, ", end='')
		print(f"{self.timesteps_per_batch} timesteps per batch for a total of {total_timesteps} timesteps")
		t_so_far = 0 # Timesteps simulated so far
		i_so_far = 0 # Iterations ran so far
		angle_list = np.array([self.target_angle, 60, 90, 60, 30, 20, 45, 70])
		noise_time = (total_timesteps/2)//6
		angle_time = (total_timesteps/2)//10
		angle_t = 0
		noise_t = 0
		angle_index = 0
		
		while t_so_far < total_timesteps:                                                                       # ALG STEP 2
			
			if self.change_angle and (angle_t >= angle_time):
				if angle_index > len(angle_list):
					angle_index = 0
				else:
					angle_index += 1
				self.target_angle = angle_list[angle_index]
				angle_t = 1
				print(f">> Target angle: {self.target_angle}")
			else:
				self.target_angle = angle_list[angle_index]

			
			# Autobots, roll out (just kidding, we're collecting our batch simulations here)
			batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens = self.rollout()                     # ALG STEP 3

			# Calculate how many timesteps we collected this batch
			t_so_far += np.sum(batch_lens)
			angle_t += np.sum(batch_lens)

			# Increment the number of iterations
			i_so_far += 1

			# Logging timesteps so far and iterations so far
			self.logger['t_so_far'] = t_so_far
			self.logger['i_so_far'] = i_so_far

			# Calculate advantage at k-th iteration
			# V, _ = self.evaluate(batch_obs, batch_acts)
			V, _= self.evaluateV2(batch_obs, batch_acts)
			A_k = batch_rtgs - V.detach()                                                                       # ALG STEP 5

			# One of the only tricks I use that isn't in the pseudocode. Normalizing advantages
			# isn't theoretically necessary, but in practice it decreases the variance of 
			# our advantages and makes convergence much more stable and faster. I added this because
			# solving some environments was too unstable without it.
			A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)

			# This is the loop where we update our network for some n epochs
			for _ in range(self.n_updates_per_iteration):                                                       # ALG STEP 6 & 7
				# Calculate V_phi and pi_theta(a_t | s_t)
				V, curr_log_probs = self.evaluate(batch_obs, batch_acts)
				# V, curr_log_probs = self.evaluateV2(batch_obs, batch_acts)

				# Calculate the ratio pi_theta(a_t | s_t) / pi_theta_k(a_t | s_t)
				# NOTE: we just subtract the logs, which is the same as
				# dividing the values and then canceling the log with e^log.
				# For why we use log probabilities instead of actual probabilities,
				# here's a great explanation: 
				# https://cs.stackexchange.com/questions/70518/why-do-we-use-the-log-in-gradient-based-reinforcement-algorithms
				# TL;DR makes gradient ascent easier behind the scenes.
				ratios = torch.exp(curr_log_probs - batch_log_probs)

				# Calculate surrogate losses.
				surr1 = ratios * A_k
				surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * A_k

				# Calculate actor and critic losses.
				# NOTE: we take the negative min of the surrogate losses because we're trying to maximize
				# the performance function, but Adam minimizes the loss. So minimizing the negative
				# performance function maximizes it.
				actor_loss = (-torch.min(surr1, surr2)).mean()
				critic_loss = nn.MSELoss()(V, batch_rtgs)

				# Calculate gradients and perform backward propagation for actor network
				self.actor_optim.zero_grad()
				actor_loss.backward(retain_graph=True)
				self.actor_optim.step()

				# Calculate gradients and perform backward propagation for critic network
				self.critic_optim.zero_grad()
				critic_loss.backward()
				self.critic_optim.step()

				# loss = 0.5 * critic_loss + actor_loss - 0.001 * entropy

				# Log actor loss
				self.logger['actor_losses'].append(actor_loss.detach())
				self.logger['critic_losses'].append(critic_loss.detach())
				# self.logger['actor_losses'].append(loss.detach())

			if (noise_t >= noise_time):
				self.noise_suppressor = True
				noise_t = 1
			else:
				noise_t += np.sum(batch_lens)

			# Print a summary of our training so far
			self._log_summary(total_timesteps)

			# Save our model if it's time
			if (i_so_far % self.save_freq == 0) or (t_so_far >= total_timesteps):
				torch.save(self.actor.state_dict(), './actor-critic/ppo_actor.pth')
				torch.save(self.critic.state_dict(), './actor-critic/ppo_critic.pth')
		print(" >> Complete <<")

	def rollout(self):
		"""
			Too many transformers references, I'm sorry. This is where we collect the batch of data
			from simulation. Since this is an on-policy algorithm, we'll need to collect a fresh batch
			of data each time we iterate the actor/critic networks.

			Parameters:
				None

			Return:
				batch_obs - the observations collected this batch. Shape: (number of timesteps, dimension of observation)
				batch_acts - the actions collected this batch. Shape: (number of timesteps, dimension of action)
				batch_log_probs - the log probabilities of each action taken this batch. Shape: (number of timesteps)
				batch_rtgs - the Rewards-To-Go of each timestep in this batch. Shape: (number of timesteps)
				batch_lens - the lengths of each episode this batch. Shape: (number of episodes)
		"""
		# Batch data. For more details, check function header.
		batch_obs = []
		batch_acts = []
		batch_log_probs = []
		batch_rews = []
		batch_rtgs = []
		batch_lens = []

		# Episodic data. Keeps track of rewards per episode, will get cleared
		# upon each new episode
		ep_rews = []

		last_obs = 0.0
		theta_dot = 0.0

		last_vel = 0.0
		theta_ddot = 0.0

		#self.theta_good = 0

		t = 0 # Keeps track of how many timesteps we've run so far this batch

		# Keep simulating until we've run more than or equal to specified timesteps per batch
		while t < self.timesteps_per_batch:
			ep_rews = [] # rewards collected per episode

			self.theta_good = 0

			# Reset the environment. sNote that obs is short for observation. 
			obs, _ = self.env.reset(seed=123)
			done = False

			obs_n = np.array([obs.item(), 0.0, 0.0, math.radians(self.target_angle)])

			# Example usage:
   			# self.ou_process(self, dt, theta, sigma, x0, T, beta)
			# dt: Time step (1/50) || theta: Mean reversion rate || sigma: Noise strength || x0: Initial state
			# T: Total simulation time || 
			beta = (0.01) ** (1/25) # reach 0.01 after 25 steps
			self.last_noise = self.ou_process(0.02, 1, self.sigma_x0_noise, self.sigma_x0_noise, 200, beta)

			# Run an episode for a maximum of max_timesteps_per_episode timesteps
			for ep_t in range(self.max_timesteps_per_episode):
				# If render is specified, render the environment
				if self.render and (self.logger['i_so_far'] % self.render_every_i == 0) and len(batch_lens) == 0:
					self.env.render()

				t += 1 # Increment timesteps ran this batch so far

				# Track observations in this batch
				batch_obs.append(obs_n)
				#batch_obs.append(obs)
				#batch_obs = np.array(batch_obs)
				#batch_obs = np.array([obs.numpy() for obs in batch_obs])

				# Calculate action and make a step in the env. 
				# Note that rew is short for reward.
				action, log_prob = self.get_action(obs_n)
				# action, log_prob = self.get_actionV2(obs_n)
				#action, log_prob = self.get_actionV3(obs_n)

				self.logger['pwm_signal'].append(action.item())

				last_obs = obs.item()
				last_vel = obs_n[1]

				obs, rew, terminated, truncated, _ = self.env.step(action)		

				obs_n[0] = obs.item()
				theta_dot = obs_n[0] - last_obs
				obs_n[1] = theta_dot
				theta_ddot = obs_n[1] - last_vel
				obs_n[2] = theta_ddot

				#done = terminated or truncated or (np.absolute(action)>1)								# For Gymnasium version
				done = terminated or truncated or (action>1) or (action<-0.5)								# For Gymnasium version

				#rew = self.calculate_reward(obs_n, action, math.radians(self.target_angle))
				rew = self.calculate_rewardV2(obs_n, action.item(), math.radians(self.target_angle))

				# Track recent reward, action, and action log probability
				ep_rews.append(rew)
				batch_acts.append(action)
				batch_log_probs.append(log_prob)

				if self.noise_suppressor:
					self.sigma_x0_noise = self.sigma_x0_noise * 0.8
					if self.noise_threshold > 2:
						self.sigma_x0_noise = 0.7629
						self.noise_threshold -= 1
					elif self.noise_threshold <= 2 and self.noise_threshold > 0:
						self.sigma_x0_noise = 0.488256
						self.noise_threshold -= 1
					elif self.noise_threshold == 0:
						self.noise_threshold -= 1
						print(" Manejo del ruido - normal")
					self.last_noise = self.ou_process(0.02, 1, self.sigma_x0_noise, self.sigma_x0_noise, 200, beta)
					self.noise_suppressor = False
					print(" Factor sigma / x0 >>", self.sigma_x0_noise)

				# If the environment tells us the episode is terminated, break
				if done:
					break

			# Track episodic lengths and rewards
			batch_lens.append(ep_t + 1)
			batch_rews.append(ep_rews)

		# Reshape data as tensors in the shape specified in function description, before returning
		batch_obs = torch.tensor(batch_obs, dtype=torch.float, device=device)
		batch_acts = torch.tensor(batch_acts, dtype=torch.float, device=device)
		batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float, device=device)
		batch_rtgs = self.compute_rtgs(batch_rews)                                                              # ALG STEP 4

		# Log the episodic returns and episodic lengths in this batch.
		self.logger['batch_rews'] = batch_rews
		self.logger['batch_lens'] = batch_lens

		return batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens

	def compute_rtgs(self, batch_rews):
		"""
			Compute the Reward-To-Go of each timestep in a batch given the rewards.

			Parameters:
				batch_rews - the rewards in a batch, Shape: (number of episodes, number of timesteps per episode)

			Return:
				batch_rtgs - the rewards to go, Shape: (number of timesteps in batch)
		"""
		# The rewards-to-go (rtg) per episode per batch to return.
		# The shape will be (num timesteps per episode)
		batch_rtgs = []

		# Iterate through each episode
		for ep_rews in reversed(batch_rews):

			discounted_reward = 0 # The discounted reward so far

			# Iterate through all rewards in the episode. We go backwards for smoother calculation of each
			# discounted return (think about why it would be harder starting from the beginning)
			for rew in reversed(ep_rews):
				discounted_reward = rew + discounted_reward * self.gamma
				batch_rtgs.insert(0, discounted_reward)

		# Convert the rewards-to-go into a tensor
		batch_rtgs = torch.tensor(batch_rtgs, dtype=torch.float, device=device)

		return batch_rtgs

	def get_actionV3(self, obs):
		mean = self.actor(obs)
		beta = 0.99
		
		dist = MultivariateNormal(mean*0, self.cov_mat)
		noise = dist.sample()
		self.last_noise = self.last_noise * beta + (1.0 - beta)*noise

		action = torch.clamp(mean + self.last_noise, min=0, max=2)

		log_prob = dist.log_prob(action)

		return action.detach().cpu().numpy(), log_prob.detach()

	def get_actionV2(self, obs):

		mean = self.actor(obs)

		noise = np.random.choice(self.last_noise)
		noise = torch.tensor(noise, dtype=float)

		action = torch.clamp(mean + noise, min=0, max=0.25)

		dist = Normal(action, self.dev_std)

		log_prob = torch.sum(dist.log_prob(action), dim=-1)

		return action.detach().cpu().numpy(), log_prob.detach()
	
	def ou_process(self, dt, theta, sigma, x0, T, beta):
		# Number of steps
		N = int(T/dt)

		# Initialize process output
		x = np.zeros(N)
		x[0] = x0

		y = x
		
		k1 = np.exp(-theta*dt)
		k2 = sigma*np.sqrt(dt)
		
		# Generate OU process
		for i in range(1, N):
			# Update state using Euler-Maruyama method
			x[i] = x[i-1] * k1 + k2 * np.random.randn(1); # original
			y[i] = beta * y[i-1] + (1-beta) * x[i]; # smoothed
		return y

	def get_action(self, obs):
		"""
			Queries an action from the actor network, should be called from rollout.

			Parameters:
				obs - the observation at the current timestep

			Return:
				action - the action to take, as a numpy array
				log_prob - the log probability of the selected action in the distribution
		"""
		# Query the actor network for a mean action
		mean = self.actor(obs)
		
		# Create a distribution with the mean action and std from the covariance matrix above.
		# For more information on how this distribution works, check out Andrew Ng's lecture on it:
		# https://www.youtube.com/watch?v=JjB58InuTqM
		dist = MultivariateNormal(mean, self.cov_mat)

		# Sample an action from the distribution
		action = dist.sample()
		action = torch.clamp(action, min=0, max=0.5)

		# Calculate the log probability for that action
		log_prob = dist.log_prob(action)
		# Return the sampled action and the log probability of that action in our distribution
		return action.detach().cpu().numpy(), log_prob.detach()

	def evaluateV2(self, batch_obs, batch_acts):
		V = self.critic(batch_obs).squeeze()

		mean = self.actor(batch_obs)

		noise = np.random.choice(self.last_noise)
		noise = torch.tensor(noise, dtype=float)

		action = torch.clamp(mean + noise, min=0, max=0.25)

		dist = Normal(action, self.dev_std)

		log_probs = torch.sum(dist.log_prob(batch_acts), dim=-1)

		return V, log_probs

	def evaluate(self, batch_obs, batch_acts):
		"""
			Estimate the values of each observation, and the log probs of
			each action in the most recent batch with the most recent
			iteration of the actor network. Should be called from learn.

			Parameters:
				batch_obs - the observations from the most recently collected batch as a tensor.
							Shape: (number of timesteps in batch, dimension of observation)
				batch_acts - the actions from the most recently collected batch as a tensor.
							Shape: (number of timesteps in batch, dimension of action)

			Return:
				V - the predicted values of batch_obs
				log_probs - the log probabilities of the actions taken in batch_acts given batch_obs
		"""
		# Query critic network for a value V for each batch_obs. Shape of V should be same as batch_rtgs
		V = self.critic(batch_obs).squeeze()

		# Calculate the log probabilities of batch actions using most recent actor network.
		# This segment of code is similar to that in get_action()
		mean = self.actor(batch_obs)
		mean = torch.clamp(mean, min=0.0, max=0.5)

		dist = MultivariateNormal(mean, self.cov_mat)

		log_probs = dist.log_prob(batch_acts)

		# Return the value vector V of each observation in the batch
		# and log probabilities log_probs of each action in the batch
		return V, log_probs

	def _init_hyperparameters(self, hyperparameters):
		"""
			Initialize default and custom values for hyperparameters

			Parameters:
				hyperparameters - the extra arguments included when creating the PPO model, should only include
									hyperparameters defined below with custom values.

			Return:
				None
		"""
		# Initialize default values for hyperparameters
		# Algorithm hyperparameters
		self.timesteps_per_batch = 4800                 # Number of timesteps to run per batch
		self.max_timesteps_per_episode = 1600           # Max number of timesteps per episode
		self.n_updates_per_iteration = 5                # Number of times to update actor/critic per iteration
		self.lr = 0.005                                 # Learning rate of actor optimizer
		self.gamma = 0.95                               # Discount factor to be applied when calculating Rewards-To-Go
		self.clip = 0.2                                 # Recommended 0.2, helps define the threshold to clip the ratio during SGA

		# Miscellaneous parameters
		self.render = True                              # If we should render during rollout
		self.render_every_i = 10                        # Only render every n iterations
		self.save_freq = 10                             # How often we save in number of iterations
		self.seed = None                                # Sets the seed of our program, used for reproducibility of results

		# Change any default values to custom values for specified hyperparameters
		for param, val in hyperparameters.items():
			exec('self.' + param + ' = ' + str(val))

		# Sets the seed if specified
		if self.seed != None:
			# Check if our seed is valid first
			assert(type(self.seed) == int)

			# Set the seed 
			torch.manual_seed(self.seed)
			print(f"Successfully set seed to {self.seed}")

	def _log_summary(self, total_timesteps):
		"""
			Print to stdout what we've logged so far in the most recent batch.

			Parameters:
				None

			Return:
				None
		"""
		# Calculate logging values. I use a few python shortcuts to calculate each value
		# without explaining since it's not too important to PPO; feel free to look it over,
		# and if you have any questions you can email me (look at bottom of README)
		delta_t = self.logger['delta_t']
		self.logger['delta_t'] = time.time_ns()
		delta_t = (self.logger['delta_t'] - delta_t) / 1e9
		delta_t = str(round(delta_t, 2))

		t_so_far = self.logger['t_so_far']
		i_so_far = self.logger['i_so_far']
		avg_ep_lens = np.mean(self.logger['batch_lens'])

		#avg_ep_rews = np.mean([np.sum(ep_rews) for ep_rews in self.logger['batch_rews']])
		avg_ep_rews = np.mean([np.mean(ep_rews) for ep_rews in self.logger['batch_rews']])

		#avg_actor_loss = np.mean([losses.float().mean() for losses in self.logger['actor_losses']])
		avg_actor_loss = np.mean([losses.float().cpu().mean().numpy() for losses in self.logger['actor_losses']])
		avg_critic_loss = np.mean([losses.float().cpu().mean().numpy() for losses in self.logger['critic_losses']])

		avg_pwm_signal = np.mean(self.logger['pwm_signal'])

		# Round decimal places for more aesthetic logging messages
		avg_ep_lens = str(round(avg_ep_lens, 2))
		avg_ep_rews = str(round(avg_ep_rews, 2))
		#avg_actor_loss = str(round(avg_actor_loss, 5))
		#avg_critic_loss = str(round(avg_critic_loss, 5))

		average_reward.append(float(avg_ep_rews))
		average_loss.append(float(avg_actor_loss))
		timesteps_sofar.append(float(t_so_far))
		time_for_iteration.append(float(delta_t))
		average_len.append(float(avg_ep_lens))

		print(f" >> Iteration #{i_so_far} >> Timesteps: {t_so_far}/{total_timesteps} >> Target angle: {self.target_angle}")

		# plt.figure(1,figsize=(10, 10))

		# # Clear previous plot
		# plt.clf()

		# plt.subplot(2, 2, 1)
		# plt.plot(timesteps_sofar, average_reward, color="blue")
		# plt.title("Training rewards (average)")
		# plt.grid(True)
		# plt.subplot(2, 2, 2)
		# plt.plot(timesteps_sofar, average_loss, color="green")
		# plt.title("Average Loss")
		# plt.grid(True)
		# plt.subplot(2, 2, 3)
		# plt.plot(timesteps_sofar, time_for_iteration, color="red")
		# plt.title("Time for iteration (seg)")
		# plt.grid(True)
		# plt.subplot(2, 2, 4)
		# plt.plot(timesteps_sofar, average_len, color="purple")
		# plt.title("Average Batch Length")
		# plt.grid(True)
		# plt.pause(0.01)
		# if is_ipython and (t_so_far < total_timesteps):
		# 	display.display(plt.gcf())
		# 	display.clear_output(wait=True)
  
		avg_theta_error_cost = np.mean(self.logger['theta_error_cost'])
		avg_velocity_cost = np.mean(self.logger['velocity_cost'])
		avg_extra_cost = np.mean(self.logger['extra_cost'])
		avg_theta_good = np.mean(self.logger['theta_good'])

		wandb.log({"Average Reward": float(avg_ep_rews), 
			 "Average Actor Loss": float(avg_actor_loss),
			 "Average Critic Loss": float(avg_critic_loss), 
			 "Iteration/Epoch took": float(delta_t), 
			 "Average Batch Length": float(avg_ep_lens),
			 "Last Noise": float(self.last_noise.mean()),
			 "avg_PWM_signal": avg_pwm_signal,
			 
			 'avg_theta_error_cost': avg_theta_error_cost,
			 'avg_velocity_cost': avg_velocity_cost,
			 'avg_extra_cost': avg_extra_cost,
			 'theta_good': avg_theta_good,
			 })
		
		# path_to_video = "video/ppo_pahm.mp4"
		# wandb.log({"Video": wandb.Video(path_to_video, fps=4, format="gif")})

		# Reset batch-specific logging data
		self.logger['batch_lens'] = []
		self.logger['batch_rews'] = []
		self.logger['actor_losses'] = []
		self.logger['pwm_signal'] = []

		self.logger['theta_error_cost'] = []
		self.logger['velocity_cost'] = []
		self.logger['extra_cost'] = []
		self.logger['theta_good'] = []

		
