
import torch
import numpy as np
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def calculate_reward(observ, pwm, target_angle): # Todos los valores estan en radianes
		theta = observ.item()
		
		theta_n = ((theta + np.pi) % (2*np.pi)) - np.pi

		theta_error = np.abs(theta_n - target_angle)
		
		if pwm < 0.0 or pwm > 1.0:
			costs = theta_error**2 + (10**(np.absolute(pwm - 1.0)))
		else:
			costs = theta_error**2

		if theta_error <= 0.1745: # ~ 10°
			reward_n = -costs + math.exp(-(6*theta_error)**2)
		else:
			reward_n = -costs
		# reward_n = -costs
		return reward_n.item()

def _log_summary(ep_len, ep_ret, ep_num, target_angle):
		"""
			Print to stdout what we've logged so far in the most recent episode.

			Parameters:
				None

			Return:
				None
		"""		
        # Round decimal places for more aesthetic logging messages
		ep_len = str(round(ep_len, 2))
		ep_ret = str(round(ep_ret, 2))

		# Print logging statements
		print(flush=True)
		print(f"-------------------- Episode #{ep_num} --------------------", flush=True)
		print(f"Target Angle: {target_angle}", flush=True)
		print(f"Episodic Length: {ep_len}", flush=True)
		print(f"Episodic Return: {ep_ret}", flush=True)
		print(f"------------------------------------------------------", flush=True)
		print(flush=True)
		

def rollout(policy, env, render, target_angle):
	"""
		Returns a generator to roll out each episode given a trained policy and
		environment to test on. 

		Parameters:
			policy - The trained policy to test
			env - The environment to evaluate the policy on
			render - Specifies whether to render or not
		
		Return:
			A generator object rollout, or iterable, which will return the latest
			episodic length and return on each iteration of the generator.

		Note:
			If you're unfamiliar with Python generators, check this out:
				https://wiki.python.org/moin/Generators
			If you're unfamiliar with Python "yield", check this out:
				https://stackoverflow.com/questions/231767/what-does-the-yield-keyword-do
	"""
	# Rollout until user kills process
	while True:
		# obs = env.reset()				# For Gym version
		obs, _ = env.reset()			# for Gymnasium version
		done = False

		# number of timesteps so far
		t = 0

		# Logging data
		ep_len = 0            # episodic length
		ep_ret = 0            # episodic return
		
		while not done:
			t += 1
			# Render environment if specified, off by default
			if render:
				env.render()

			# Query deterministic action from policy and run it
			action = policy(obs).detach().cpu().numpy()
			#obs, rew, done, _ = env.step(action)					# For Gym version
			obs, rew, terminated, truncated, _ = env.step(action)	# For Gymnasium version

			done = terminated or truncated							# For Gymnasium

			rew = calculate_reward(obs, action, math.radians(target_angle))

			# Sum all episodic rewards as we go along
			ep_ret += rew
			
		# Track episodic length
		ep_len = t
		
		# returns episodic length and return in this iteration
		yield ep_len, ep_ret


def eval_policy(policy, env, render=False, target_angle=45):
	"""
		The main function to evaluate our policy with. It will iterate a generator object
		"rollout", which will simulate each episode and return the most recent episode's
		length and return. We can then log it right after. And yes, eval_policy will run
		forever until you kill the process. 

		Parameters:
			policy - The trained policy to test, basically another name for our actor model
			env - The environment to test the policy on
			render - Whether we should render our episodes. False by default.

		Return:
			None

		NOTE: To learn more about generators, look at rollout's function description
	"""
	# Rollout with the policy and environment, and log each episode's data
	for ep_num, (ep_len, ep_ret) in enumerate(rollout(policy, env, render, target_angle)):
		_log_summary(ep_len=ep_len, ep_ret=ep_ret, ep_num=ep_num, target_angle=math.radians(target_angle))


