
import torch
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def discretize_action(action, num_intervals):
    """
        AAAAAAA
    """
    # Calcula el índice de la acción discreta
    #discrete_action = int(action.item() * num_intervals)
    #discrete_action = int(action.item() * 18)
    discrete_action = int(action.item() * 36)
    return torch.clamp(torch.tensor(discrete_action, device=device), min=0, max=num_intervals-1)
    
def undiscretize_action(discrete_action, num_intervals):
    """
        BBBBBBB
    """
    # Calcula el valor normalizado dentro del rango [0, 1]
    #continuous_action = discrete_action / 18
    continuous_action = discrete_action / 36
    return continuous_action

def calculate_rewardV2(observ, pwm, target_angle, theta_good): # Todos los valores estan en radianes
        theta = observ[0]
        theta_dot = observ[1]
        
        theta_n = ((theta + np.pi) % (2*np.pi)) - np.pi
        
        theta_error = np.abs(theta_n - target_angle)
        
        theta_error_cost = (theta_error ** 2)
        
        velocity_cost = 100 * (theta_dot ** 2)
        
        if theta_error <= 0.0873: # 0.1745 ~ 10° # 0.0873 ~ 5°
            theta_good = 0.3 * math.exp(-(20*theta_error) ** 2)
        else:
            theta_good = 0.0

        if pwm < 0.0 or pwm > 0.25:
            extra_cost = 10 ** np.absolute(pwm - 0.25)
        else:
            extra_cost = 0.0
            
        reward_n = np.min([-velocity_cost, -theta_error_cost, -extra_cost]) + theta_good
        #reward_n = np.min([-velocity_cost, -theta_error_cost, -extra_cost])
        
        return torch.tensor([reward_n.item()], device=device)

def calculate_reward(observ, target_angle): # Todos los valores estan en radianes
        theta = observ[0]
        theta_dot = observ[1]
        
        theta_n = ((theta + np.pi) % (2*np.pi)) - np.pi
        
        theta_error = np.abs(theta_n - target_angle)
        
        theta_error_cost = (theta_error ** 2)
        
        velocity_cost = 100 * (theta_dot ** 2)

        if theta_error <= 0.0873: # ~ 5°
            theta_good = np.max([0.0, 1.0 - theta_error]) * (1.0 - np.abs(velocity_cost))
        else:
            theta_good = 0.0    

        reward_n = np.min([-velocity_cost, -theta_error_cost]) + theta_good
        #reward_n = theta_good
        
        return torch.tensor([reward_n.item()], device=device)

def _log_summary(ep_rew, ep_num, target_angle):

	"""
	    Print to stdout what we've logged so far in the most recent episode.
	    
		Parameters:
	        None

		Return:
			None
	"""		
    # Round decimal places for more aesthetic logging messages
	ep_rew = str(round(ep_rew, 2))

	# Print logging statements
	print(flush=True)
	print(f"-------------------- Episode #{ep_num} --------------------", flush=True)
	print(f"Target Angle: {target_angle}", flush=True)
	print(f"Episodic Reward: {ep_rew}", flush=True)
	print(f"------------------------------------------------------", flush=True)
	print(flush=True)
		

def rollout(policy_net, env, render, target_angle):
	"""
		SSS
	"""
	num_intervals = 10
	
	last_obs = 0.0
	theta_dot = 0.0
	last_vel = 0.0
	theta_ddot = 0.0

	# Rollout until user kills process
	while True:
		obs, info = env.reset()
		obs_n = np.array([obs.item(), 0.0, 0.0, math.radians(target_angle)])
		obs_n = torch.tensor(obs_n, dtype=torch.float32, device=device).unsqueeze(0)

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
			action = torch.argmax(policy_net(obs_n), dim=1)
			action_step = undiscretize_action(action.cpu().detach().numpy(), num_intervals)

			last_obs = obs.item()
			last_vel = obs_n[0, 1].item()

			obs, rew, terminated, truncated, _ = env.step(action_step)	# For Gymnasium version

			obs_n[0, 0] = obs.item()
			theta_dot = obs.item() - last_obs
			obs_n[0, 1] = theta_dot
			theta_ddot = theta_dot - last_vel
			obs_n[0, 2] = theta_ddot
			done = terminated or truncated							# For Gymnasium

			#rew = calculate_reward(obs_n, action, math.radians(target_angle), ep_len)
			rew = calculate_reward(obs_n[0].cpu().detach().numpy(), math.radians(target_angle))

			# Sum all episodic rewards as we go along
			ep_ret += rew.item()

			print(" >> Angle: ", math.degrees(obs_n[0, 0]), " >> PWM: ", action_step.item(), " >> Reward: ", rew.item(), end="\r")
			
		# Track episodic length
		ep_len = t
		
		# returns episodic length and return in this iteration
		yield ep_ret


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
	for ep_num, (ep_rew) in enumerate(rollout(policy, env, render, target_angle)):
		_log_summary(ep_rew=ep_rew, ep_num=ep_num, target_angle=math.radians(target_angle))