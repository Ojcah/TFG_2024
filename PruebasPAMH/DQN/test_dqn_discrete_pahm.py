
import torch
import numpy as np
import math
import time
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def discretize_action(action, num_intervals):
    """
        AAAAAAA
    """
	# Calcula el índice de la acción discreta
    if action.item() > 0.01:
        # discrete_action = int(action.item() * 36)
        # discrete_action = int((action.item() ** 2) / 0.007)
        discrete_action = int(9 + (np.log(action.item()/0.25)) / 0.4)
    else:
        discrete_action = 0
        
    return torch.clamp(torch.tensor(discrete_action, device=device), min=0, max=9)
    
def undiscretize_action(discrete_action, num_intervals):
    """
        BBBBBBB
    """
    # Calcula el valor normalizado dentro del rango [0, 1]
    # continuous_action = discrete_action / 36
	# continuous_action = np.sqrt(0.007 * discrete_action)
    continuous_action = 0.25 * np.exp(0.4 * (discrete_action - 9))
    if continuous_action > 0.25:
        continuous_action = np.array([0.25])
    elif continuous_action < 0.007:
        continuous_action = np.array([0.0])

    return continuous_action

def calculate_rewardV3(observ, target_angle): # Todos los valores estan en radianes
        theta = observ[0]
        theta_dot = observ[1]
        
        theta_n = ((theta + np.pi) % (2*np.pi)) - np.pi
        
        theta_error = np.abs(theta_n - target_angle)
        
        theta_error_cost = (theta_error ** 2)
        
        velocity_cost = 100 * (theta_dot ** 2)

        variance = 0.01
        #theta_good = np.max([0.0, 1.0 - theta_error_cost]) * (1.0 - np.abs(velocity_cost))  
        theta_good = np.exp(- theta_error_cost/variance) * (1.0 - np.abs(velocity_cost))  

        reward_n = -theta_error_cost + theta_good
        
        return torch.tensor([reward_n.item()], device=device)

def calculate_rewardV2(observ, pwm, target_angle, pwm_logger): # Todos los valores estan en radianes
        theta = observ[0]
        theta_dot = observ[1]
        
        theta_n = ((theta + np.pi) % (2*np.pi)) - np.pi
        
        theta_error = np.abs(theta_n - target_angle)
        
        theta_error_cost = (theta_error ** 2)
        
        velocity_cost = 100 * (theta_dot ** 2)
		
        variance = 0.005

        theta_good = 2 * np.exp(- theta_error_cost/variance) * (1.0 - np.abs(velocity_cost))

        pwm_repeated = torch.unique(pwm_logger).cpu().numpy().size
        # if pwm_repeated <= 2:
        if pwm_repeated == 1:
            extra_cost = (20 ** np.absolute(pwm)) + 1
        else:
            extra_cost = 0.0

        reward_n = np.min([-theta_error_cost, -extra_cost]) + theta_good
        #reward_n = -theta_error_cost + theta_good
        # reward_n = np.min([-theta_error_cost, -velocity_cost])

        return torch.tensor([reward_n.item()], device=device)

def calculate_reward(observ, target_angle): # Todos los valores estan en radianes
        theta = observ[0]
        theta_dot = observ[1]
        
        theta_n = ((theta + np.pi) % (2*np.pi)) - np.pi
        
        theta_error = np.abs(theta_n - target_angle)
        
        theta_error_cost = (theta_error ** 2)
        
        velocity_cost = 10 * (theta_dot ** 2)
		
        variance = 0.009
        #theta_good = np.max([0.0, 1.0 - theta_error_cost]) * (1.0 - np.abs(velocity_cost))  
        theta_good = 5 * np.exp(- theta_error_cost/variance) * (1.0 - np.abs(velocity_cost))  

        reward_n = -theta_error_cost + theta_good

        return torch.tensor([reward_n.item()], device=device)

def _log_summary(ep_rew, ep_num, target_angle, ep_len, times_limits, just_angles, just_PWM):

	"""
	    Print to stdout what we've logged so far in the most recent episode.
	    
		Parameters:
	        None

		Return:
			None
	"""		
    # Round decimal places for more aesthetic logging messages
	ep_rew = str(round(ep_rew, 2))

	# delta_t = (times_limits[1]- times_limits[0]) / 1e9
	# deltas = np.linspace(0.0, delta_t, ep_len+1)
	delta_t = 20 / 1e3 # son 20ms
	deltas = np.linspace(0.0, delta_t * (ep_len+1), num=(ep_len+1))

	# Print logging statements
	print(flush=True)
	print(f"-------------------- Episode #{ep_num} --------------------", flush=True)
	print(f"Target Angle: {target_angle}", flush=True)
	print(f"Episodic Reward: {ep_rew}", flush=True)
	print(f"------------------------------------------------------", flush=True)
	print(flush=True)

	if ep_num == 0:
		# plt.figure(figsize=(10, 10))
		# # Clear previous plot
		# plt.clf()
		# plt.plot(deltas, just_angles, color="blue")
		# plt.title("Current Angle")
		# plt.grid(True)
		# plt.pause(0.01)

		plt.figure(figsize=(10, 10))

		# Plotear just_angles en el eje y izquierdo
		plt.plot(deltas, just_angles, color="blue", label="Angles")
		# plt.xlabel("Time [ns]")
		plt.xlabel("Time [s]")
		plt.ylabel("Angles [°]", color="blue")
		plt.title("Current Angle")
		plt.grid(True)

		# Crear un segundo eje y para just_PWM
		ax2 = plt.gca().twinx()
		ax2.plot(deltas, just_PWM, color="red", label="PWM", linewidth=0.5)
		ax2.set_ylabel("PWM", color="red")

		# Añadir leyendas
		plt.legend(loc="upper left")
		ax2.legend(loc="upper right")

		# Mostrar la figura
		#plt.show()
		

def rollout(policy_net, env, render, target_angle):
	"""
		SSS
	"""
	num_intervals = 10

	pwm_logger = torch.zeros(30, dtype=torch.long, device=device)
	
	last_obs = 0.0
	theta_dot = 0.0
	last_vel = 0.0
	theta_ddot = 0.0

	# Rollout until user kills process
	while True:
		just_for_the_angle = np.array([0.0])
		just_for_the_PWM = np.array([0.0])
		obs, info = env.reset()
		obs_n = np.array([obs.item(), 0.0, 0.0, math.radians(target_angle)])
		obs_n = torch.tensor(obs_n, dtype=torch.float32, device=device).unsqueeze(0)

		done = False

		# number of timesteps so far
		t = 0

		# Logging data
		ep_len = 0            # episodic length
		ep_ret = 0            # episodic return

		last_rew = torch.tensor([0.0], device=device)
		rew = torch.tensor([0.0], device=device)

		delta_t0 = time.time_ns()
		
		while not done:
			t += 1
			# Render environment if specified, off by default
			if render:
				env.render()

			# Query deterministic action from policy and run it
			action = torch.argmax(policy_net(obs_n), dim=1)
			action_step = undiscretize_action(action.cpu().detach().numpy(), num_intervals)
			
			pwm_logger = torch.roll(pwm_logger, -1)
			pwm_logger[-1] = action

			last_rew = rew

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
			rew = calculate_rewardV2(obs_n[0].cpu().detach().numpy(), action_step.item(), math.radians(target_angle), pwm_logger)

			# rew_shaping = rew - last_rew

			# Sum all episodic rewards as we go along
			ep_ret += rew.item()
			# ep_ret += rew_shaping.item()

			print(" >> Angle: ", math.degrees(obs_n[0, 0]), " >> PWM: ", [action_step.item(), action.item()], " >> Reward: ", rew.item(), end="\r")
			# print(" >> Angle: ", math.degrees(obs_n[0, 0]), " >> PWM: ", [action_step.item(), action.item()], " >> Reward: ", rew_shaping.item(), end="\r")

			just_for_the_angle = np.append(just_for_the_angle, math.degrees(obs_n[0, 0]))
			just_for_the_PWM = np.append(just_for_the_PWM, action_step.item())

			if done or (t==500):
				delta_t1 = time.time_ns()
				pwm_logger.zero_()
				break	
		
		# Track episodic length
		ep_len = t

		# returns episodic length and return in this iteration
		yield ep_ret, ep_len, np.array([delta_t0, delta_t1]), just_for_the_angle, just_for_the_PWM


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
	for ep_num, (ep_rew, ep_len, times_limits, just_angles, just_PWM) in enumerate(rollout(policy, env, render, target_angle)):
		_log_summary(ep_rew=ep_rew, ep_num=ep_num, target_angle=math.radians(target_angle), ep_len=ep_len, times_limits=times_limits, just_angles=just_angles, just_PWM=just_PWM)
		if ep_num == 5:
			break