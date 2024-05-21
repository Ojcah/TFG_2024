
import torch
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

# # ///////////////////////////////////
# slider_tg = 0

# # Crear la figura y los ejes
# fig, ax = plt.subplots()
# plt.subplots_adjust(bottom=0.25)

# # Configurar los límites del eje x y del eje y
# plt.axis([0, 120, 0, 1])

# # Crear el slider
# ax_slider = plt.axes([0.1, 0.1, 0.65, 0.03], facecolor='lightgoldenrodyellow')
# slider = Slider(ax_slider, 'Target angle', 0, 120, valinit=60, valstep=1)

# # Función para manejar el evento de cambio en el slider
# def update_val(val):
#     pass

# slider.on_changed(update_val)

# # Crear el botón
# ax_button = plt.axes([0.8, 0.025, 0.1, 0.04])
# button = Button(ax_button, 'Actualizar')

# # Función para manejar el evento de clic en el botón
# def update_var(event):
#     slider_tg = int(slider.val)

# button.on_clicked(update_var)

# plt.show()


# ///////////////////////////////////

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def calculate_reward(observ, pwm, target_angle, batch_len): # Todos los valores estan en radianes
		theta = observ[0]
		theta_dot = observ[1]
		
		theta_n = ((theta + np.pi) % (2*np.pi)) - np.pi

		theta_error = np.abs(theta_n - target_angle)
		theta_error_cost = theta_error ** 2

		velocity_cost = theta_dot ** 2

		# Agregar aceleracion
		
		if pwm < 0.0 or pwm > 0.5:
			costs = theta_error_cost + 0.1*velocity_cost + (10**(np.absolute(pwm - 0.5)))
		else:
			costs = theta_error_cost + 0.1*velocity_cost

		
		if theta_error <= 0.0873: # 0.1745 ~ 10° # 0.0873 ~ 5°
			#reward_n = -costs + math.exp(-(6*theta_error)**2)
			reward_n = -costs + 1.8*math.exp(-(10*theta_error)**2)
		else:
			reward_n = -costs 

		if batch_len >= 50:
			batch_len_cost = 1.05 ** (batch_len - 200)
			reward_n = reward_n + batch_len_cost

		return reward_n.item()

def calculate_rewardV2(observ, pwm, target_angle, theta_good): # Todos los valores estan en radianes
		theta = observ[0]
		theta_dot = observ[1]
		
		theta_n = ((theta + np.pi) % (2*np.pi)) - np.pi

		theta_error = np.abs(theta_n - target_angle)

		theta_error_cost = (theta_error ** 2)
		velocity_cost = 100 * (theta_dot ** 2)

		if theta_error <= 0.1745: # 0.1745 ~ 10° # 0.0873 ~ 5°
			if theta_good < 0.0:
				theta_good = 0.0
			else:
				theta_good += 0.15
		else:
			if theta_good > 0.0:
				theta_good = 0.0
			else:
				theta_good -= 0.15

		if pwm < 0.0 or pwm > 0.25:
			extra_cost = 10 ** np.absolute(pwm - 0.25)
		else:
			extra_cost = 0.0
  
		reward_n = np.min([-velocity_cost, -theta_error_cost, -extra_cost]) + theta_good

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
		print(f"Episodic Reward: {ep_ret}", flush=True)
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
	last_obs = 0.0
	theta_dot = 0.0

	last_vel = 0.0
	theta_ddot = 0.0

	theta_good = 0

	# Rollout until user kills process
	while True:
		obs, _ = env.reset(seed=123)			# for Gymnasium version
		done = False

		obs_n = np.array([obs.item(), 0.0, 0.0, math.radians(target_angle)])

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
			#action = policy(obs_n).detach().cpu().numpy()
			action = np.clip(policy(obs_n).detach().cpu().numpy(), 0.0, 0.25)

			last_obs = obs.item()
			last_vel = obs_n[1]

			obs, rew, terminated, truncated, _ = env.step(action)	# For Gymnasium version

			obs_n[0] = obs.item()
			theta_dot = obs_n[0] - last_obs
			obs_n[1] = theta_dot
			theta_ddot = obs_n[1] - last_vel
			obs_n[2] = theta_ddot

			done = terminated or truncated							# For Gymnasium

			#rew = calculate_reward(obs_n, action, math.radians(target_angle), ep_len)
			rew = calculate_rewardV2(obs_n, action.item(), math.radians(target_angle), theta_good)

			# Sum all episodic rewards as we go along
			ep_ret += rew

			print(" >> Angle: ", math.degrees(obs_n[0]), " >> PWM: ", action, " >> Reward: ", rew, end="\r")
			
		# Track episodic length
		ep_len = t
		
		# returns episodic length and return in this iteration
		yield ep_len, ep_ret


def eval_policy(policy, env, render=False, target_angle=60):
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


