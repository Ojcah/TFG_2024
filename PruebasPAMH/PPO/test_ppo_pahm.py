
import torch
import numpy as np
import math
import time
import matplotlib.pyplot
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



def calculate_reward(observ, pwm, target_angle): # Todos los valores estan en radianes
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
				theta_good += 0.2
		else:
			if theta_good > 0.0:
				theta_good = 0.0
			else:
				theta_good -= 0.2

		if pwm < 0.0 or pwm > 0.25:
			extra_cost = 10 ** np.absolute(pwm - 0.25)
		else:
			extra_cost = 0.0
  
		#reward_n = np.min([-velocity_cost, -theta_error_cost, -extra_cost]) + theta_good
		reward_n = np.min([-velocity_cost, -theta_error_cost, -extra_cost])

		return reward_n.item()

def calculate_rewardV2(observ, pwm, target_angle): # Todos los valores estan en radianes
		theta = observ[0]
		theta_dot = observ[1]
		theta_n = ((theta + np.pi) % (2*np.pi)) - np.pi
		theta_error = np.abs(theta_n - target_angle)
		
		theta_error_cost = (theta_error ** 2)
		velocity_cost = 100 * (theta_dot ** 2)

		variance_rew = 0.005

        #theta_good = np.max([0.0, 1.0 - theta_error_cost]) * (1.0 - np.abs(velocity_cost))  
		theta_good = 2 * np.exp(- theta_error_cost/variance_rew) * (1.0 - np.abs(velocity_cost))  

		if pwm < 0.0 or pwm > 0.25:
			extra_cost = 20 ** np.absolute(pwm - 0.25)
		else:
			extra_cost = 0.0
		
		reward_n = np.min([-theta_error_cost, -extra_cost]) + theta_good
		return reward_n.item()


def _log_summary(ep_len, ep_ret, ep_num, target_angle, time_limits, just_angle):
		"""
			Print to stdout what we've logged so far in the most recent episode.

			Parameters:
				None

			Return:
				None
		"""	
		# Rise time
		# delta_t = (time_limits[1]- time_limits[0]) / 1e9
		# deltas = np.linspace(0.0, delta_t, ep_len+1)
		delta_t = 20 / 1e3 # son 20ms
		deltas = np.linspace(0.0, delta_t * (ep_len+1), num=(ep_len+1))



		final_value = just_angle[-1]
		rise_value = [0.1*final_value, 0.9*final_value]
		indice_10 = np.where(just_angle >= rise_value[0])[0][0]
		indice_90 = np.where(just_angle >= rise_value[1])[0][0]

		rise_time = deltas[indice_90] - deltas[indice_10]

		# Settling time
		tolerance = 0.02
		high_limit = final_value * (1 + tolerance)
		low_limit = final_value * (1 - tolerance)

		for i in range(len(just_angle) - 1, -1, -1):
			if not (low_limit <= just_angle[i] <= high_limit):
				settling_time = deltas[i+1]
				break

		# Overshoot
		high_value = np.max(just_angle)
		overshoot = ((high_value - final_value) / final_value) * 100

        # Round decimal places for more aesthetic logging messages
		ep_len = str(round(ep_len, 2))
		ep_ret = str(round(ep_ret, 2))

		# Print logging statements
		print(flush=True)
		print(f"-------------------- Episode #{ep_num} --------------------", flush=True)
		print(f"Target Angle: {str(target_angle)}", flush=True)
		print(f"Episodic Length: {ep_len}", flush=True)
		print(f"Episodic Reward: {ep_ret}", flush=True)
		print(f"------------------------------------------------------", flush=True)
		print(flush=True)
		
		if ep_num == 1:
			plt.figure(figsize=(10, 10))
			# Clear previous plot
			plt.clf()
			plt.plot(deltas, just_angle, color="blue")
			plt.axhline(y=final_value, color='g', linestyle='--', label='Final value')
			plt.axhline(y=high_value, color='r', linestyle='--', label='Max value')
			plt.axvline(x=deltas[indice_10], color='m', linestyle='--', label=f'Final value 10%%')
			plt.axvline(x=deltas[indice_90], color='m', linestyle='--', label=f'Final value 90%%')
			plt.xlabel("Time [s]")
			plt.ylabel("Angle [°]")
			plt.title("Current Angle")
			plt.grid(True)
			plt.pause(0.01)

			print(flush=True)
			print(f"********************************", flush=True)
			print(f" Rise time: {str(rise_time)}", flush=True)
			print(f" Settling time: {str(settling_time)}", flush=True)
			print(f" Overshoot: {str(overshoot)}", flush=True)
			print(f"********************************", flush=True)





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

	# Rollout until user kills process
	while True:
		just_for_the_angle = np.array([0.0])
		obs, _ = env.reset(seed=123)			# for Gymnasium version
		done = False

		obs_n = np.array([obs.item(), 0.0, 0.0, math.radians(target_angle)])

		# number of timesteps so far
		t = 0

		# Logging data
		ep_len = 0            # episodic length
		ep_ret = 0            # episodic return
		
		delta_t0 = time.time_ns()

		while not done:
			t += 1
			# Render environment if specified, off by default
			if render:
				env.render()

			# Query deterministic action from policy and run it
			#action = policy(obs_n).detach().cpu().numpy()
			action = policy(obs_n).detach().cpu().numpy()

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
			rew = calculate_rewardV2(obs_n, action, math.radians(target_angle))

			# Sum all episodic rewards as we go along
			ep_ret += rew

			print(" >> Angle: ", math.degrees(obs_n[0]), " >> PWM: ", action, " >> Reward: ", rew, end="\r")

			just_for_the_angle = np.append(just_for_the_angle, math.degrees(obs_n[0]))
			#just_for_the_angle.append(math.degrees(obs_n[0]))

			if done or (t==800):
				delta_t1 = time.time_ns()
				break
			
		# Track episodic length
		ep_len = t
		
		# returns episodic length and return in this iteration
		yield ep_len, ep_ret, np.array([delta_t0, delta_t1]), just_for_the_angle


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
	for ep_num, (ep_len, ep_ret, time_limits, just_angle) in enumerate(rollout(policy, env, render, target_angle)):
		_log_summary(ep_len=ep_len, ep_ret=ep_ret, ep_num=ep_num, target_angle=math.radians(target_angle), time_limits=time_limits, just_angle=just_angle)
		if ep_num == 5:
			break


