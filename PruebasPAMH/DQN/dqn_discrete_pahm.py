

import gymnasium as gym
import math
import random
import time
import wandb
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np


# if GPU is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))
class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQN:
    """
        This is the DQN class we will use as our model in main.py
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
        # Initialize hyperparameters for training with DQN
        self._init_hyperparameters(hyperparameters)

        # Extract environment information
        self.env = env
        self.obs_dim = env.observation_space.shape[0] + 3
        self.act_dim = self.num_intervals

		 # Initialize actor and critic networks
        self.policy_net = policy_class(self.obs_dim, self.act_dim).to(device)
        self.target_net = policy_class(self.obs_dim, self.act_dim).to(device) 
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.lr, amsgrad=True)
        self.memory_obj = ReplayMemory(10000)

        self.steps_done = 0
        self.noise_amplitude = 6
        self.variance = 0.005

        self.pwm_logger = torch.zeros(15, dtype=torch.long, device=device)

        # This logger will help us with printing out summaries of each iteration
        self.logger = {
			'delta_t': time.time_ns(),
			'rews': [],       # episodic rewards
            'noise': [],
			'losses': [],     # losses of actor network in current iteration
            'pwm_signal': [],     # losses of actor network in current iteration

            'theta_error_cost': [],
			'velocity_cost': [],
            'theta_good': [],
		}
    
    def calculate_reward(self, observ, target_angle): # Todos los valores estan en radianes
        theta = observ[0]
        theta_dot = observ[1]
        
        theta_n = ((theta + np.pi) % (2*np.pi)) - np.pi
        
        theta_error = np.abs(theta_n - target_angle)
        
        theta_error_cost = (theta_error ** 2)
        
        velocity_cost = 10 * (theta_dot ** 2)

        if theta_error <= 0.0873: # ~ 5°
            theta_good = np.max([0.0, 1.0 - theta_error]) * (1.0 - np.abs(velocity_cost))
        else:
            theta_good = 0.0    

        self.logger['theta_error_cost'].append(-theta_error_cost)
        self.logger['velocity_cost'].append(-velocity_cost)
        self.logger['theta_good'].append(theta_good)

        reward_n = np.min([-velocity_cost, -theta_error_cost]) + theta_good
        #reward_n = theta_good
        
        return torch.tensor([reward_n.item()], device=device)
    
    def calculate_rewardV2(self, observ, pwm, target_angle): # Todos los valores estan en radianes
        theta = observ[0]
        theta_dot = observ[1]
        
        theta_n = ((theta + np.pi) % (2*np.pi)) - np.pi
        
        theta_error = np.abs(theta_n - target_angle)
        
        theta_error_cost = (theta_error ** 2)
        
        velocity_cost = 100 * (theta_dot ** 2)

        theta_good = 2 * np.exp(- theta_error_cost/self.variance) * (1.0 - np.abs(velocity_cost))

        pwm_repeated = torch.unique(self.pwm_logger).cpu().numpy().size
        # if pwm_repeated <= 2:
        if pwm_repeated == 1:
            extra_cost = (20 ** np.absolute(pwm)) + 1
        else:
            extra_cost = 0.0

        self.logger['theta_error_cost'].append(-theta_error_cost)
        self.logger['velocity_cost'].append(-velocity_cost)
        self.logger['theta_good'].append(theta_good)     

        reward_n = np.min([-theta_error_cost, -extra_cost]) + theta_good
        # reward_n = -theta_error_cost + theta_good
        # reward_n = np.min([-theta_error_cost, -velocity_cost])

        return torch.tensor([reward_n.item()], device=device)
    

    def discretize_action(self, action):
        """
            AAAAAAA
        """
        # Calcula el índice de la acción discreta
        if action.item() > 0.01:
            discrete_action = int(action.item() * 36)
            # discrete_action = int((action.item() ** 2) / 0.007)
            # discrete_action = int(9 + (np.log(action.item()/0.25)) / 0.4)
        else:
            discrete_action = 0
        
        return torch.clamp(torch.tensor(discrete_action, device=device), min=0, max=9)
    
    def undiscretize_action(self, discrete_action):
        """
            BBBBBBB
        """
        # Calcula el valor normalizado dentro del rango [0, 1]
        continuous_action = discrete_action / 36
        # continuous_action = np.sqrt(0.007 * discrete_action)
        # continuous_action = 0.25 * np.exp(0.4 * (discrete_action - 9))
        if continuous_action > 0.25:
            continuous_action = np.array([[0.25]])
        elif continuous_action < 0.007:
            continuous_action = np.array([[0.0]])

        return continuous_action

    def learn(self):
        """
			Train the agent network. Here is where the main DQN algorithm resides.
			Return:
				None
		"""
        targets_options = np.array([self.target_angle, 60, 90, 60, 30, 20, 45, 70, 90, 45])

        epoch = 0
        epoch_save_checkspoints = self.num_episodes//5
        target_step = self.num_episodes//len(targets_options)
        i_target = 0
        u_option = 1

        pole_plots = np.array([0.0, 0.0, 0.0, 0.0])
        step = 0

        ep_rews = []

        print(f" >> Máximo ruido agregado : {self.noise_amplitude}")

        for i_episode in range(self.num_episodes):

            ## *********************************CAMBIO DEL TARGET_ANGLE****************************************
            if self.change_angle and (i_target == target_step):
                self.target_angle = targets_options[u_option]
                u_option += 1
                i_target = 1
                print("Target angle >> ", self.target_angle)
            else:
                i_target += 1

            ## ************************************************************************************************

            # Initialize the environment and get its state
            last_obs = 0.0
            theta_dot = 0.0

            last_vel = 0.0
            theta_ddot = 0.0

            last_reward = torch.tensor([0.0], device=device)
            reward = torch.tensor([0.0], device=device)

            obs, info = self.env.reset()
            obs_n = np.array([obs.item(), 0.0, 0.0, math.radians(self.target_angle)])
            obs_n = torch.tensor(obs_n, dtype=torch.float32, device=device).unsqueeze(0)

            for t in count():
                action = self.select_action(obs_n)

                action_step = self.undiscretize_action(action.cpu().detach().numpy())

                self.logger['pwm_signal'].append(action_step[0])

                last_obs = obs.item()
                last_vel = obs_n[0, 1].item()

                last_reward = reward

                obs, reward_base, terminated, truncated, _ = self.env.step(action_step[0])

                obs_n[0, 0] = obs.item()
                theta_dot = obs.item() - last_obs
                obs_n[0, 1] = theta_dot
                theta_ddot = theta_dot - last_vel
                obs_n[0, 2] = theta_ddot

                # ///////////////////////////////////////////////////////////
                #reward = torch.tensor([reward], device=device) # Pendulum original target: theta=0°
                #reward = self.calculate_reward(obs_n[0].cpu().detach().numpy(), action.item(), math.radians(self.target_angle))
                reward = self.calculate_rewardV2(obs_n[0].cpu().detach().numpy(), action_step[0].item(), math.radians(self.target_angle))

                # reward_sh = reward - last_reward
                # ///////////////////////////////////////////////////////////

                done = terminated or truncated or (t==200)
                #done = terminated or truncated or (np.absolute(action.item())>2)

                ep_rews.append(reward[0].item())
                reward = torch.tensor([reward], device=device)
                # ep_rews.append(reward_sh[0].item())
                # reward_shaping = torch.tensor([reward_sh], device=device)
                ## *********************************************************************************

                pole_plots = obs_n[0].cpu().detach().numpy()

                if terminated:
                    next_state = None
                else:
                    #next_state = torch.tensor(obs_n, dtype=torch.float32, device=device)
                    next_state = obs_n.clone().detach()

                # Store the transition in memory
                self.memory_obj.push(obs_n, action, next_state, reward)
                # self.memory_obj.push(obs_n, action, next_state, reward_shaping)

                # Move to the next state
                obs_n = next_state

                # Perform one step of the optimization (on the policy network)
                self.optimize_model()
                
                # Soft update of the target network's weights
                # θ′ ← τ θ + (1 −τ )θ′
                target_net_state_dict = self.target_net.state_dict()
                policy_net_state_dict = self.policy_net.state_dict()
                for key in policy_net_state_dict:
                    target_net_state_dict[key] = policy_net_state_dict[key]*self.tau + target_net_state_dict[key]*(1-self.tau)
                self.target_net.load_state_dict(target_net_state_dict)

                if done:
                    self.logger['rews'].append(ep_rews)
                    ep_rews = []
                    step = t
                    self.pwm_logger.zero_()
                    break
            
            if i_episode == (self.num_episodes//2):
                self.noise_amplitude = 4
                print(f" >> Máximo ruido agregado : {self.noise_amplitude}")
            elif i_episode == (3*self.num_episodes//4):
                self.noise_amplitude = 2
                print(f" >> Máximo ruido agregado : {self.noise_amplitude}")

            self._log_summary(i_episode, pole_plots, step)
            ## ***********************************************SAVE CHECKPOINTS*********************************************
            if (epoch == epoch_save_checkspoints):
                epoch = 1
                # Guardar el modelo
                torch.save(self.policy_net.state_dict(), f"Checkpoints/dqn_actor_{i_episode}.pth")
            elif ((i_episode+1) == self.num_episodes):
                epoch = 1
                # Guardar el modelo
                torch.save(self.policy_net.state_dict(), f"actor-critic/dqn_actor.pth")
            else:
                epoch += 1

        
    def optimize_model(self):
        if len(self.memory_obj) < self.batch_size:
            return
        transitions = self.memory_obj.sample(self.batch_size)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
     
        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        #state_action_values = self.policy_net(state_batch).gather(1, action_batch)
        
        q_values = self.policy_net(state_batch)
        state_action_values = q_values.gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1).values
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.batch_size, dtype=torch.float32, device=device)


        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1).values

            #next_state_values[non_final_mask] = torch.argmax(self.target_net(non_final_next_states), dim=1)
            
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        # criterion = nn.CrossEntropyLoss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()

        # Log actor loss
        self.logger['losses'].append(loss.detach())

        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()





    def select_action(self, state):
        """
            EEEEEE
        """
        sample = random.random()
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * \
        math.exp(-1. * self.steps_done / self.eps_decay)
        self.steps_done += 1
        # *************************************************************************
        with torch.no_grad():
            value_discrete = torch.argmax(self.policy_net(state), dim=1)
        if sample > eps_threshold:
            newvalue = torch.clamp(value_discrete, min=0, max=9)
            self.logger['noise'].append(0.0)

        else:
            noise = random.randint(-self.noise_amplitude, self.noise_amplitude)
            self.logger['noise'].append(noise)
            newvalue = torch.clamp(value_discrete + noise, min=0, max=9)
        
        self.pwm_logger = torch.roll(self.pwm_logger, -1)
        self.pwm_logger[-1] = newvalue        
        # *************************************************************************
        if newvalue == 0:
            return torch.tensor([[0]], dtype=torch.long, device=device)
        else:
            return torch.tensor([[newvalue]], dtype=torch.long, device=device)
 

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

    def _log_summary(self, i_episode, obs, step):
        """
			Print to stdout what we've logged so far in the most recent batch.

			Parameters:
				None

			Return:
				None
		"""
        print(f" >> Episodio #{i_episode} / {self.num_episodes} >> Target angle: {self.target_angle}")

        delta_t = self.logger['delta_t']
        self.logger['delta_t'] = time.time_ns()
        delta_t = (self.logger['delta_t'] - delta_t) / 1e9
		

        rewards = torch.tensor(self.logger['rews'], dtype=torch.float)
        losses = torch.tensor(self.logger['losses'], dtype=torch.float)
        noises = torch.tensor(self.logger['noise'], dtype=torch.float)

        rew_means = torch.mean(rewards, dtype=float)
        loss_means = torch.mean(losses, dtype=float)
        noise_means = torch.mean(noises, dtype=float)

        avg_pwm_signal = np.median(self.logger['pwm_signal'])

        avg_theta_error_cost = np.mean(self.logger['theta_error_cost'])
        avg_velocity_cost = np.mean(self.logger['velocity_cost'])
        avg_theta_good = np.mean(self.logger['theta_good'])

        # rew_means = avg_rewards.unfold(0, 100, 1).mean(1).view(-1)
        # rew_means = torch.cat((torch.zeros(99), rew_means))
        # loss_means = avg_loss.unfold(0, 100, 1).mean(1).view(-1)
        # loss_means = torch.cat((torch.zeros(99), loss_means))
        # noise_means = avg_noise.unfold(0, 100, 1).mean(1).view(-1)
        # noise_means = torch.cat((torch.zeros(99), noise_means))
        
        # wandb.log({"Average Reward": rew_means.numpy(), "Average Loss": loss_means.numpy(), "Pole Angle": pole_angle})
        wandb.log({"Average Reward": rew_means.item(),
                       "Average Loss":  loss_means.item(),
                       "Average Noise":  noise_means.item(),
                       "Episode length": step,
                       "Average PWM": avg_pwm_signal,
                       
                       "avg_theta_error_cost": avg_theta_error_cost,
                       "avg_velocity_cost": avg_velocity_cost,
                       "avg_theta_good": avg_theta_good,

                       "current angle": math.degrees(obs[0]),
                       "velocity pole": obs[1],
                       "acceleration pole": obs[2],
                       })
            
        self.logger['rews'] = []
        self.logger['losses'] = []
        self.logger['noise'] = []
        self.logger['pwm_signal'] = []

        self.logger['theta_error_cost'] = []
        self.logger['velocity_cost'] = []
        self.logger['theta_good'] = []

        
		



