

import gymnasium as gym
import math
import random
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
        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.shape[0]

		 # Initialize actor and critic networks
        self.policy_net = policy_class(self.obs_dim, self.act_dim).to(device)
        self.target_net = policy_class(self.obs_dim, self.act_dim).to(device) 
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.lr, amsgrad=True)
        self.memory_obj = ReplayMemory(10000)

        self.steps_done = 0
        self.avg_rewards = []
        self.avg_loss = []


    def calculate_reward(self, observ, pwm, target_angle):
        """
            AAAAAA
        """
        
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
        return torch.tensor(np.array([reward_n]), device=device)

    def discretize_action(self, action):
        """
            AAAAAAA
        """
        # Calcula el índice de la acción discreta
        discrete_action = int(action * self.num_intervals)
        return np.clip(discrete_action, 0, self.num_intervals-1)
    
    def undiscretize_action(self, discrete_action):
        """
            BBBBBBB
        """
        # Calcula el valor normalizado dentro del rango [0, 1]
        continuous_action = discrete_action / self.num_intervals
        return continuous_action

    def learn(self):
        """
			Train the agent network. Here is where the main DQN algorithm resides.
			Return:
				None
		"""

        targets_options = np.array([self.target_angle, 60, 90, 30, 60])

        epoch = 0
        epoch_save_checkspoints = self.num_episodes//5
        target_step = self.num_episodes//len(targets_options)
        i_target = 0
        u_option = 0

        ep_rews = []
        ep_loss = []

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
            state, info = self.env.reset()
            state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

            for t in count():
                action = self.select_action(state)

                action_step = self.undiscretize_action(action.cpu().detach().numpy())

                observation, reward, terminated, truncated, _ = self.env.step(action_step)

                # ///////////////////////////////////////////////////////////
                #reward = torch.tensor([reward], device=device) # Pendulum original target: theta=0°
                reward = self.calculate_reward(observation, action.item(), math.radians(self.target_angle))

                # ///////////////////////////////////////////////////////////

                done = terminated or truncated
                #done = terminated or truncated or (np.absolute(action.item())>2)

               
                ep_rews.append(reward)
                ## *********************************************************************************

                if terminated:
                    next_state = None
                else:
                    next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

                # Store the transition in memory
                self.memory_obj.push(state, action, next_state, reward)

                # Move to the next state
                state = next_state

                # Perform one step of the optimization (on the policy network)
                self.optimize_model(ep_loss)
                
                # Soft update of the target network's weights
                # θ′ ← τ θ + (1 −τ )θ′
                target_net_state_dict = self.target_net.state_dict()
                policy_net_state_dict = self.policy_net.state_dict()
                for key in policy_net_state_dict:
                    target_net_state_dict[key] = policy_net_state_dict[key]*self.tau + target_net_state_dict[key]*(1-self.tau)
                self.target_net.load_state_dict(target_net_state_dict)

                if done:
                    self.avg_rewards = self.avg_rewards + ep_rews
                    self.avg_loss = self.avg_loss + ep_loss
                    ep_rews = []
                    ep_loss = []
                    self._log_summary(i_episode, math.degrees(observation.item()))
                    break

            ## ***********************************************SAVE CHECKPOINTS*********************************************
            if epoch == epoch_save_checkspoints:
                epoch = 1
                # Guardar el modelo
                torch.save(self.policy_net.state_dict(), f"actor-critic/Pendulum_{i_episode}eps_DQN_discrete.pth")
            else:
                epoch += 1

        


    def optimize_model(self, ep_loss):
        if len(self.memory_obj) < self.batch_size:
            return
        transitions = self.memory_obj.sample(self.batch_size)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        # non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.bool)
        # non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        
        #////////////////////////////
        # Filter out terminated experiences before creating the mask
        non_terminal_states = [state for state in batch.next_state if state is not None]

        # Create the mask using the filtered states
        non_final_mask = torch.tensor(tuple(map(lambda s: True, non_terminal_states)), device=device, dtype=torch.bool)

        # Rest of your code using non_final_mask and non_terminal_states...
        non_final_next_states = torch.cat(non_terminal_states)

        #////////////////////////////

        state_batch = torch.cat([state for state in batch.state if state is not None])
        action_batch = torch.cat([action for action, state in zip(batch.action, batch.state) if state is not None]).unsqueeze(1)
        reward_batch = torch.cat([reward for reward, state in zip(batch.reward, batch.state) if state is not None])
        
        
        # state_batch = torch.cat(batch.state)
        # action_batch = torch.cat(batch.action).unsqueeze(1)
        # reward_batch = torch.cat(batch.reward)
        # Use non_terminal_states (filtered list from previous step)
        


        #///////////////////////////

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        
        state_action_values = self.policy_net(state_batch).gather(1,action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1).values
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        # next_state_values = torch.zeros(self.batch_size, dtype=torch.float32, device=device)

        # Use the length of non_terminal_states for the size
        next_state_values = torch.zeros(len(non_terminal_states), dtype=torch.float32, device=device)


        with torch.no_grad():
            actions_target = self.target_net(non_final_next_states)
            actions_index = actions_target.argmax(dim=1)
            print(non_final_mask.shape, non_final_next_states.shape)
            next_state_values[non_final_mask] = actions_target[non_final_mask, actions_index]  # ERROR ACA


        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
        print(loss)
        ep_loss.append(loss.item())
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
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
            value_discrete = self.policy_net(state).max(1).indices.view(1, 1)
        if sample > eps_threshold:
            newvalue = value_discrete
        else:
            #noise = np.random.normal(0, 0.3, size=None)
            #value_noise = undiscretize_action(value_discrete, n_actions) + noise
            #newvalue = discretize_action(value_noise, n_actions)
            newvalue = self.discretize_action(self.env.action_space.sample().item())
        
        # *************************************************************************
        if newvalue == 0:
            return torch.tensor([0], dtype=torch.long, device=device)
        else:
            return torch.tensor([newvalue], dtype=torch.long, device=device)
 

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

    def _log_summary(self, i_episode, pole_angle):
        """
			Print to stdout what we've logged so far in the most recent batch.

			Parameters:
				None

			Return:
				None
		"""
        print(f" >> Episodio #{i_episode} / {self.num_episodes} >> Target angle: {self.target_angle}")

        rewards_t = torch.tensor(self.avg_rewards, dtype=torch.float)
        loss_t = torch.tensor(self.avg_loss, dtype=torch.float)
        if len(rewards_t) >= 100:
            rew_means = rewards_t.unfold(0, 100, 1).mean(1).view(-1)
            rew_means = torch.cat((torch.zeros(99), rew_means))
            # loss_means = loss_t.unfold(0, 100, 1).mean(1).view(-1)
            # loss_means = torch.cat((torch.zeros(99), loss_means))
        
            # wandb.log({"Average Reward": rew_means.numpy(), "Average Loss": loss_means.numpy(), "Pole Angle": pole_angle})
            wandb.log({"Average Reward": rew_means.numpy(), "Pole Angle": pole_angle})

        
		



