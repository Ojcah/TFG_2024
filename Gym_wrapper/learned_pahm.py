__credits__ = ["Pablo Alvarado"]

from os import path
from typing import Optional

import numpy as np

import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.classic_control import utils
from gymnasium.error import DependencyNotInstalled

from pahm_model import PAHMModel
import torch

class LearnedPAHM(gym.Env):
    """## Description

    The PAHM is a pendulum with a motor driven propeller. The input
    value is a normalized PWM driving signal, where 0 means the motor
    turned off and 1 means the highest motor speed.

    The output is the angle of the pendulum, measured agains a
    vertical ray pointing down the rotation axis of the pendulum.
    Positive angles are described counter clockwise.

    - `x-y`: cartesian coordinates of the pendulum's end in meters.
    - `theta` : angle in radians.
    - `pwm`: normalized PWM value to drive the motor.

    ## Action Space

    The action is a `ndarray` with shape `(1,)` representing the PWM value applied to the propeller motor

    | Num | Action | Min  | Max |
    |-----|--------|------|-----|
    | 0   | PWM    | 0.0  | 1.0 |

    ## Observation Space

    The observation is a `ndarray` with shape `(1,)` representing the angle of the pendulum

    | Num | Observation      | Min  | Max |
    |-----|------------------|------|-----|
    | 0   | theta            | -pi  | pi  |

    ## Rewards

    The reward function is not defined.

    ## Starting State

    The starting state is the resting position (angle 0, pendulum pointing down)

    ## Episode Truncation

    The episode truncates at 2000 time steps, or the value specified at construction time

    ## Arguments

    - `model_name`: .

    You should specify which model to use

    ```python
    >>> import gymnasium as gym
    >>> from learned_pahm import LearnedPAHM
    >>> env = learned_pahm()
    >>> observation, info = env.reset(seed=123, options={})
    >>> done = False
    >>> while not done:
    >>>    action = env.action_space.sample()
    >>>    observation,reward,terminated,truncated,info = env.step(action)
    >>>    done=terminated or truncated
    >>> env.close()
    
    For more information see https://gymnasium.farama.org/

    <TimeLimit<OrderEnforcing<PassiveEnvChecker<PendulumEnv<Pendulum-v1>>>>>
    >>> env.reset(seed=123, options={"low": -0.7, "high": 0.5})  # default low=-0.6, high=-0.5
    (array([ 0.4123625 ,  0.91101986, -0.89235795], dtype=float32), {})

    ```

    ## Version History

    * v0: Initial version release

    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30,
    }

    def __init__(self,
                 render_mode: Optional[str] = None,
                 model_name="pahm.pth",
                 model_normangles=True):
        
        self.model_name = model_name
        self.render_mode = render_mode

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pahm_model = PAHMModel.load_model(self.model_name)
        self.pahm_model.to(device)
        self.normangles = model_normangles

        # Initialize hidden state (layers,batch_size,hidden units )
        self.pahm_model.reset()

        self.screen_dim = 500
        self.screen = None
        self.clock = None
        self.isopen = True

        self.min_pwm = 0.0
        self.max_pwm = 1.0

        # Every Gym environment must have the attributes action_space
        # and observation_space
        
        # This will throw a warning in tests/envs/test_envs in
        #   utils/env_checker.py as the space is not
        #   symmetric. Ignoring the issue here as the values are
        #   meaningful in this context.
        self.action_space = spaces.Box(low=self.min_pwm,
                                       high=self.max_pwm,
                                       shape=(1,),
                                       dtype=np.float32)
        
        self.observation_space = spaces.Box(low=np.deg2rad(-120.0),
                                            high=np.deg2rad(120.0),
                                            shape=(1,), dtype=np.float32)

    def step(self, action):
        # Ultra basic computations will rely on this
        self.last_angle = self.state[0]
        self.last_action = action  # for rendering
        
        # The model produces an angle in a normalized frame from -1 to 1.
        pwm_sample = torch.full((1,1), action).to(self.device)
        angle = self.pahm_model.predict(pwm_sample).cpu().numpy().item()
       
        # Denormalize the angle, which assumes the model was created
        # with normalized angles (default setting)
        angle = np.deg2rad(angle*120) if self.normangles else np.deg2rad(angle)
        
        self.state = np.array([angle])

        if self.render_mode == "human":
            self.render()

        angular_velocity = angle-self.last_angle
        terminated = np.abs(angle)>np.deg2rad(120)

        # A basic reward will be given if the pendulum is at rest
        reward = 2*np.exp(-np.abs(angular_velocity)) - 1
                    
        # truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`
        # observation, reward, terminated, truncated,info
        return self.state, reward, terminated, False, {}

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        self.pahm_model.reset()
        self.state = np.array([0])

        self.last_action = None
        self.last_angle = None

        if self.render_mode == "human":
            self.render()
            
        return self.state, {}

    def render(self):
        if self.render_mode is None:
            assert self.spec is not None
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym.make("{self.spec.id}", render_mode="rgb_array")'
            )
            return

        try:
            import pygame
            from pygame import gfxdraw
        except ImportError as e:
            raise DependencyNotInstalled(
                'pygame is not installed, run `pip install "gymnasium[classic_control]"`'
            ) from e

        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode(
                    (self.screen_dim, self.screen_dim)
                )
            else:  # mode in "rgb_array"
                self.screen = pygame.Surface((self.screen_dim, self.screen_dim))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        self.surf = pygame.Surface((self.screen_dim, self.screen_dim))
        self.surf.fill((255, 255, 255))

        bound = 2.2
        scale = self.screen_dim / (bound * 2)
        offset = self.screen_dim // 2

        rod_length = 1 * scale
        rod_width = 0.2 * scale
        l, r, t, b = 0, rod_length, rod_width / 2, -rod_width / 2
        coords = [(l, b), (l, t), (r, t), (r, b)]
        transformed_coords = []
        for c in coords:
            c = pygame.math.Vector2(c).rotate_rad(self.state[0] - np.pi / 2)
            c = (c[0] + offset, c[1] + offset)
            transformed_coords.append(c)

        gfxdraw.filled_polygon(self.surf, transformed_coords, (31,47,95,255))
        gfxdraw.aapolygon(self.surf, transformed_coords, (31,47,95,255))

        gfxdraw.filled_circle(
            self.surf, offset, offset, int(rod_width / 2), (31,47,95,255)
        )
        gfxdraw.aacircle(self.surf, offset, offset, int(rod_width / 2), (31,47,95,255))

        rod_end = (rod_length, 0)
        rod_end = pygame.math.Vector2(rod_end).rotate_rad(self.state[0] - np.pi / 2)
        rod_end = (int(rod_end[0] + offset), int(rod_end[1] + offset))
        gfxdraw.filled_circle(
            self.surf, rod_end[0], rod_end[1], int(rod_width / 2), (31,47,95,255)
        )
        gfxdraw.aacircle(
            self.surf, rod_end[0], rod_end[1], int(rod_width / 2), (31,47,95,255)
        )

        if self.last_u is not None:
            # The PWM action is drawn as a line showing the wind
            # direction blown by the propeller
            wind = (rod_length,rod_length*self.last_u)

            wind = pygame.math.Vector2(wind).rotate_rad(self.state[0] - np.pi / 2)
            wind = (int(wind[0] + offset), int(wind[1] + offset))

            gfxdraw.line(self.surf,rod_end[0], rod_end[1], wind[0], wind[1], (210,64,64))

        # drawing axle
        gfxdraw.filled_circle(self.surf, offset, offset, int(0.05 * scale), (0, 0, 0))
        gfxdraw.aacircle(self.surf, offset, offset, int(0.05 * scale), (0, 0, 0))

        self.surf = pygame.transform.flip(self.surf, False, True)
        self.screen.blit(self.surf, (0, 0))
        if self.render_mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()

        else:  # mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )

    def close(self):
        if self.screen is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.isopen = False
