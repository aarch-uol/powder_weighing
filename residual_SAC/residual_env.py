import gymnasium as gym
import numpy as np
from SAC.SACAgent import SACAgent
import argparse
from scooping_machine import ScoopingMachine
import torch
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from weighing_environment import MotionFaultException

class UserDefinedSettings(object):

    def __init__(self, BASE_RL_METHOD='SAC'):

        self.DEVICE = torch.device("cuda:0")
        self.ENVIRONMENT_NAME = 'powder_weighing_envII_small'
        dir_name = 'train'
        self.LOG_DIRECTORY = os.path.join(os.environ['HOME'], 'logs', self.ENVIRONMENT_NAME, BASE_RL_METHOD, dir_name)

        self.LSTM_FLAG = True
        self.DOMAIN_RANDOMIZATION_FLAG = True

        self.BASE_RL_METHOD = BASE_RL_METHOD
        self.seed = 1337
        self.save_image = True

        self.num_steps = 1e6
        self.batch_size = 16
        self.policy_update_start_episode_num = 20
        self.learning_episode_num = 4000
        self.total_episode_num = 4000

        self.learning_rate = self.lr = 1e-4

        self.HIDDEN_NUM = 128
        self.onPolicy_distillation = True
        self.entropy_tuning_scale = 1.

        self.memory_size = 1e6
        self.gamma = 0.99
        self.soft_update_rate = 0.005
        self.entropy_tuning = True
        self.entropy_coefficient = 0.2
        self.multi_step_reward_num = 1
        self.updates_per_step = 1
        self.target_update_interval = 1  # episode num
        self.evaluate_interval = 10  # episode num
        self.initializer = 'xavier'
        self.run_num_per_evaluate = 3
        self.average_num_for_model_save = self.run_num_per_evaluate
        self.LEARNING_REWARD_SCALE = 1.
        self.MODEL_SAVE_INDEX = 'test'  # test, train

        self.ACTION_DISCRETE_FLAG = False

        self.TEST_FLAG = True
        if self.TEST_FLAG:
            self.TEST_DIR = None

        self.RENDER_FLAG = False
        self.network_type = 'basic'

        self.current_episode_num = 0

        self.TEST_DIR = 'test_dir'

        self.goal = 1.0

        self.flag = '1111111111'

from abc import ABCMeta, abstractmethod

class InterfaceEnvironment():
    def __init__(self, env):
        self.env=env
        self.STATE_DIM = 3
        self.ACTION_DIM = 2
        self.MAX_EPISODE_LENGTH = 10
        self.ACTION_MAPPING_FLAG = True
        self.DOMAIN_PARAMETER_DIM=5
    
    def get_state_action_space(self):
       
        return self.STATE_DIM, self.ACTION_DIM

   
    def reset(self, target_weight=None):
        state, _ = self.env.reset(options={'target_weight': target_weight} if target_weight is not None else None)
        return state

   
    def step(self, action, get_task_achievement=False):
        next_state, reward, done, truncated, info = self.env.step(action)
        
        if get_task_achievement:
            return next_state, reward, done, np.zeros(5), info["raw_weight_obs"]
        return next_state, reward, done, np.zeros(5)
    
    @abstractmethod
    def get_max_episode_steps(self):
        return self.env._max_episode_steps

    
    def random_action_sample(self):
        action = np.random.rand(2)
        action = 2. * (action - 0.5)
        return action

    @abstractmethod
    def render(self):
        pass

    @abstractmethod
    def __del__(self):
        pass

class ResidualAwareWrapper(gym.Wrapper):
    """
    A wrapper that appends the base controller's planned action to the observation,
    allowing the agent to see what the heuristic intends to do before adding its residual.
    """
    def __init__(self, env, scooping_machine, action_scaling_factor=0.1, model_path=None):
        """
        :param env: The original environment to wrap.
        :param scooping_machine: An instance of the ScoopingMachine class to perform the sco
        :param action_scaling_factor: The maximum magnitude of the residual action.
        :param model_path: Path to the pre-trained base controller model.
        :param powders: List of powders to be used in the environment.
        :param total_episodes: Total number of episodes for the training. These are divided 
                    over the number of powders for equal training. 
        """
        super().__init__(env)
        
        # 1. Restrict the Action Space to safe residual bounds (+/- 0.2)
        self.action_space = gym.spaces.Box(
            low=-1.0, 
            high=1.0, 
            shape=self.env.action_space.shape, 
            dtype=np.float32
        )
        
        # 2. Expand the Observation Space
        # Original obs (shape: 3) + Base Action (shape: 2) = New obs (shape: 5)
        orig_obs_shape = self.env.observation_space.shape[0]
        base_act_shape = self.env.action_space.shape[0]
        
        self.observation_space = gym.spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(orig_obs_shape + base_act_shape,), 
            dtype=np.float32
        )
        self.action_scaling_factor = action_scaling_factor
        # initialise the base controller
        settings = UserDefinedSettings()
        self.base_controller = SACAgent(InterfaceEnvironment(self.env), settings)
        # save the model
        self.base_controller.load_model(model_path)
        self.scooping_machine = scooping_machine
        # State tracker for the base action
        self.current_base_action = np.zeros(base_act_shape, dtype=np.float32)

    def reset(self, **kwargs):
        """
            Reset the environment and scoop
        """
        self.step_num = 0
        #  the spoon needs to be reloaded on reset 
        while(True):
            try:
                scoop_success, scoop_angle = self.scooping_machine.scoop(vision_check=True, starting_angle=40, length=0.02)
                break
            except Exception as e:
                print(f"Error occurred while scooping: {e}")
            

        # 1. Get original observation from the physical environment
        obs, info = self.env.reset(**kwargs)
        print(f"Reset observation: {obs}, Info: {info}")
        # 2. Calculate the base action for this initial state
        self.current_base_action, _ = self.base_controller.actor.get_action(obs, step=0, deterministic=True)
        
        # 3. Append base action to the observation
        augmented_obs = np.concatenate([obs, self.current_base_action], dtype=np.float32)
        
        print(f"Reset augmented observation: {augmented_obs}, Base action: {self.current_base_action}")
        return augmented_obs, info

    def step(self, residual_action):
        # 1. Combine previous base action with the agent's residual
        final_action = self.current_base_action + residual_action*self.action_scaling_factor
        # Clip the action to the expected space
        final_action = np.clip(final_action, -1.0, 1.0)
        print(f"Base intended: {self.current_base_action}, Agent added: {residual_action*self.action_scaling_factor}, Executed: {final_action}")
        # 2. Step the physical environment
        try: 
            next_obs, reward, terminated, truncated, info = self.env.step(final_action)
        except MotionFaultException as e:
            self.current_base_action, _ = self.base_controller.actor.get_action(e.observation, step=self.step_num, deterministic=True)
            augmented_next_obs = np.concatenate([e.observation, self.current_base_action], dtype=np.float32)
            return augmented_next_obs, e.reward, e.terminated, e.truncated, e.info
        # 3. Calculate the NEXT base action based on the new state
        self.current_base_action, _ = self.base_controller.actor.get_action(next_obs, step=self.step_num, deterministic=True)
        
        # 4. Append the next base action to the next observation
        augmented_next_obs = np.concatenate([next_obs, self.current_base_action], dtype=np.float32)
        
        # 5. Log actions in the info dict so you can easily compare approaches
        info["base_action"] = self.current_base_action
        info["residual_action"] = residual_action
        info["executed_action"] = final_action

        print(f"Residual Step observation: {augmented_next_obs}, Reward: {reward}, ")
        
        
        
        return augmented_next_obs, reward, terminated, truncated, info