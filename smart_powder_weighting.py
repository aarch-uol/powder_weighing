from weighing_environment import WeighingEnv
import numpy as np
import torch
import os
import csv
import sys

class UserDefinedSettings(object):

    def __init__(self, BASE_RL_METHOD='SAC'):

        self.DEVICE = torch.device("cuda:0")
        self.ENVIRONMENT_NAME = 'powder_weighing_envII_small'
        dir_name = 'train'
        # self.HOST_NAME = socket.gethostname()
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
        self.MAX_EPISODE_LENGTH = 20
        self.ACTION_MAPPING_FLAG = True
        self.DOMAIN_PARAMETER_DIM=5
    
    def get_state_action_space(self):
       
        return STATE_DIM, ACTION_DIM

   
    def reset(self, target_weight=None):
        state = self.env.reset(target_weight=target_weight)
        # print(f"state is {state[:, 2:5].cpu().numpy().reshape(-1)}")
        return state

   
    def step(self, action, get_task_achievement=False):
        # print(action)
        # print(self.env)
        next_state, reward, done, task_achievement = self.env.step(action)
        # print(reward)
        if get_task_achievement:
            return next_state, reward, done, np.zeros(5), task_achievement
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


from SAC.SACAgent import SACAgent
import argparse
from scooping_mechanism import ScoopingMachine

def main():
    
    parser = argparse.ArgumentParser(description="Run scooping sequence with force feedback.")
    parser.add_argument("scooping_filename", help="JSON file with Pre/Post-scooping moves.")
    parser.add_argument("positions_filename", help="JSON file with container and spoon positions.")
    parser.add_argument("directory", help="Output directory name")
    parser.add_argument("model", help="Model to be used")
    parser.add_argument("powder", help="Name of powder to be used")
    args = parser.parse_args()


    print("Running full experiment on different powders on simulation trained agent")
    env = WeighingEnv('10.0.0.1', scale_port='/dev/ttyACM0', gripper_port='/dev/ttyUSB0')
    env = InterfaceEnvironment(env)
    settings = UserDefinedSettings()
    agent = SACAgent(env, settings)
    
    if env.env.library=='franky':
        scooper = ScoopingMachine(args.scooping_filename, args.positions_filename, verbose=True, robot=env.env.robot.robot)
    else:
        scooper = ScoopingMachine(args.scooping_filename, args.positions_filename, verbose=True)

    
    
    # powders = ['sand', 'salt', 'sugar', 'flour']
    
    directory = args.directory

    if not os.path.exists(directory):
        os.makedirs(directory)

    models = {
        'curriculum':'./models/SAC_ISAAC_POWDER_WEIGHING_CURICULLUM_ENVII_7_per_class_32025-03-16 17-25-44.862095',
        'random': './models/SAC_ISAAC_POWDER_RANDOM_WEIGHING_ENVII_7_per_class_32025-03-21 10-37-53.712456',
        'dr': './models/SAC_DR_ADHESION_ISAAC_POWDER_WEIGHING_ENVII2025-03-16 22-18-29.590903',
        'reverse':'./models/SAC_ISAAC_POWDER_WEIGHING_REVERSE_ENVII_7_per_class_32025-04-02 23-50-08.863929',
        'random_acute': './models/SAC_ISAAC_POWDER_WEIGHING_acute_angle'
    }

    model = args.model
    powder = args.powder

    print(f"Results destination is {directory}")
    try:
        model_path = models[model]
    except:
        print(f"""Model {model} not known. Exiting """)
        exit(1)

    scooper.load_powder()
    scooper.pickup_spoon()
    

    for target in range(15, 21, 5):
        skip = input(f'Skip current target weight {target} for current powder {powder} ? y/n: ')
        while skip.strip().lower() != 'n':
            if skip.strip().lower() == 'y':
                skip = True    
                break 
            skip = input(f'Skip current target weight {target} for current powder {powder} ? y/n: ')
                
        if skip == True:    
            continue
        
        with open(os.path.join(directory, f'experiment_{powder}_{target}g.csv'), 'w') as file:
            csv_writer = csv.writer(file, delimiter = ' ', )
            csv_writer.writerow(['Final Weight', 'Target weight', 'Error'])
            means =[]
            i=0
            while i<5:

                if not scooper.scoop():
                    input("System was not able to achieve a good scoop. Press ENTER to continue or close program...")
                    continue
                # try:
                agent.test(model_path=model_path, test_num=1, render_flag=False, target_weight=target)
                # except:
                #     print("Failed to load agent")
                #     continue
                i+=1 
                final_weight = env.env.get_observation()[0]*2
                print(f'Experiment results are: {[final_weight, target, abs(target-final_weight)]}')
                csv_writer.writerow([final_weight, target, abs(target-final_weight)])
                file.flush()
                means.append(abs(target-final_weight)) 
                scooper.reset_scoop_pose()

            print(means)
            csv_writer.writerow(['Average', np.mean(means), np.std(means)])
    scooper.drop_spoon()
    scooper.unload_powder()
        

if __name__=="__main__":
    main()