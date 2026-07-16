import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from weighing_environment import WeighingEnv
import time
import numpy as np
import math

env = WeighingEnv('10.0.0.1', scale_port='/dev/ttyACM1', gripper_port='/dev/ttyUSB0', pitch_adjustment=False, new_setup=True, min_target=10, max_target=20, duration_based_shake=True)
env.reset()

actions = np.array([[0.12612994742010758,0.024618991718629968]])
try:
    while True:
        
        action_cycler = 0 
        for i in range(10):
            # action = np.random.uniform(-1,1, (2))
            print(f'Action send is : {actions[action_cycler]}')
            # obs = env.step(action)
            obs=env.step(actions[action_cycler])
            print(obs)
            print(math.degrees(env.robot.get_pitch()))
            action_cycler = (action_cycler + 1) % len(actions)
            
        

        env.reset()
except KeyboardInterrupt:
    print('Exiting')
