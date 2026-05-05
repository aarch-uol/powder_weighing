import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from weighing_environment import WeighingEnv
import time
import numpy as np
import math

env = WeighingEnv('10.0.0.1', scale_port='/dev/ttyACM0', gripper_port='/dev/ttyUSB0', pitch_adjustment=False, min_target=10, max_target=20, normalisation=True, step_observation=True)
env.reset()

try:
    while True:
        
        for i in range(10):
            action = np.random.uniform(-1,1, (2))
            print(f'Action send is : {action}')
            # obs = env.step(action)
            obs=env.step(np.array([1,-1]))
            print(obs)
            print(math.degrees(env.robot.get_pitch()))
            
        

        env.reset()
except KeyboardInterrupt:
    print('Exiting')
