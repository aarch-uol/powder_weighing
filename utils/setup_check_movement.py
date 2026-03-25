from weighing_environment import WeighingEnv
import time
import numpy as np


env = WeighingEnv('10.0.0.1', scale_port='/dev/ttyACM0', gripper_port='/dev/ttyUSB0')
env.reset()

try:
    while True:
        
        for i in range(10):
            action = np.random.uniform(-1,1, (2))
            print(f'Action send is : {action}')
            obs = env.step(action)
            obs=env.step(np.array([1,-1]))
            print(obs)
        

        env.reset()
except KeyboardInterrupt:
    print('Exiting')
