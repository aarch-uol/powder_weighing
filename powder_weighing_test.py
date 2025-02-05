from weighing_environment import WeighingEnv
import time
import numpy as np

# time.sleep(10)
env = WeighingEnv('10.6.203.101', '/dev/ttyUSB1')
# print(env.get_observation())
# env.step(np.array([[1,-1]]))
try:
    while True:
        # env.step(np.array([[1,-1]]))
        for i in range(10):
            action = np.random.uniform(-1,1, (1,2))
            print(f'Action send is : {action}')
            env.step(action)

        env.reset()
except KeyboardInterrupt:
    print('Exiting')
