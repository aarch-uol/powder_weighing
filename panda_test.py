import roboticstoolbox as rtb
from custom_panda import Panda
from swift import Swift
import time
import panda_py
import spatialgeometry as sg

panda_model = Panda()
robot = panda_py.Panda("10.0.0.1")
panda_model.q=robot.q
viz = Swift()
viz.launch( )
viz.add(panda_model, readonly=True)
tcp_viz=sg.Axes(0.1)
tcp_viz.T=panda_model.fkine(robot.q)
viz.add(tcp_viz)

while(True):
    input("Move robot to desired point then press ENTER ...")
    print(f'Sampled robot pose is : {robot.q}')
    panda_model.q = robot.q
    tcp_viz.T=panda_model.fkine(robot.q)
    time.sleep(0.05)
    viz.step()
    skip = input(f'Happy with robot pose? y/n: ')
    while skip.strip().lower() != 'n':
        if skip.strip().lower() == 'y':
            skip = True    
            exit() 
        skip = input(f'Happy with robot pose? y/n: ')