import pyRobotiqGripper
import time
gripper = pyRobotiqGripper.RobotiqGripper()		
gripper.activate()

gripper.open()
time.sleep(5)
gripper.close()