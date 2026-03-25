from pyrobotiqgripper import RobotiqGripper
import time
gripper = RobotiqGripper()		
gripper.activate()

gripper.open()
time.sleep(5)
gripper.close()