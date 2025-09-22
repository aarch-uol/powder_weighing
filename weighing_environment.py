import numpy as np
import time
from Scale import SartoriusEntrisScale, FisherScale
from Robot import *

		



class WeighingEnv:

	def __init__(self, robot_hostname, scale_port='/dev/ttyUSB0', gripper_port='/dev/ttyUSB1', library='franky', scale='entris'):
		
		if scale == 'entris':
			self.scale = SartoriusEntrisScale(scale_port)
		else:
			scale= FisherScale(scale_port)
		if library == 'franky':
			self.robot=FrankyRobot(robot_hostname, gripper_port)
		elif library == 'pandapy':
			self.robot = PandaPyRobot(robot_hostname, gripper_port) 
		else:
			print("Unknown robot control library requested")
			exit(1)
		# move robot to the initial position
		self.TOTAL_STEPS=10
		self.step_no=0
		self.finished=False
		# self.robot.reset()
		# self.robot.load_tool()
		self.target_weight = np.random.randint(5, 15)
		self.library= library

	def __shake_surplus(self):
		"""
			Small shake movement to dump extra material after loading spoon
			Call to even out before actual dumping
			This is needed as the trnsportation in between scooping 
			and dumnping would naturaly remove excess powders from spoon
		"""
		for i in range(0,5):
			try:
				self.robot.shake(-0.80)
			except:
				continue

	def get_observation(self):
		# adjust the pitch angle to relfect the angle seen in the simulator
		pitch = self.robot.get_pitch() - np.pi/2
		# print(np.rad2deg(pitch))
		#  the original work had the weights devided by 2 in the observation space. So do MOST of our agennts (see notes)
		# return np.array([0,pitch*-5,0])
		return np.array([self.scale.get_weight()/2,pitch*-5,self.target_weight/2])

	def step(self, action):
		
		# print(f"Received action is {action}")
		step_complete=True
		# if there's an incline exception try again
		counter =0
		while True:
			try:	
				self.robot.incline(action[1])
				break
			except InclineException:
				counter+=1
				if counter ==3:
					raise Exception('Environment needs to be reset')
		counter=0
		while True:
			try:
				self.robot.shake(action[0])
				break
			except ShakeException:
				# mark the step as if it had failed for it to be repeated
				step_complete = False
			except ReturnException:
				raise Exception("Shake return failed. Environment needs to be reset")
		# sleep a bit for the scale to chill
		time.sleep(2)
		observation = self.get_observation()
		reward = -np.abs(observation[0]*2-observation[2]*2)
		if reward >-1:
			reward +=1
		reward = reward * (1.1 **min(self.step_no, 10))
		
		#  if overdumped stop
		if observation[0]*2 - observation[2]*2 > 0:
			self.finished = True
		# early stop on 1mg approach
		elif(np.abs(observation[0]*2-observation[2]*2)<1):
			self.finished = True 	
		if step_complete:
			self.step_no+=1
		
		print(f'Step observation is{observation}, reward:{reward}')
		return observation, reward, self.finished, observation[0] 
				
		
	
	def reset(self, target_weight=None):
		print('MSG: Environment reset ...')
		self.robot.reset()
		if target_weight is not None:
			self.target_weight = target_weight
		else: 
			self.target_weight = np.random.randint(5, 15)
		
		time.sleep(3)
		# self.robot.load_tool()
		# self.__shake_surplus()
		self.scale.reset()
		
		time.sleep(1)
	# if  isinstance(self.scale, FisherScale):
		if abs(self.scale.get_weight() - 0)>1:
			input('Manual scale reset needed. Press ENTER to continue')
			self.scale.reset()
		# time.sleep(5)
		self.step_no=0
		self.finished=False
		print('MSG: Environment reset successfully')
		return self.get_observation()

 
