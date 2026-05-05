import numpy as np
import time
from scale import SartoriusEntrisScale, FisherScale
from robot import *
import math
		



class WeighingEnv:

	def __init__(self, robot_hostname, scale_port='/dev/ttyUSB0', gripper_port='/dev/ttyUSB1', library='franky', scale='entris', step_observation=False, normalisation = False, pitch_adjustment = False, min_target=10, max_target=20):
		
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
		self.min_target = min_target
		self.max_target = max_target
		self.finished=False
		self.target_weight = np.random.randint(self.min_target, self.max_target)
		self.library= library
		self.add_step_observation = step_observation
		self.normalisation = normalisation
		self.weight_obs_cap = 43
		#  add tracker of current weight for early stopping and reward shaping
		self.current_weight = 0
		self.pitch_adjustment = pitch_adjustment

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
		if self.pitch_adjustment:
			pitch = self.robot.get_pitch() - np.pi/2
		else:
			pitch = self.robot.get_pitch()

		#  the original work had the weights devided by 2 in the observation space. So do MOST of our agennts (see notes)
		self.current_weight = self.scale.get_weight()
		while self.current_weight is None:
			self.current_weight = self.scale.get_weight()
		if self.normalisation == True:
			current_weight = 2*(self.current_weight)/self.weight_obs_cap-1
			# clip to 1
			current_weight = min(current_weight, 1)
			target_weight = 2*(self.target_weight-(self.min_target))/(self.max_target-self.min_target)-1
		    
			step_no  = 2*self.step_no/self.TOTAL_STEPS-1
			print(f"robot pitch is {pitch} robot minimum pitch is {self.robot.pitch_min} robot maximum pitch is {self.robot.pitch_max}")
			pitch = 2*(pitch - self.robot.pitch_min)/(self.robot.pitch_max-self.robot.pitch_min)-1 
			if self.add_step_observation:
				observation = np.array([current_weight, pitch, step_no, target_weight])
			else:
				observation = np.array([current_weight, pitch, target_weight])
		else:
			observation = np.array([self.current_weight/2,pitch*-5,self.target_weight/2])
		return observation
	def step(self, action):
		
		# print(f"Received action is {action}")
		
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
				print("The step was incomplete")
				raise Exception("Shake return failed. Environment needs to be reset")
		# sleep a bit for the scale to chill
		time.sleep(2)
		observation = self.get_observation()
		print(f'Current weight: {self.current_weight}, Target weight: {self.target_weight}')
		reward = -np.abs(self.current_weight-self.target_weight)
		if reward >-1:
			reward +=1
		reward = reward * (1.1 **min(self.step_no, 10))
		
		#  if overdumped stop
		if self.current_weight - self.target_weight > 0:
			self.finished = True
		# early stop on 1mg approach
		elif(np.abs(self.current_weight-self.target_weight)<1):
			self.finished = True 	
		
		self.step_no+=1
		
		print(f'Step observation is{observation}, reward:{reward}')
		return observation, reward, self.finished, observation[0] 
				
		
	
	def reset(self, target_weight=None):
		print('MSG: Environment reset ...')
		self.robot.reset()
		if target_weight is not None:
			self.target_weight = target_weight
		else: 
			self.target_weight = np.random.randint(self.min_target, self.max_target)
		
		time.sleep(3)
		self.scale.reset()
		
		time.sleep(1)
		current_weight = self.scale.get_weight()
		while current_weight is None:
			current_weight = self.scale.get_weight()
		if abs(current_weight)>1:
			input('Manual scale reset needed. Press ENTER to continue')
			self.scale.reset()
		self.step_no=0
		self.finished=False
		print('MSG: Environment reset successfully')
		return self.get_observation()

 
