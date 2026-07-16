import numpy as np
import time
from scale import SartoriusEntrisScale, FisherScale
from robot import *
import math
import gymnasium as gym 
from gymnasium import spaces
		

class MotionFaultException(Exception):
    def __init__(self, message, observation, reward, terminated, truncated, info):
        super().__init__(message)
        self.observation = observation
        self.reward = reward
        self.terminated = terminated  # standard gym naming for 'finished'
        self.truncated = truncated
        self.info = info

class WeighingEnv(gym.Env):
	"""
    	Gymnasium-compliant environment for robotic powder weighing.
    """
	metadata = {"render_modes": ["human"]}
	def __init__(self, robot_hostname, scale_port='/dev/ttyUSB0', gripper_port='/dev/ttyUSB1', library='franky', scale='entris', pitch_adjustment = False, new_setup = False, min_target=10, max_target=20, duration_based_shake=False, reward_type='negative_abs_error', early_stop=True):
		
		super().__init__()
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
		self.duration_based_shake = duration_based_shake
		self.step_no=0
		self.min_target = min_target
		self.max_target = max_target
		self.new_setup = new_setup
		self.finished=False
		self.target_weight = np.random.randint(self.min_target, self.max_target)
		self.library= library
		self.weight_obs_cap = 43
		#  add tracker of current weight for early stopping and reward shaping
		self.current_weight = 0
		self.pitch_adjustment = pitch_adjustment
		self.reward_type = reward_type
		self.past_error = None
		# Action space: [shake_intensity, incline_angle]
		self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
		self.early_stop = early_stop
		# Observation space: [current_weight/2, pitch*5 (or -5), target_weight/2]
		self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)


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

		print(pitch)
		#  the original work had the weights devided by 2 in the observation space. So do MOST of our agennts (see notes)
		self.current_weight = self.scale.get_weight()
		while self.current_weight is None:
			self.current_weight = self.scale.get_weight()

		if self.new_setup is True:
			observation = np.array([self.current_weight/2,pitch*5,self.target_weight/2])
		else:
			observation = np.array([self.current_weight/2,pitch*-5,self.target_weight/2])
		return observation.astype(np.float32)
	
	def render(self):
		return None
	
	def step(self, action):
		# print(f"Received action is {action}")
		motion_fault = False
		# if there's an incline exception try again
		counter =0
		while True:
			try:	
				self.robot.incline(action[1])
				break
			except InclineException:
				counter+=1
				if counter ==3:
					motion_fault = True
		counter=0
		if motion_fault is False:
			while True:
				try:
					if self.duration_based_shake:
						self.robot.shake(action[0], duration=0.12)
					else:
						self.robot.shake(action[0])
					break
				except ShakeException:
					# mark the step as if it had failed for it to be repeated
					print("The step was incomplete")
					motion_fault = True
					# raise Exception("Shake return failed. Environment needs to be reset")
			# sleep a bit for the scale to chill
			time.sleep(2)

		observation = self.get_observation()
		print(f'Current weight: {self.current_weight}, Target weight: {self.target_weight}')
		
		if self.reward_type=='negative_abs_error':
			reward = -np.abs(self.current_weight-self.target_weight)
			if reward >-1:
				reward +=1
			reward = reward * (1.1 **min(self.step_no, 10))
		else:
			error = np.abs(self.current_weight-self.target_weight)
			if self.past_error is None:
				reward = (self.target_weight-error)/self.target_weight
			else:
				reward = (self.past_error-error)/self.target_weight
			self.past_error = error
			reward = reward 
				
		if self.early_stop:
			#  if overdumped stop
			if self.current_weight - self.target_weight > 0:
				self.finished = True
			# early stop on 1mg approach
			elif(np.abs(self.current_weight-self.target_weight)<1):
				self.finished = True 	
		
		self.step_no+=1
		info={}
		
		truncated = self.step_no >= self.TOTAL_STEPS or motion_fault
		info["raw_weight_obs"] = self.current_weight - self.target_weight
		if self.finished or truncated:
			finalerror = np.abs(self.current_weight - self.target_weight)
			info["final_error"] = finalerror
		print(f'Step observation is{observation}, reward:{reward}')
		if motion_fault:
			print("Motion fault detected. Raising MotionFaultException.")
			raise MotionFaultException("Shake motion failed. Environment needs to be reset.", observation, reward, self.finished, True, info)
		return observation, reward, self.finished, truncated, info
				
		
	
	def reset(self, seed=None, options=None):

		super().reset(seed=seed)
		print('MSG: Environment reset ...')
		self.robot.reset()
		self.past_error=None
		if options is not None and 'target_weight' in options:
			self.target_weight = options['target_weight']
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
		
		return self.get_observation(), {}

 
