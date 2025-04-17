import panda_py
import serial
import pyRobotiqGripper
import numpy as np
import time
import roboticstoolbox as rtb
from spatialmath import SE3
from swift import Swift
from custom_panda import Panda
import spatialgeometry as sg

class Scale:
	def __init__(self, scale_port):
		self.serial_com = serial.Serial(
			port=scale_port,
			baudrate=9600,
			timeout=1
		)
		self._weight=0
		
	def __read_weight(self):
		""" 
			Reads the incoming weight through serial
			Will wait for a stable weight before getting a 
			final reading

		"""
		
		# counter to check if the weight reading is stable
		counter = 0
		value = None
		data = None
		#  empty buffer 
		while self.serial_com.in_waiting > 0:
			data = self.serial_com.readline().decode('utf-8').strip()
		
		# make sure we have a stable reading
		skip=1
		while counter<=50:
			data = self.serial_com.readline().decode('utf-8').strip()
			# only compare evry 3rd reading 
			if skip<2:
				skip +=1
				continue
			skip = 1
			data=data.replace('?', '')
			data = int(data)
			if data==value:
				counter+=1
			else: 
				counter=0
			value=data
		
		return data

	def reset(self):
		self._weight=0

	def get_weight(self, timeout=90):
		"""
			The weight should not decrease, that is behaviour not seen
			in training, so if if the weight is smaller it reads again. 
			Occasionally the scale might get stuck on a lower value. 
			To overcome that we introduce a timeout, after which, the largest 
			observed value is taken
		"""
		start_time = time.time()
		weights=[]
		new_weight = self.__read_weight()
		#  if the error between this and last reading are greater than 1 the redo. 
		while(abs(new_weight-self._weight)>1):
			weights.append(new_weight)
			new_weight = self.__read_weight()
			elapsed=time.time() - start_time
			if elapsed>timeout:
				self._weight=max(weights)
				return self._weight
		self._weight = new_weight
		return self._weight

		
class ShakeException(Exception):
	pass

class ReturnException(Exception):
	pass

class InclineException(Exception):
	pass

class Robot:
	def __init__(self, robot_hostname, gripper_port):
		self.gripper = pyRobotiqGripper.RobotiqGripper(portname=gripper_port)
		try:
			self.gripper.activate()
		except:
			input('Gripper activation failed. Pres ENTER to conitnue or close program')
		self.robot = panda_py.Panda(robot_hostname)
		# self.robot.move_to_start()
		self.shake_scale=65
		self.panda_model = Panda()
		
		self.pitch_max = np.pi/6
		self.pitch_min = -0
		# self.panda_model.links[1].qlim=np.array([-0.05,0.05])
		
		# self.initial_pos = np.array([0.0, -0.27, 0.0, -2.9, 0.0, 2.53, 0.78])	
		# bellow is a test position for vial support 
		self.initial_pos = np.array([0.01373907, 0.02742517, -0.01605499, -2.39315202, -0.00844878, 2.22456631, 0.77540909])
		self.initial_tcp = self.panda_model.fkine(self.initial_pos)
		self.current_tcp = self.panda_model.fkine(self.initial_pos)
		
		self.robot.move_to_joint_position(self.initial_pos)
		
		self.current_pose = self.initial_pos
		# open swift to visualise	
		self.viz = Swift()
		self.viz.launch( )
		self.viz.add(self.panda_model, readonly=True)
		self.tcp_viz=sg.Axes(0.1)
		self.viz.add(self.tcp_viz)
		


	def move_to_joint_position(self, joint_pos):
		self.robot.move_to_joint_position(joint_pos)

	def load_tool(self):
		# self.gripper.open()
		# time.sleep(5)
		# self.gripper.close()
		print("Please make sure tool is properly in place on gripper")
		input('Press Enter to continue')
	
	def move_to_home(self):
		self.robot.move_to_start()

	def get_pitch(self):
		"""
			Return current pitch of the tcp in rads
		"""
		pose = self.panda_model.fkine(self.robot.q)
		# print(pose, pose.rpy())
		# pitch = np.arcsin(-pose[2,0])
		
		return pose.rpy()[1]


		

	def shake(self, shake_amplitude):
		"""
			Shake of the spoon on the x axis
		"""
		
		# update model pose wrt. the real robot
		self.current_pose = self.robot.get_state().q
		self.panda_model.q=self.current_pose
		self.current_tcp=self.panda_model.fkine(self.current_pose)
		# the shake should also correct any deviation from the vial space caused by
		# the incline


		tolerance = 1e-7
		displacement = -((shake_amplitude+1.0)/2)/self.shake_scale
		if displacement >= -0.001:
			return
		
		target_tcp = self.current_tcp * SE3(displacement, 0,0)		
		
		trajc = rtb.ctraj(self.current_tcp, target_tcp, 2)
		trajc_return = rtb.ctraj(target_tcp, self.current_tcp, 2)
		while True:
			try:
				traj = self.panda_model.ikine_LM(trajc, q0=np.tile(self.current_pose,(300,1)), tol=tolerance, ilimit=100, slimit=300) 
				traj_return = self.panda_model.ikine_LM(trajc_return, q0=np.tile(traj.q[-1], (300,1)), tol=tolerance, ilimit=100, slimit=300) 
				# check for large joint deviations
				if np.any(np.abs(traj.q-self.current_pose) > np.radians(10)) or np.any(np.abs(traj_return.q-self.current_pose) > np.radians(10)):
					continue
				break
			except:
				tolerance=tolerance*10
				if tolerance >1e-3:
					raise ShakeException('Trajectory failed to compute')
		waypoints = np.vstack((traj.q, traj_return.q))
		for q in waypoints:
			self.panda_model.q = q
			self.tcp_viz.T=self.panda_model.fkine(q)
			time.sleep(0.05)
			self.viz.step()
			
		
		# self.panda_model.plot(waypoints, block=True)
		traj_list = [q.reshape(7,1) for q in traj.q]
		traj_return_list = [q.reshape(7,1) for q in traj_return.q]
		
		try:
			self.robot.move_to_joint_position(traj_list, speed_factor=0.8)
		except:
			raise ShakeException('Failed to do forwards movement')
		try:
			self.robot.move_to_joint_position(traj_return_list, speed_factor=0.8)
		except:
			raise ReturnException('Failed to do return motion')




	def incline(self, incline_angle):
		"""
			Incline of the spoon on the pitch
		"""
		# update model pose wrt. the real robot
		self.current_pose = self.robot.get_state().q
		self.panda_model.q=self.current_pose
		self.current_tcp=self.panda_model.fkine(self.current_pose)

		incline_action = -3*np.pi/180 + (incline_angle+1.0)/2 * (6*np.pi/180) 
		new_pitch = self.get_pitch() - incline_action
		action = incline_action
		# print(new_pitch, self.pitch_max, self.pitch_min)
		# if action zero or action outside of limits return
		if action ==0:
			return
		elif new_pitch>self.pitch_max:
			return
		elif new_pitch < self.pitch_min:
			return
		# tcp tends to go forwards on negative incline so correct for that
		if incline_angle <0:
			target_tcp = self.current_tcp* SE3.Ry(action) *SE3(-0.0017, 0, 0)
		else:
			# for positive incline leave as is for now
			target_tcp = self.current_tcp* SE3.Ry(action) *SE3(0, 0, -0.0013)

		tolerance=1e-7
		while True:
			try: 
				traj = self.panda_model.ikine_LM(target_tcp, q0=np.tile(self.current_pose,(300,1)), tol=tolerance, ilimit=100, slimit=300) 
				if np.any(np.abs(traj.q-self.current_pose) > np.radians(10)):
					continue
				self.panda_model.q = traj.q
				self.tcp_viz.T=self.panda_model.fkine(traj.q)
				self.viz.step()
				time.sleep(1)
				self.robot.move_to_joint_position(traj.q, speed_factor=0.1)
				break
			except: 
				print(f'MSG: Incline failed. Reattempting with tolerance {tolerance}')
				tolerance = tolerance * 10
				if tolerance>1e-3:
					raise InclineException("Incline trajectory failed to compute")
	

	def reset(self):
		# self.robot.move_to_start()
		self.current_tcp = self.panda_model.fkine(self.initial_pos)
		self.robot.move_to_joint_position(self.initial_pos)
		self.current_pose = self.initial_pos


class WeighingEnv:

	def __init__(self, robot_hostname, scale_port='/dev/ttyUSB0', gripper_port='/dev/ttyUSB1'):
		
		self.scale = Scale(scale_port)
		self.robot = Robot(robot_hostname, gripper_port) 
		# move robot to the initial position
		self.TOTAL_STEPS=10
		self.step_no=0
		self.finished=False
		# self.robot.load_tool()
		self.target_weight = np.random.randint(5, 15)

	def __shake_surplus(self):
		"""
			Small shake movement to dump extra material after loading spoon
			Call to even out before actual dumping
			This is needed as the trnsportation in between scooping 
			and dumnping would naturaly remove excess powders from spoon
		"""
		for i in range(0,10):
			try:
				self.robot.shake(-0.75)
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
		self.step_no+=1
		print(f"Received action is {action}")

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
				counter+=1
				if counter==3:
					raise Exception('Environment needs to be reset')
			except ReturnException:
				raise Exception("Shake return failed. Environment needs to be reset")
		# sleep a bit for the scale to chill
		time.sleep(4)
		observation = self.get_observation()
		reward = -np.abs(observation[0]-observation[2])
		if reward >-1:
			reward +=1
		if self.step_no >= self.TOTAL_STEPS:
			self.finished = True
		
		print(f'Step observation is{observation}, reward:{reward}')
		return observation, reward, self.finished, observation[0] 
				
		
	
	def reset(self, target_weight=None):
		print('MSG: Environment reset ...')
		self.robot.reset()
		self.scale.reset()
		if target_weight is not None:
			self.target_weight = target_weight
		else: 
			self.target_weight = np.random.randint(5, 15)
		
		# time.sleep(5)
		self.robot.load_tool()
		self.__shake_surplus()
		if self.scale.get_weight()!=0:
			input('Manual scale reset needed. Press ENTER to continue')
			self.scale.reset()
		time.sleep(5)
		self.step_no=0
		self.finished=False
		return self.get_observation()

 
