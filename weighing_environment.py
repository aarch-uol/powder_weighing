import panda_py
import serial
import pyRobotiqGripper
import numpy as np
import time
import roboticstoolbox as rtb
from spatialmath import SE3



class Scale:
	def __init__(self, scale_port):
		self.serial_com = serial.Serial(
			port=scale_port,
			baudrate=9600,
			timeout=1
		)
		
	def read_weight(self):
		# counter to check if the weight reading is stable
		counter = 0
		value = None
		data = None
		#  empty buffer 
		while self.serial_com.in_waiting > 0:
			data = self.serial_com.readline().decode('utf-8').strip()
		
		# make sure we have a stable reading
		while counter<=10:
			data = self.serial_com.readline().decode('utf-8').strip()
			data=data.replace('?', '')
			data = int(data)
			if data==value:
				counter+=1
			else: 
				counter=0
			value=data

		return data


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
		self.shake_scale=50
		self.panda_model = rtb.models.Panda()
		
		self.pitch_max = np.pi/7
		self.pitch_min = -0
		self.panda_model.links[1].qlim=np.array([-0.05,0.05])
		
		self.initial_pos = np.array([0.0, -0.27, 0.0, -2.9, 0.0, 2.53, 0.78])	
		self.current_tcp = self.panda_model.fkine(self.initial_pos)
		
		self.robot.move_to_joint_position(self.initial_pos)
		current_tcp = self.panda_model.tool
		
		yaw = SE3.Rz(-np.pi/4)
		roll = SE3.Rx(-np.pi/180)
		current_tcp = current_tcp *SE3.Tx(0.26)
		new_tcp = current_tcp
		# print(self.current_tcp, new_tcp)
		self.panda_model.tool = new_tcp
		self.current_pose = self.initial_pos
		


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
		pose = self.robot.get_pose()
		pitch = np.arcsin(-pose[2,0])
		return pitch


		

	def shake(self, shake_amplitude):
		"""
		SHake of the spoon on the x axis
		"""
		displacement = -((shake_amplitude+1.0)/2)/self.shake_scale
		if(displacement==0):
			return
		target_tcp = self.current_tcp * SE3(displacement, 0,0)		
		
		trajc = rtb.ctraj(self.current_tcp, target_tcp, 2)
		trajc_return = rtb.ctraj(target_tcp, self.current_tcp, 2)

		traj = self.panda_model.ikine_LM(trajc, q0=self.current_pose, tol=1e-7, ilimit=100, slimit=300) 
		traj_return = self.panda_model.ikine_LM(trajc_return, q0=traj.q[-1], tol=1e-7, ilimit=100, slimit=300) 
		waypoints = np.vstack((traj.q, traj_return.q))
		# self.panda_model.plot(waypoints, backend='pyplot', movie='panda_motion1.gif')
		traj_list = [q.reshape(7,1) for q in traj.q]
		traj_return_list = [q.reshape(7,1) for q in traj_return.q]
		

		try:	
			self.robot.move_to_joint_position(traj_list, speed_factor=0.6)
		except:
			raise ShakeException("Trajectroy failed to compute") 
		
		
		while True: 
			try:	
				self.robot.move_to_joint_position(traj_return_list, speed_factor=0.6)
				break
			except: 
				print("Return failed. Trying to recompute trajectory")
				traj_return = self.panda_model.ikine_LM(trajc_return, q0=traj.q[-1], tol=1e-7, ilimit=100, slimit=300) 
				traj_return_list = [q.reshape(7,1) for q in traj_return.q]
				counter+=1
				if counter >=3:
					raise ReturnException("Cannot compute return trajectory")

		self.current_pose = traj_return.q[-1]


	def incline(self, incline_angle):
		incline_action = -3*np.pi/180 + (incline_angle+1.0)/2 * (6*np.pi/180) 
		new_pitch = self.get_pitch() - incline_action
		action = incline_action
		if action ==0:
			return
		# print(self.get_pitch(), action, new_pitch, self.pitch_min, self.pitch_max)
		if new_pitch>self.pitch_max:
			action = 0
		elif new_pitch < self.pitch_min:
			action = 0
		# get the current tooln positon
		current_tcp_tool = self.current_tcp * self.panda_model.tool
		# compute adjusted tool position
		target_tcp_tool = current_tcp_tool* SE3.Ry(action)
		# given tool position compute necesarry end efector position
		target_tcp = target_tcp_tool * self.panda_model.tool.inv()
		trajc = rtb.ctraj(self.current_tcp, target_tcp, 10)
		traj = self.panda_model.ikine_LM(trajc, q0=self.current_pose, tol=1e-7, ilimit=100, slimit=300) 
		# self.panda_model.plot(traj.q, backend='pyplot', movie='panda_motion1.gif')
		traj_list = [q.reshape(7,1) for q in traj.q]
		try: 
			self.robot.move_to_joint_position(traj_list, speed_factor=0.1)
		except: 
			raise InclineException("Incline trajectory failed to compute")

		self.current_tcp = target_tcp
		#  the current pose is the last pose executed
		self.current_pose = traj.q[-1]

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

	def get_observation(self):
		# adjust the pitch angle to relfect the angle seen in the simulator
		pitch = self.robot.get_pitch() - np.pi/2
		# print(np.rad2deg(pitch))
		#  the original work had the weights devided by 2 in the observation space. So do MOST of our agennts (see notes)
		return np.array([self.scale.read_weight()/2,pitch*-5,self.target_weight/2])

	def step(self, action):
		self.step_no+=1
		print(f"Received action is {action}")
		try:	
			self.robot.incline(action[1])
		except InclineException:
			self.robot.incline(action[1])
		try:
			self.robot.shake(action[0])
		except ShakeException:
			self.robot.shake(action[0])
		except ReturnException:
			Exception("Shake return failed. Environment needs to be reset")
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
		
		if target_weight is not None:
			self.target_weight = target_weight
		else: 
			self.target_weight = np.random.randint(5, 15)
		# input('Manual environment reset needed. Press ENTER to continue')
		# time.sleep(5)
		self.robot.load_tool()
		time.sleep(5)
		self.step_no=0
		self.finished=False
		return self.get_observation()

 
