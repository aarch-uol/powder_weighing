import panda_py
# import serial
# import pyRobotiqGripper
import numpy as np
import time
import roboticstoolbox as rtb
from spatialmath import SE3

# class Scale:
# 	def __init__(self, scale_port):
# 		self.serial_com = serial.Serial(
# 			port=scale_port,
# 			baudrate=9600,
# 			timeout=1
# 		)
		
# 	def read_weight(self):
# 		# counter to check if the weight reading is stable
# 		counter = 0
# 		value = None
# 		while counter < 2:
# 			if self.serial_com.in_waiting > 0:
# 				data = self.serial_com.readline().decode('utf-8').strip()
# 				data=data.replace('?', '')
# 				data = int(data)
# 				if data == value:
# 					counter+=1
# 				value = data
# 		return value
						
class Robot:
	def __init__(self, robot_hostname):
		# self.gripper = pyRobotiqGripper.RobotiqGripper(portname='/dev/ttyUSB0')
		# self.gripper.activate()
		self.robot = panda_py.Panda(robot_hostname)
		self.robot.move_to_start()
		self.shake_scale=40
		self.panda_model = rtb.models.Panda()
		current_tcp = self.panda_model.tool
		yaw = SE3.Rz(-np.pi/4)
		roll = SE3.Rx(-np.pi/32)
		new_tcp = yaw*current_tcp*roll
		self.panda_model.tool = new_tcp
		self.pitch_max = np.pi/6
		self.pitch_min = -np.pi/6
		self.panda_model.links[1].qlim=np.array([-0.05,0.05])
		print(self.panda_model)
		self.initial_pos = np.array([0.0, -0.27, 0.0, -2.9, 0.0, 2.53, 0.78])
		# self.initial_pos = self.panda_model.qr	
		self.current_tcp = self.panda_model.fkine(self.initial_pos)
		self.robot.move_to_joint_position(self.initial_pos)
		self.current_pose = self.initial_pos


	def move_to_joint_position(self, joint_pos):
		self.robot.move_to_joint_position(joint_pos)

	def load_tool(self):
		self.gripper.open()
		time.sleep(5)
		self.gripper.close()
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
		displacement = ((shake_amplitude+1.0)/2)/self.shake_scale
		print(f'Dsiaplecemtn value is {displacement}m')
		pose = self.robot.get_pose()
		init_pose = np.copy(pose)
		
		pose[0,3] -=displacement*pose[0,0]
		pose[1,3] -=displacement*pose[1,0]
		pose[2,3] -=displacement*pose[2,0]
		
		print(f'Pose is {pose}')
		print(f'Initial Pose is {init_pose}')
		q = panda_py.ik(pose)
		print(pose)
		self.robot.move_to_pose(pose,
						   speed_factor=0.1,
						   success_threshold=0.001)
		
		self.robot.move_to_joint_position(q)
		time.sleep(1)
	
		self.robot.move_to_start()
		
	def shake_jtraj(self, shake_amplitude):
		displacement = -((shake_amplitude+1.0)/2)/self.shake_scale
		print(f'Dsiaplecemtn value is {displacement[0]}m')
		target_tcp = self.current_tcp * SE3(displacement[0], 0,0)
		
		new_translation= np.array([self.current_tcp.t[0]+displacement[0],
							self.current_tcp.t[1],
							self.current_tcp.t[2]])
		target_tcp = SE3.Rt(self.current_tcp.R, new_translation)

		target_pos = self.panda_model.ik_LM(target_tcp, tol=1e-7, ilimit=100, slimit=300)[0]
		traj = rtb.jtraj(self.initial_pos, target_pos, 25)
		traj_return = rtb.jtraj(target_pos, self.initial_pos, 25)
	
		# self.panda_model.plot(waypoints, backend='pyplot', movie='panda_motion1.gif')


		traj_list = [q.reshape(7,1) for q in traj.q]
		traj_return_list = [q.reshape(7,1) for q in traj_return.q]
		self.robot.move_to_joint_position(traj_list, speed_factor=0.1)
		self.robot.move_to_joint_position(traj_return_list, speed_factor=0.1)

	def shake_ctraj(self, shake_amplitude):
		displacement = -((shake_amplitude+1.0)/2)/self.shake_scale
		print(f'Dsiaplecemtn value is {displacement[0]}m')
		print('Initial TCP is')
		print(self.current_tcp)
		# print(self.robot.get_pose())
		print('The displacement is')
		print(SE3(displacement[0], 0,0))
		target_tcp = self.current_tcp * SE3(displacement[0], 0,0)
		
		# new_translation= np.array([self.current_tcp.t[0]+displacement[0],
		# 					self.current_tcp.t[1],
		# 					self.current_tcp.t[2]])
		# target_tcp = SE3.Rt(self.current_tcp.R, new_translation)

		print('target to achieve is')
		print(target_tcp)
		# print('WHat it actually achieves is')
		# print(self.panda_model.fkine(target_pos))
		
		
		trajc = rtb.ctraj(self.current_tcp, target_tcp, 10)
		trajc_return = rtb.ctraj(target_tcp, self.current_tcp, 10)

		# traj_list=[self.panda_model.ik_LM(tcp_point, tol=1e-7, ilimit=100, slimit=300)[0].reshape(7,1) for tcp_point in trajc]
		# traj_return_list=[self.panda_model.ik_LM(tcp_point, tol=1e-7, ilimit=100, slimit=300)[0].reshape(7,1) for tcp_point in trajc_return]
		# self.panda_model.plot(waypoints, backend='pyplot', movie='panda_motion1.gif')

		for pose in trajc.data:
			print(pose)
			self.robot.move_to_pose(pose, speed_factor=0.1)
		for pose in trajc_return.data:
			self.robot.move_to_pose(pose, speed_factor=0.1)
		# self.robot.move_to_pose(trajc.data, speed_factor=0.1)
		# self.robot.move_to_pose(trajc_return.data, speed_factor=0.1)


	def shake_ctraj2(self, shake_amplitude):
		displacement = -((shake_amplitude+1.0)/2)/self.shake_scale
		if(displacement[0]==0):
			return
		# print(f'Dsiaplecemnt value is {displacement[0]}m')
		# print('Initial TCP is')
		# print(self.current_tcp)
		# print(self.robot.get_pose())
		# print('The displacement is')
		# print(SE3(displacement[0], 0,0))
		target_tcp = self.current_tcp * SE3(displacement[0], 0,0)		
		
		trajc = rtb.ctraj(self.current_tcp, target_tcp, 5)
		trajc_return = rtb.ctraj(target_tcp, self.current_tcp, 5)

		traj = self.panda_model.ikine_LM(trajc, q0=self.current_pose, tol=1e-7, ilimit=100, slimit=300) 
		traj_return = self.panda_model.ikine_LM(trajc_return, q0=traj.q[-1], tol=1e-7, ilimit=100, slimit=300) 

		waypoints = np.vstack((traj.q, traj_return.q))
		# self.panda_model.plot(waypoints, backend='pyplot', movie='panda_motion1.gif')
		traj_list = [q.reshape(7,1) for q in traj.q]
		traj_return_list = [q.reshape(7,1) for q in traj_return.q]
		# print(traj)
		self.robot.move_to_joint_position(traj_list, speed_factor=0.1)
		self.robot.move_to_joint_position(traj_return_list, speed_factor=0.1)
		self.current_pose = traj_return.q[-1]

	def incline(self, incline_angle):
		incline_action = -3*np.pi/180 + (incline_angle+1.0)/2 * (6*np.pi/180) 
		new_pitch = self.get_pitch() - incline_action[0]
		action = incline_action[0]
		if action ==0:
			return
		# print(self.get_pitch(), action, new_pitch, self.pitch_min, self.pitch_max)
		if new_pitch>self.pitch_max:
			action = 0
		elif new_pitch < self.pitch_min:
			action = 0
		target_tcp = self.current_tcp * SE3.Ry(action)
		trajc = rtb.ctraj(self.current_tcp, target_tcp, 5)
		traj = self.panda_model.ikine_LM(trajc, q0=self.current_pose, tol=1e-7, ilimit=100, slimit=300) 
		# self.panda_model.plot(traj.q, backend='pyplot', movie='panda_motion1.gif')
		traj_list = [q.reshape(7,1) for q in traj.q]
		self.robot.move_to_joint_position(traj_list, speed_factor=0.2)
		self.current_tcp = target_tcp
		#  the current pose is the last pose executed
		self.current_pose = traj.q[-1]

	def reset(self):
		self.robot.move_to_start()
		self.current_tcp = self.panda_model.fkine(self.initial_pos)
		self.robot.move_to_joint_position(self.initial_pos)
		self.current_pose = self.initial_pos


class WeighingEnv:

	def __init__(self, robot_hostname, scale_port='/dev/ttyUSB1', gripper_port='/dev/ttyUSB0'):
		
		# self.scale = Scale(scale_port)
		self.robot = Robot(robot_hostname) 
		# move robot to the initial position
		
		# self.robot.load_tool()
		self.target_weight = np.random.randint(5, 15)

	def get_observation(self):
		return np.array([self.scale.read_weight(),self.robot.get_pitch(),self.target_weight])

	def step(self, action):
		self.robot.incline(action[:,1])
		self.robot.shake_ctraj2(action[:,0])
		# self.get_observation()
	
	def reset(self):
		print('MSG: Environment reset ...')
		self.robot.reset()
		self.target_weight = np.random.randint(5, 15)
	

 
