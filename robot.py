from abc import ABC, abstractmethod
import panda_py
#import pyRobotiqGripper
import roboticstoolbox as rtb
from swift import Swift
from custom_panda import Panda
from spatialmath import SE3
import numpy as np
import time 
import franky
from scipy.spatial.transform import Rotation as R

import spatialgeometry as sg

class ShakeException(Exception):
	pass

class ReturnException(Exception):
	pass

class InclineException(Exception):
	pass


class Robot(ABC):
	
	def __init__(self, robot_hostname, gripper_port):
		self.shake_scale=65
		self.pitch_max = 45*np.pi/180
		self.pitch_min = -0
	
	@abstractmethod
	def shake(self, shake_amplitude):
		pass

	@abstractmethod
	def incline(self, incline_angle):
		pass

	@abstractmethod
	def reset(self):
		pass

	@abstractmethod
	def get_pitch(self):
		pass

	def load_tool(self):
		self.gripper.open()
		time.sleep(5)
		self.gripper.close()
		print("Please make sure tool is properly in place on gripper")
		input('Press Enter to continue')

class FrankyRobot(Robot):
	def __init__(self, robot_hostname, gripper_port):
		
		super().__init__(robot_hostname,gripper_port)
		self.robot = franky.Robot(robot_hostname)
		self.robot.relative_dynamics_factor= franky.RelativeDynamicsFactor(
 			   velocity=1, acceleration=1, jerk=1
		)
		
		self.shake_dynamics_factor=[0.25, 0.13, 0.05]

	def set_shake_dynamics_factor(self, dynamics_factor: list[float]):
		self.shake_dynamics_factor=np.array(dynamics_factor)

	def reset(self):
		home_pos = [0.03241183, 0.02479055, -0.03119808, -2.4527532,  -0.01299625,  2.32665869, 0.78508846]
		self.robot.move(franky.JointMotion(home_pos,relative_dynamics_factor= franky.RelativeDynamicsFactor(
 			   velocity=0.1, acceleration=0.1, jerk=0.05
		)))
		
	def get_pitch(self):
		end_effector_pose = self.robot.current_pose.end_effector_pose
		current_quat = end_effector_pose.quaternion
		rot = R.from_quat(current_quat).as_matrix()
		euler = R.from_matrix(rot).as_euler('xyz')
		return euler[1]
		
	def incline(self, incline_angle):

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
		
		
		end_effector_pose = self.robot.current_pose.end_effector_pose
		position = end_effector_pose.translation
		current_quat = end_effector_pose.quaternion

		# x axis correction for inclination
		print(incline_action, new_pitch)
		if incline_angle<-0.15:
			if new_pitch > 15*np.pi/180:
				print("correction used")
				correction = franky.Affine(translation=np.array([-0.0015, 0, 0.0018]))
				position = (end_effector_pose * correction).translation
		else:
			if incline_angle>0:
				print("Declining correction")
				correction = franky.Affine(translation=np.array([-0.0001, 0, -0.0018]))
				position = (end_effector_pose * correction).translation
		
			
		rot = R.from_quat(current_quat).as_matrix()
		euler = R.from_matrix(rot).as_euler('xyz')
		euler[1] = new_pitch
		new_rot = R.from_euler('xyz', euler).as_matrix()
		new_quat = R.from_matrix(new_rot).as_quat()
			
		motion = franky.CartesianMotion(franky.RobotPose(franky.Affine(position, new_quat)), relative_dynamics_factor= franky.RelativeDynamicsFactor(
 			   velocity=0.05, acceleration=0.05, jerk=0.05
		))
		self.robot.move(motion)
		
			
		achieved_quat = self.robot.current_pose.end_effector_pose.quaternion
		achieved_pitch = R.from_quat(achieved_quat).as_euler('xyz', degrees=True)[1]
		return True
	
	def shake(self, shake_amplitude):
		# get current pose and orientation

		ee_pose = self.robot.current_pose.end_effector_pose

		rot = R.from_quat(ee_pose.quaternion)
	

		displacement = -((shake_amplitude+1.0)/2)/self.shake_scale
		if displacement >= -0.0009:
			return
		displacement = rot.apply(np.array([displacement, 0, 0]))
		# compute x axis translation
		x_translation = franky.Affine(translation=displacement)
		
		displaced_position = x_translation * ee_pose
	

		shake_motion = franky.CartesianWaypointMotion(
			[
				franky.CartesianWaypoint(ee_pose),
				franky.CartesianWaypoint(franky.CartesianState(displaced_position, velocity=franky.Twist([-0.0, 0.0, 0.0]))),
				franky.CartesianWaypoint(ee_pose)
			],
			# sand optimised values: 0.2, 0.12, 0.05
			relative_dynamics_factor=franky.RelativeDynamicsFactor(self.shake_dynamics_factor[0], self.shake_dynamics_factor[1], self.shake_dynamics_factor[2] )
		)

		# if dynamics of movement are 1 ,1 ,1 maintains main robot ones
		# dynamics is relative to the rest of the system
		success = False
		dynamics_dicount = 0.02
		while not success:
			try:
				self.robot.move(shake_motion)
				success=True
			except:
				
				print("Failed to do shake motion")
				if dynamics_dicount < min(self.shake_dynamics_factor):	
						self.shake_dynamics_factor -= dynamics_dicount
						print(self.shake_dynamics_factor)
				else:
					raise ShakeException('Failed to do forwards movement')
				if self.robot.recover_from_errors():
					return_motion = franky.CartesianWaypointMotion(
						[
							franky.CartesianWaypoint(ee_pose),
						],
						# sand optimised values: 0.2, 0.12, 0.05
						relative_dynamics_factor=franky.RelativeDynamicsFactor(0.2, 0.12, 0.05)
					)	
					try: 
						self.robot.move(return_motion)
					except:
						raise ShakeException('Failed to recover')


class PandaPyRobot(Robot):
	def __init__(self, robot_hostname, gripper_port):
		
		super().__init__(robot_hostname, gripper_port)
		self.robot = panda_py.Panda(robot_hostname)
		
		self.panda_model = Panda()
		
		self.initial_pos = np.array([0.01373907, 0.02742517, -0.01605499, -2.39315202, -0.00844878, 2.22456631, 0.77540909])
		self.initial_tcp = self.panda_model.fkine(self.initial_pos)
		self.current_tcp = self.panda_model.fkine(self.initial_pos)
		
		self.robot.move_to_joint_position(self.initial_pos)
		
		self.current_pose = self.initial_pos
		# open swift to visualise	
		self.viz = Swift()
		self.viz.launch(browser='firefox')
		self.viz.add(self.panda_model, readonly=True)
		self.tcp_viz=sg.Axes(0.1)
		self.viz.add(self.tcp_viz)
		


	def move_to_joint_position(self, joint_pos):
		self.robot.move_to_joint_position(joint_pos)

	
	def move_to_home(self):
		self.robot.move_to_start()

	def get_pitch(self):
		"""
			Return current pitch of the tcp in rads
		"""
		pose = self.panda_model.fkine(self.robot.q)
		
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
		if displacement >= -0.0009:
			return
		
		target_tcp = self.current_tcp * SE3(displacement, 0,0)		
		
		trajc = rtb.ctraj(self.current_tcp, target_tcp, 2)
		trajc_return = rtb.ctraj(target_tcp, self.current_tcp, 2)
		while True:
			try:
				traj = self.panda_model.ikine_LM(trajc, q0=np.tile(self.current_pose,(300,1)), tol=tolerance, ilimit=50, slimit=300) 
				traj_return = self.panda_model.ikine_LM(trajc_return, q0=np.tile(traj.q[-1], (300,1)), tol=tolerance, ilimit=50, slimit=300) 
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
			
		
		traj_list = [q.reshape(7,1) for q in traj.q]
		traj_return_list = [q.reshape(7,1) for q in traj_return.q]
		
		try:
			self.robot.move_to_joint_position(traj_list, speed_factor=0.6)
		except:
			raise ShakeException('Failed to do forwards movement')
		try:
			self.robot.move_to_joint_position(traj_return_list, speed_factor=0.6)
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
		print(incline_action)
		action = incline_action
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
			target_tcp = self.current_tcp* SE3.Ry(action) *SE3(0, 0, -0.0023)

		tolerance=1e-7
		while True:
			try: 
				traj = self.panda_model.ikine_LM(target_tcp, q0=np.tile(self.current_pose,(300,1)), tol=tolerance, ilimit=50, slimit=300) 
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
		self.current_tcp = self.panda_model.fkine(self.initial_pos)
		self.robot.move_to_joint_position(self.initial_pos)
		self.current_pose = self.initial_pos