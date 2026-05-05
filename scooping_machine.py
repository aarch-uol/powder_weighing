#Created by: Thomas Little
#Email: sgtlittl@liverpool.ac.uk
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import deque
import json
import math
from franky import *
import argparse
import sys
from scipy.spatial.transform import Rotation as R
import os
from datetime import datetime
from pyrobotiqgripper import RobotiqGripper	
from vision_powder_detection import detect_powder


#


class ScoopingMachine():

    def __init__(self, scooping_filename, positions_filename, verbose = False, robot=None, config=None):
        # Load config if not provided
        if config is None:
            config = load_config()
            
        #  --- Configuration Parameters ---
        # Cartesian impedance parameters
        self.STIFFNESS = config["cartesian_impedance"]["stiffness"]

        # Final position and parameters
        self.FINAL_POSITION = config["final_position"]["position"]
        self.FINAL_SPEED = config["final_position"]["speed"]

        # Home position
        self.HOME_POSITION = config["home_position"]["position"]
        self.HOME_CARTESIAN = config["home_position"]["cartesian"]

        # Force visualization parameters
        self.PLOT_WINDOW_SECONDS = config["force_visualization"]["plot_window_seconds"]
        self.UPDATE_FREQUENCY_HZ = config["force_visualization"]["update_frequency_hz"]

        # Depth parameters
        self.MIN_DEPTH = config["depth_parameters"]["min_depth"]
        self.MAX_DEPTH = config["depth_parameters"]["max_depth"]
        self.DEPTH_STEP = config["depth_parameters"]["depth_step"]

        # Pitch parameters
        self.MIN_PITCH = config["pitch_parameters"]["min_pitch"]
        self.MAX_PITCH = config["pitch_parameters"]["max_pitch"]
        self.PITCH_STEP = config["pitch_parameters"]["pitch_step"]

        # Coverage parameters
        self.OPTIMAL_COVERAGE = config["coverage_parameters"]["optimal_coverage"]
        self.MINIMUM_COVERAGE = config["coverage_parameters"]["minimum_coverage"]

        self.gripper = RobotiqGripper()
        # initialise robot
        if robot is not None:
            self.robot = robot
        else:
            self.robot=Robot('10.0.0.1')
        self.verbose = verbose
        self.speed = 0.01

        # Ask for container and spoon names
        container_name = input("Enter the name of the container: ").strip()
        spoon_name = input("Enter the name of the spoon: ").strip()

        if not container_name or not spoon_name:
            print("Error: Container and spoon names cannot be empty")
            sys.exit(1)

        try:
            # Load scooping movements
            with open(scooping_filename) as f:
                self.scooping_data = json.load(f)
                
            # Load positions file
            with open(positions_filename) as f:
                self.positions_data = json.load(f)
        except Exception as e:
            print(f"ERROR: Failed to load JSON files\n{e}")
            sys.exit(1)

        # Get specific moves from positions file
        moves = self.positions_data["moves"]
        self.container_move = self._get_move_by_name(moves, container_name)
        self.spoon_move = self._get_move_by_name(moves, spoon_name)
        self.endpoint_move = self._get_move_by_name(moves, "endpoint")
        
        # Validate moves
        if not self.container_move:
            print(f"Error: Container '{container_name}' not found in positions file")
            sys.exit(1)
        if not self.spoon_move:
            print(f"Error: Spoon '{spoon_name}' not found in positions file")
            sys.exit(1)
        if not self.endpoint_move:
            print("Error: 'endpoint' position not found in positions file")
            sys.exit(1)

        # Create category parameters for safe moves
        self.category_params = {
            "SafeMove": {
                "speed": 0.1,
                "stiffness": [600, 600, 600, 600, 250, 150, 50],
                "damping": [50, 50, 50, 20, 20, 20, 10]
            }
        }        

        # Activate gripper
        self.gripper.activate()
        self.gripper.calibrate(0, 80)
        self.gripper.open() 

        

    def _generate_parabolic_trajectory(self, num_waypoints: int, start_pose: np.ndarray,
                                    parabola_length: float, parabola_depth: float,
                                    max_velocity: float):
        """Generate a parabolic trajectory for scooping motion."""
        if num_waypoints < 2:
            raise ValueError("Number of waypoints must be at least 2.")
        if parabola_length <= 0 or parabola_depth <= 0:
            raise ValueError("Parabola length and depth must be positive.")
    
        trajectory = []
        l = parabola_length
        d = parabola_depth
    
        for i in range(num_waypoints):
            progress = i / (num_waypoints - 1)
            x_local = progress * l
            z_local = (4 * d / (l**2)) * x_local * (x_local - l)
            current_coord = start_pose + np.array([x_local, 0.0, z_local])
    
            velocity_magnitude = max_velocity * math.sin(math.pi * progress)
            slope = (4 * d / (l**2)) * (2 * x_local - l)
            tangent_vector = np.array([1.0, 0.0, slope])
            direction_vector = tangent_vector / np.linalg.norm(tangent_vector)
            velocity_vector = direction_vector * velocity_magnitude
            
            trajectory.append((current_coord, velocity_vector))
    
        return trajectory

    def _adjust_spoon_pitch(self, angle_degrees, speed=0.1):
        """Adjust the spoon pitch angle to the specified degrees at the given speed (0-1)."""
        if self.verbose:
            print(f"\nAdjusting spoon pitch to {angle_degrees}° at speed {speed}")
        
        try:
            current_pose = self.robot.current_pose
            end_effector_pose = current_pose.end_effector_pose
            position = end_effector_pose.translation
            current_quat = end_effector_pose.quaternion
            
            rot = R.from_quat(current_quat).as_matrix()
            euler = R.from_matrix(rot).as_euler('xyz', degrees=True)
            euler[1] = angle_degrees
            new_rot = R.from_euler('xyz', euler, degrees=True).as_matrix()
            new_quat = R.from_matrix(new_rot).as_quat()
            
            
            adjustment = 0.00013*(angle_degrees-20)
            adjusted_position=np.array([position[0]- adjustment, position[1], position[2]])
            
            motion = CartesianMotion(RobotPose(Affine(adjusted_position, new_quat)), relative_dynamics_factor= RelativeDynamicsFactor(
 			   velocity=speed, acceleration=speed, jerk=speed
		    ))
            self.robot.move(motion)
            
            
            achieved_quat = self.robot.current_pose.end_effector_pose.quaternion
            achieved_pitch = R.from_quat(achieved_quat).as_euler('xyz', degrees=True)[1]
            if self.verbose:
                print(f"Final pitch: {achieved_pitch:.1f}°")
            
            return True
            
        except Exception as e:
            print(f"Pitch adjustment failed: {str(e)}")
            return False

    def _execute_scooping_with_force_feedback(self, depth=0.01, length=0.04, speed=0.3):
        """Execute scooping while showing force feedback in real-time."""
        if self.verbose:
            print("\n--- EXECUTING SMOOTH SCOOPING MOTION WITH FORCE FEEDBACK ---")
        current_pose = self.robot.current_pose
        end_effector_pose = current_pose.end_effector_pose
        position = end_effector_pose.translation
        orientation = end_effector_pose.quaternion

        #Generate trajectory
        trajectory = self._generate_parabolic_trajectory(
            num_waypoints=100,
            start_pose=position,
            parabola_length=length,
            parabola_depth=depth,
            max_velocity=speed
        )
        
        try:
            #Initialize force visualizer with output filename
            
            #Set Cartesian impedance for scooping
            self.robot.set_cartesian_impedance(self.STIFFNESS)
            
            waypoints = []
            
            #Create waypoints with velocity profiles
            for i, (coord, velocity) in enumerate(trajectory):
                if 0 < i < len(trajectory)-1:
                    state = CartesianState(
                        pose=Affine(coord, orientation),
                        velocity=Twist(
                            linear_velocity=np.array(velocity, dtype=np.float64),
                            angular_velocity=np.zeros(3, dtype=np.float64)
                        )
                    )
                    waypoints.append(CartesianWaypoint(state))
                else:
                    waypoints.append(CartesianWaypoint(Affine(coord, orientation)))
            
            #Execute motion 
            motion = CartesianWaypointMotion(waypoints, relative_dynamics_factor=RelativeDynamicsFactor(
 			   velocity=speed, acceleration=speed, jerk=speed
		    ))
            self.robot.move(motion)
            
            
            #Reset to joint impedance mode after Cartesian motion
            self.robot.set_joint_impedance([600.0, 600.0, 600.0, 600.0, 250.0, 150.0, 50.0])
            
            
            if self.verbose:
                print("\nScooping motion complete.")
            
            return True
            
        except Exception as e:
            print(f"Execution error: {str(e)}")
            return False

    def _execute_category(self, moves, category_name, category_params):
        """Execute smooth joint-space moves with velocity-continuous waypoints."""
        if category_name not in category_params:
            print(f"Warning: No parameters found for {category_name}, using defaults")
            speed = 0.1
            stiffness = [600., 600., 600., 600., 250., 150., 50.]
            damping = [50., 50., 50., 20., 20., 20., 10.]
        else:
            speed = category_params[category_name]["speed"]
            stiffness = category_params[category_name]["stiffness"]
            damping = category_params[category_name]["damping"]

        if self.verbose:
            print(f"\n--- EXECUTING {category_name.upper()} ({len(moves)} WAYPOINTS) ---")
            print(f"Speed: {speed}\nStiffness: {stiffness}\nDamping: {damping}")

        try:
            #Ensure we're in joint impedance mode
            self.robot.set_joint_impedance(stiffness)
            
            
            #Convert moves to JointWaypoints with velocities
            waypoints = []
            
            #First waypoint (current position with zero velocity)
            current_joints = self.robot.current_joint_positions
            waypoints.append(JointWaypoint(current_joints))
            
            #Add intermediate waypoints with velocities
            for i in range(len(moves)):
                target_joints = moves[i]["position"]
                
                #Calculate velocity profile (bell curve)
                t = (i+1)/len(moves)  #Normalized position in sequence
                velocity_scale = speed * 4 * t * (1 - t)  #Parabolic profile
                
                if i < len(moves)-1:
                    #Calculate direction to next waypoint
                    next_joints = moves[i+1]["position"]
                    direction = np.array(next_joints) - np.array(target_joints)
                    direction_norm = np.linalg.norm(direction)
                    if direction_norm > 1e-6:
                        direction = direction / direction_norm
                    
                    #Create JointState with velocity
                    joint_velocity = direction * velocity_scale
                    state = JointState(
                        position=target_joints,
                        velocity=joint_velocity.tolist()
                    )
                    waypoints.append(JointWaypoint(state))
                else:
                    #Final waypoint with zero velocity
                    waypoints.append(JointWaypoint(target_joints))
            
            #Execute motion
            motion = JointWaypointMotion(waypoints, relative_dynamics_factor=RelativeDynamicsFactor(
 			   velocity=speed, acceleration=speed, jerk=speed
		    ))
            self.robot.move(motion)
            
            
            return True
            
        except Exception as e:
            print(f"Error in {category_name}: {str(e)}")
            return False

        
    def _move_to_final_position(self):
        """Move the robot to the predefined final position."""
        if self.verbose:
            print("\n--- MOVING TO FINAL POSITION ---")
        try:
            motion = JointMotion(self.FINAL_POSITION, relative_dynamics_factor=RelativeDynamicsFactor(
 			   velocity=self.FINAL_SPEED, acceleration=self.FINAL_SPEED, jerk=self.FINAL_SPEED
		    ))
            self.robot.move(motion)
            return True
        except Exception as e:
            print(f"Error moving to final position: {str(e)}")
            return False
        
    def _move_to_home_position(self):  
        self.robot.move(JointMotion(self.HOME_POSITION, relative_dynamics_factor=RelativeDynamicsFactor(
 			   velocity=0.1, acceleration=0.1, jerk=0.1
        )))
        

    def _get_move_by_name(self, moves, name):
        """Find a move by name in the moves list"""
        for move in moves:
            if move["name"] == name:
                return move
        return None

    def _execute_safe_pick_place(self, move_dict, action, category_params=None, safe_height=0.1):
        """Execute a pick/place action with safe approach and retreat using relative Cartesian motions"""
        # Flip gripper action if it's a return operation
        gripper_action = move_dict["gripper"]
        if action == "return":
            gripper_action = "open" if gripper_action == "close" else "close"
        
        # Get default parameters if not provided
        if category_params is None:
            category_params = {
                "SafeMove": {
                    "speed": 0.01,
                    "stiffness": [600, 600, 600, 600, 250, 150, 50],
                    "damping": [50, 50, 50, 20, 20, 20, 10]
                }
            }
        
        # Get parameters for safe moves
        safe_params = category_params.get("SafeMove", category_params.get("Pick-and-Place", {}))
        stiffness = safe_params.get("stiffness", [600, 600, 600, 600, 250, 150, 50])
        damping = safe_params.get("damping", [50, 50, 50, 20, 20, 20, 10])
        
        if self.verbose:
            print(f"\n--- EXECUTING SAFE {action.upper()} ---")
            print(f"Stiffness: {stiffness}\nDamping: {damping}")

        try:
          
            # --- SAFE APPROACH PHASE ---
            if self.verbose:
                print("!SAFE APPROACH!")

            # Extract target position and orientation
            q = np.array(move_dict["position"])
            f_t_ee = self.robot.state.F_T_EE
            ee_t_k = self.robot.state.EE_T_K
            current_pose = self.robot.model.pose(Frame.EndEffector, q, f_t_ee, ee_t_k)
            
            # Check if this is a spoon move (we'll use the name to determine)
            is_spoon = "spoon" in move_dict["name"].lower()
            
            if is_spoon:
                # For spoon moves, first move 0.15m behind in x-axis and up in z-axis
                xz_translation = Affine(translation=np.array([-0.15, 0, 0.1]), quaternion=np.array([0, 0, 0, 1]))
                new_pose = xz_translation * current_pose
                self.robot.move(CartesianMotion(new_pose, relative_dynamics_factor=RelativeDynamicsFactor(
 			        velocity=0.05, acceleration=0.05, jerk=0.05
		        )))
                
                # Then move forward to the target x position (still at safe height)
                x_translation = Affine(translation=np.array([0.15, 0, 0]), quaternion=np.array([0, 0, 0, 1]))
                new_pose = x_translation * new_pose
                self.robot.move(CartesianMotion(new_pose, relative_dynamics_factor=RelativeDynamicsFactor(
 			        velocity=0.05, acceleration=0.05, jerk=0.05
		        )))
            else:
                # Define the movement steps
                movement_steps = [0.1, 0.05, 0.025, 0.01]
                current_step_index = 0

                while current_step_index < len(movement_steps):
                    # Get the current movement step
                    step = movement_steps[current_step_index]
                    
                    # Apply the Z translation
                    z_translation = Affine(
                        translation=np.array([0, 0, step]), 
                        quaternion=np.array([0, 0, 0, 1])
                    )
                    new_pose = z_translation * current_pose  # Assuming current_pose is defined
                    
                    # Move the robot
                    self.robot.move(CartesianMotion(new_pose, relative_dynamics_factor=RelativeDynamicsFactor(
 			        velocity=0.05, acceleration=0.05, jerk=0.05
		        )))  # Assuming robot is your robot controller
                    
                    # Increment to the next step
                    current_step_index += 1
                
            # --- FINAL APPROACH PHASE ---
            if self.verbose:
                print("!FINAL APPROACH!")   
            # Move down to target position (Cartesian for straight line)
            final_motion = JointMotion(move_dict["position"], relative_dynamics_factor=RelativeDynamicsFactor(
 			    velocity=0.05, acceleration=0.05, jerk=0.05
		    ))
            self.robot.move(final_motion)
            
            # --- GRIPPER ACTION ---
            if self.verbose:
                print("!GRIPPER ACTION!")
            if gripper_action == "open":
                self.gripper.open()
            else:
                self.gripper.close()
            time.sleep(1)  # Ensure gripper completes action
            
            # --- SAFE RETREAT PHASE ---
            if self.verbose:
                print("!SAFE RETREAT!")
            # Move back up using relative Cartesian motion
            if is_spoon:
                # For spoon, first move straight up
                z_translation = Affine(translation=np.array([0, 0, 0.1]), quaternion=np.array([0, 0, 0, 1]))
                new_pose = z_translation * self.robot.current_pose.end_effector_pose
                self.robot.move(CartesianMotion(new_pose, relative_dynamics_factor=RelativeDynamicsFactor(
 			        velocity=0.05, acceleration=0.05, jerk=0.05
		        )))
                
                # Then move back 0.25m in x-axis
                x_translation = Affine(translation=np.array([-0.25, 0, 0]), quaternion=np.array([0, 0, 0, 1]))
                new_pose = x_translation * new_pose
                self.robot.move(CartesianMotion(new_pose, relative_dynamics_factor=RelativeDynamicsFactor(
 			        velocity=0.05, acceleration=0.05, jerk=0.05
		        )))
            else:
                # For non-spoon, just move straight up
                z_translation = Affine(translation=np.array([0, 0, 0.1]), quaternion=np.array([0, 0, 0, 1]))
                new_pose = z_translation * self.robot.current_pose.end_effector_pose
                self.robot.move(CartesianMotion(new_pose, relative_dynamics_factor=RelativeDynamicsFactor(
 			        velocity=0.05, acceleration=0.05, jerk=0.05
		        )))
            
            # Restore original dynamics only at the very end
            return True
            
        except Exception as e:
            print(f"Error in safe pick/place ({action}): {str(e)}")
            return False

    def load_powder(self):
        # 1. Pick up container and move to endpoint
        if self.verbose:
            print("\n=== PICKING UP CONTAINER ===")
        self._move_to_home_position()
        if not self._execute_safe_pick_place(self.container_move, "pick", self.category_params):
            print("Failed to pick container")
            sys.exit(1)
        self._move_to_home_position()
        
        if self.verbose:
            print("\n=== PLACING CONTAINER AT ENDPOINT ===")
        if not self._execute_safe_pick_place(self.endpoint_move, "place", self.category_params):
            print("Failed to place container at endpoint")
            sys.exit(1)
        self._move_to_home_position()

    def pickup_spoon(self):
        # 2. Pick up spoon
        if self.verbose:
            print("\n=== PICKING UP SPOON ===")
        if not self._execute_safe_pick_place(self.spoon_move, "pick", self.category_params):
            print("Failed to pick spoon")
            sys.exit(1)
        self._move_to_home_position()

    def unload_powder(self):
        # 2. Return container to original position
        if self.verbose:
            print("\n=== RETURNING CONTAINER TO ORIGINAL POSITION ===")
        if not self._execute_safe_pick_place(self.endpoint_move, "return", self.category_params):
            print("Failed to pick container from endpoint")
            sys.exit(1)
        self._move_to_home_position()
        if not self._execute_safe_pick_place(self.container_move, "return", self.category_params):
            print("Failed to return container to original position")
            sys.exit(1)
        self._move_to_home_position()

    def drop_spoon(self):
        # 1. Return spoon
        if self.verbose:
            print("\n=== RETURNING SPOON ===")
        if not self._execute_safe_pick_place(self.spoon_move, "return", self.category_params):
            print("Failed to return spoon")
            sys.exit(1)
        self._move_to_home_position()


    def scoop(self, starting_angle=40,length=0.025, vision_check=True):
        final_modal_outcome = False #Vision scooping detection result
        angle = starting_angle
        depth = 0.015
        length = length
        while (not final_modal_outcome):

            print(f"\n=== EXECUTING SCOOPING SEQUENCE : DEPTH={depth} LENGTH={length} PITCH={angle} ===")
            pre_moves = [m for m in self.scooping_data["moves"] if m["category"] == "Pre-scooping"]
            if not self._execute_category( pre_moves, "Pre-scooping", self.scooping_data["categories"]):
                sys.exit(1)

            if not self._adjust_spoon_pitch(angle):
                sys.exit(1)

            if not self._execute_scooping_with_force_feedback(depth, length, self.speed):
                sys.exit(1)
            
            post_moves = [m for m in self.scooping_data["moves"] if m["category"] == "Post-scooping"]
            if not self._adjust_spoon_pitch(15, 0.02):
                sys.exit(1)

            current_pose = self.robot.current_cartesian_state
            # Move to camera
            right_translation = Affine(translation=np.array([-0.035, -0.065, 0.04]), quaternion=np.array([0, 0, 0, 1]))
            new_pose = right_translation * current_pose
            self.robot.move(CartesianMotion(new_pose, relative_dynamics_factor=RelativeDynamicsFactor(
 			   velocity=self.speed, acceleration=self.speed, jerk=self.speed
		    )))
            coverage=0
            if vision_check:
                final_modal_outcome, coverage = detect_powder()
            print(coverage)
            if not vision_check:
                final_modal_outcome=True
            
            
            if self.verbose:
                print(f"\n Empty? : {final_modal_outcome}; estimated coverage: {coverage}; OPTIMAL_COVERAGE : {self.OPTIMAL_COVERAGE}")
            if vision_check:
                if coverage<self.MINIMUM_COVERAGE:
                    print(f"WARNING: detected empty scoop. Trying again")
                    if angle-self.PITCH_STEP >= self.MIN_PITCH:
                        angle -=self.PITCH_STEP
                    elif(depth+self.DEPTH_STEP)<= self.MAX_DEPTH:
                        depth+=self.DEPTH_STEP
                    else:
                        return False, angle
                elif coverage > self.OPTIMAL_COVERAGE:
                    final_modal_outcome=False
                    print(f"WARNING: scoop contains too much powder. Trying again")
                    if angle+self.PITCH_STEP <= self.MAX_PITCH:
                        angle +=self.PITCH_STEP
                    elif(depth-self.DEPTH_STEP)>= self.MIN_DEPTH:
                        depth-=self.DEPTH_STEP
                    else:
                        return False, angle
                

            print(final_modal_outcome)         

        if not self._execute_category( post_moves, "Post-scooping", self.scooping_data["categories"]):
            sys.exit(1)

        if not self._move_to_final_position():
            sys.exit(1)
        time.sleep(10)
        
        self.best_angle=angle
        return True, angle
    
    def reset_scoop_pose(self):
        
        backwards_from_final = CartesianMotion(
            Affine([-0.2, 0.0, 0.0]),
            ReferenceType.Relative,
            relative_dynamics_factor=RelativeDynamicsFactor(
 			   velocity=self.speed, acceleration=self.speed, jerk=self.speed
            ))
        try:
            self.robot.move(backwards_from_final)
        except: 
            if self.robot.recover_from_errors():
                backwards_from_final = CartesianMotion(
                    Affine([-0.05, 0.0, 0.0]),
                    ReferenceType.Relative,
                    relative_dynamics_factor=RelativeDynamicsFactor(
                        velocity=0.05, acceleration=0.05, jerk=0.05
            ))
        self._move_to_home_position()
    



def main():
    parser = argparse.ArgumentParser(description="Run scooping sequence with force feedback.")
    parser.add_argument("scooping_filename", help="JSON file with Pre/Post-scooping moves.")
    parser.add_argument("positions_filename", help="JSON file with container and spoon positions.")
    args = parser.parse_args()

    scooper = ScoopingMachine(args.scooping_filename, args.positions_filename, verbose=True)

    scooper.load_powder()
    scooper.pickup_spoon()
    for  length in np.arange(0.025, 0.03, 0.002):
        print(length)
        for i in range (5):
            scooper.scoop(starting_angle=40,length=length, vision_check=False)
            input('Measure scooped quantity and press ENTER to continue')
    scooper.reset_scoop_pose()
    scooper.drop_spoon()
    scooper.unload_powder()
    

    

    print("\n=== EXPERIMENT COMPLETE ===")

if __name__ == "__main__":
    main()
