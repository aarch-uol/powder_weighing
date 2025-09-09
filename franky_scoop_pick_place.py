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


# --- Configuration ---
#Cartesian impedance parameters
STIFFNESS = [600.0, 600.0, 600.0, 40.0, 40.0, 40.0]  #Translational, rotational

#Final position and parameters
FINAL_POSITION = [0.01373907, 0.02742517, -0.01605499, -2.39315202, -0.00844878, 2.22456631, 0.77540909]
FINAL_SPEED = 0.05

HOME_POSITION = [-0.00737469, -0.88195892, -0.01844868, -2.25721947, -0.02814894, 0.92399915, 0.75734471]
HOME_CARTESIAN = "(t=[0.456037 -0.0117829 0.495123], q=[0.974623 0.000202117 -0.223852 0.000627159])"

#Force visualization parameters
PLOT_WINDOW_SECONDS = 60
UPDATE_FREQUENCY_HZ = 100  #Reduced frequency to avoid overloading

gripper = RobotiqGripper()

class RealTimeForceVisualizer:
    """Modified version that can be used during motion execution"""
    def __init__(self, robot, update_frequency, plot_window_s, output_filename=None):
        self.robot = robot
        self.update_interval_ms = 1000 / update_frequency
        self.max_data_points = int(plot_window_s * update_frequency)
        self.output_filename = output_filename

        #Data storage
        self.timestamps = deque(maxlen=self.max_data_points)
        self.force_x = deque(maxlen=self.max_data_points)
        self.force_y = deque(maxlen=self.max_data_points)
        self.force_z = deque(maxlen=self.max_data_points)
        
        self.start_time = time.time()
        self.fig = None
        self.ax = None
        self.line_x = None
        self.line_y = None
        self.line_z = None
        self.ani = None
        self.is_running = False
        self.recording = True  #Flag to control whether we're recording new data

    def _initialize_plot(self):
        """Sets up the matplotlib figure and axes for plotting."""
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        self.line_x, = self.ax.plot([], [], lw=2, label='Force X')
        self.line_y, = self.ax.plot([], [], lw=2, label='Force Y')
        self.line_z, = self.ax.plot([], [], lw=2, label='Force Z')

        self.ax.set_title("Real-time External End-Effector Forces During Scooping")
        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("Force (N)")
        self.ax.legend(loc='upper right')
        self.ax.grid(True)
        
        #Set initial plot limits
        self.ax.set_xlim(0, PLOT_WINDOW_SECONDS)
        self.ax.set_ylim(-10, 10)

    def _update_plot(self, frame):
        """
        This function is called by FuncAnimation at each update interval.
        It reads the latest robot state and updates the plot data.
        """
        try:
            if self.recording:  #Only record new data if we're in recording mode
                #Get the estimated external wrench (forces/torques) in the base frame
                wrench = self.robot.state.O_F_ext_hat_K
                
                current_time = time.time() - self.start_time
                
                #Append new data to our deques
                self.timestamps.append(current_time)
                self.force_x.append(wrench[0])
                self.force_y.append(wrench[1])
                self.force_z.append(wrench[2])
            
            #Update the plot lines with the current data (even if not recording)
            self.line_x.set_data(self.timestamps, self.force_x)
            self.line_y.set_data(self.timestamps, self.force_y)
            self.line_z.set_data(self.timestamps, self.force_z)
            
            #Dynamically adjust plot limits for better visualization
            if len(self.timestamps) > 1:
                self.ax.set_xlim(self.timestamps[0], self.timestamps[-1] + 1)  #Add 1s buffer
                min_force = min(min(self.force_x), min(self.force_y), min(self.force_z))
                max_force = max(max(self.force_x), max(self.force_y), max(self.force_z))
                force_range = max_force - min_force
                self.ax.set_ylim(min_force - 0.2*force_range, max_force + 0.2*force_range)

        except Exception as e:
            print(f"Error updating plot: {e}")
            
        return self.line_x, self.line_y, self.line_z,

    def start(self):
        """Starts the visualization in a non-blocking way"""
        self._initialize_plot()
        self.ani = FuncAnimation(self.fig, self._update_plot, 
                                interval=self.update_interval_ms,
                                blit=True, cache_frame_data=False)
        self.is_running = True
        plt.show(block=False)

    def stop_recording(self):
        """Stops recording new data points but keeps the plot open"""
        self.recording = False
        #Save the plot immediately when recording stops
        self.save_data()

    def stop(self):
        """Clean up the visualization and save data"""
        if self.ani:
            self.ani.event_source.stop()
        self.is_running = False
        self.save_data()

    def save_data(self):
        """Save the collected force data to a file"""
        try:
            #Create a directory for saving data if it doesn't exist
            os.makedirs("force_data", exist_ok=True)
            
            #Generate filename
            if self.output_filename:
                #Use user-provided filename if specified
                filename = f"force_data/{self.output_filename}.csv"
                plot_filename = f"force_data/{self.output_filename}.png"
            else:
                #Otherwise use timestamp
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"force_data/force_measurement_{timestamp}.csv"
                plot_filename = f"force_data/force_plot_{timestamp}.png"
            
            #Convert deques to lists for saving
            timestamps = list(self.timestamps)
            force_x = list(self.force_x)
            force_y = list(self.force_y)
            force_z = list(self.force_z)
            
            #Write data to CSV file
            with open(filename, 'w') as f:
                f.write("timestamp,force_x,force_y,force_z\n")
                for t, fx, fy, fz in zip(timestamps, force_x, force_y, force_z):
                    f.write(f"{t:.4f},{fx:.4f},{fy:.4f},{fz:.4f}\n")
            
            print(f"\nForce data saved to {filename}")
            
            #Also save the plot as an image
            self.fig.savefig(plot_filename, dpi=300, bbox_inches='tight')
            print(f"Force plot saved to {plot_filename}")
            
        except Exception as e:
            print(f"Error saving force data: {e}")

    def get_data(self):
        """Return the collected data as numpy arrays"""
        return {
            'timestamps': np.array(self.timestamps),
            'force_x': np.array(self.force_x),
            'force_y': np.array(self.force_y),
            'force_z': np.array(self.force_z)
        }

def generate_parabolic_trajectory(num_waypoints: int, start_pose: np.ndarray,
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

def adjust_spoon_pitch(robot, angle_degrees, speed=0.1):
    """Adjust the spoon pitch angle to the specified degrees at the given speed (0-1)."""
    print(f"\nAdjusting spoon pitch to {angle_degrees}° at speed {speed}")
    
    try:
        current_pose = robot.current_pose
        end_effector_pose = current_pose.end_effector_pose
        position = end_effector_pose.translation
        current_quat = end_effector_pose.quaternion
        print(position, type(position))
        
        rot = R.from_quat(current_quat).as_matrix()
        euler = R.from_matrix(rot).as_euler('xyz', degrees=True)
        euler[1] = angle_degrees
        new_rot = R.from_euler('xyz', euler, degrees=True).as_matrix()
        new_quat = R.from_matrix(new_rot).as_quat()
        
        original_dynamics = robot.relative_dynamics_factor
        robot.relative_dynamics_factor = speed
        adjustment = 0.00013*(angle_degrees-20)
        adjusted_position=np.array([position[0]- adjustment, position[1], position[2]])
        print(adjusted_position)
        motion = CartesianMotion(RobotPose(Affine(adjusted_position, new_quat)))
        robot.move(motion)
        robot.relative_dynamics_factor = original_dynamics
        
        achieved_quat = robot.current_pose.end_effector_pose.quaternion
        achieved_pitch = R.from_quat(achieved_quat).as_euler('xyz', degrees=True)[1]
        print(f"Final pitch: {achieved_pitch:.1f}°")
        
        return True
        
    except Exception as e:
        print(f"Pitch adjustment failed: {str(e)}")
        return False

def execute_scooping_with_force_feedback(robot, depth=0.01, length=0.04, speed=0.3, output_filename=None):
    """Execute scooping while showing force feedback in real-time."""
    print("\n--- EXECUTING SMOOTH SCOOPING MOTION WITH FORCE FEEDBACK ---")
    current_pose = robot.current_pose
    end_effector_pose = current_pose.end_effector_pose
    position = end_effector_pose.translation
    orientation = end_effector_pose.quaternion

    #Generate trajectory
    trajectory = generate_parabolic_trajectory(
        num_waypoints=100,
        start_pose=position,
        parabola_length=length,
        parabola_depth=depth,
        max_velocity=speed
    )
    
    try:
        #Initialize force visualizer with output filename
        # visualizer = RealTimeForceVisualizer(robot, UPDATE_FREQUENCY_HZ, PLOT_WINDOW_SECONDS, output_filename)
        # visualizer.start()
        
        #Set Cartesian impedance for scooping
        robot.set_cartesian_impedance(STIFFNESS)
        original_dynamics = robot.relative_dynamics_factor
        robot.relative_dynamics_factor = speed

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
        
        #Execute motion asynchronously
        motion = CartesianWaypointMotion(waypoints)
        robot.move(motion, asynchronous=True)

        #Keep updating the plot while the robot is moving
        while not robot.poll_motion():
            plt.pause(0.001)  #Allow the plot to update
        
        #Wait for motion to fully complete
        time.sleep(0.1)  #Small delay to ensure motion is really done
        
        #Add 1-second pause to see the base forces
        print("\nScooping complete. Recording final forces for 1 second...")
        time.sleep(1.0)
        
        #Stop recording new data points but keep the plot open
        #visualizer.stop_recording()
        
        #Reset to joint impedance mode after Cartesian motion
        robot.set_joint_impedance([600.0, 600.0, 600.0, 600.0, 250.0, 150.0, 50.0])
        robot.relative_dynamics_factor = original_dynamics
        
        print("\nScooping motion complete.")
        
        # #Keep the plot alive for a fixed duration (e.g., 5 seconds after motion)
        # time_to_keep_open = 5  #seconds
        # start_wait = time.time()
        # while time.time() - start_wait < time_to_keep_open:
        #     if plt.fignum_exists(visualizer.fig.number):
        #         plt.pause(0.1)
        #     else:
        #         break

        #Cleanly stop and close the plot
        # visualizer.stop()
        # plt.close(visualizer.fig)
            
        return True
        
    except Exception as e:
        print(f"Execution error: {str(e)}")
        robot.relative_dynamics_factor = original_dynamics
        # if 'visualizer' in locals():
        #     visualizer.stop()
        return False

def execute_category(robot, moves, category_name, category_params):
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

    print(f"\n--- EXECUTING {category_name.upper()} ({len(moves)} WAYPOINTS) ---")
    print(f"Speed: {speed}\nStiffness: {stiffness}\nDamping: {damping}")

    try:
        #Ensure we're in joint impedance mode
        robot.set_joint_impedance(stiffness)
        
        #Save and set dynamics
        original_dynamics = robot.relative_dynamics_factor
        robot.relative_dynamics_factor = speed
        
        #Convert moves to JointWaypoints with velocities
        waypoints = []
        
        #First waypoint (current position with zero velocity)
        current_joints = robot.current_joint_positions
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
        motion = JointWaypointMotion(waypoints)
        robot.move(motion)
        
        robot.relative_dynamics_factor = original_dynamics
        return True
        
    except Exception as e:
        print(f"Error in {category_name}: {str(e)}")
        robot.relative_dynamics_factor = original_dynamics
        return False

    
def move_to_final_position(robot):
    """Move the robot to the predefined final position."""
    print("\n--- MOVING TO FINAL POSITION ---")
    try:
        original_dynamics = robot.relative_dynamics_factor
        robot.relative_dynamics_factor = FINAL_SPEED
        motion = JointMotion(FINAL_POSITION)
        robot.move(motion)
        robot.relative_dynamics_factor = original_dynamics
        return True
    except Exception as e:
        print(f"Error moving to final position: {str(e)}")
        robot.relative_dynamics_factor = original_dynamics
        return False
    
def move_to_home_position(robot):  
    original_dynamics = robot.relative_dynamics_factor
    robot.relative_dynamics_factor = 0.1
    robot.move(JointMotion(HOME_POSITION))
    robot.relative_dynamics_factor = original_dynamics

def get_move_by_name(moves, name):
    """Find a move by name in the moves list"""
    for move in moves:
        if move["name"] == name:
            return move
    return None

def execute_safe_pick_place(robot, move_dict, action, category_params=None, safe_height=0.1):
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
    
    print(f"\n--- EXECUTING SAFE {action.upper()} ---")
    print(f"Stiffness: {stiffness}\nDamping: {damping}")

    try:
        # Save original dynamics
        original_dynamics = robot.relative_dynamics_factor
        # Set consistent speed for all motions
        robot.relative_dynamics_factor = 0.05

        # --- SAFE APPROACH PHASE ---
        print("!SAFE APPROACH!")

        # Extract target position and orientation
        q = np.array(move_dict["position"])
        f_t_ee = robot.state.F_T_EE
        ee_t_k = robot.state.EE_T_K
        current_pose = robot.model.pose(Frame.EndEffector, q, f_t_ee, ee_t_k)
        
        # Check if this is a spoon move (we'll use the name to determine)
        is_spoon = "spoon" in move_dict["name"].lower()
        
        if is_spoon:
            # For spoon moves, first move 0.15m behind in x-axis and up in z-axis
            xz_translation = Affine(translation=np.array([-0.15, 0, 0.1]), quaternion=np.array([0, 0, 0, 1]))
            new_pose = xz_translation * current_pose
            robot.move(CartesianMotion(new_pose))
            
            # Then move forward to the target x position (still at safe height)
            x_translation = Affine(translation=np.array([0.15, 0, 0]), quaternion=np.array([0, 0, 0, 1]))
            new_pose = x_translation * new_pose
            robot.move(CartesianMotion(new_pose))
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
                robot.move(CartesianMotion(new_pose))  # Assuming robot is your robot controller
                
                # Increment to the next step
                current_step_index += 1
            
        # --- FINAL APPROACH PHASE ---
        print("!FINAL APPROACH!")   
        # Move down to target position (Cartesian for straight line)
        final_motion = JointMotion(move_dict["position"])
        robot.move(final_motion)
        
        # --- GRIPPER ACTION ---
        print("!GRIPPER ACTION!")
        if gripper_action == "open":
            gripper.open()
        else:
            gripper.close()
        time.sleep(1)  # Ensure gripper completes action
        
        # --- SAFE RETREAT PHASE ---
        print("!SAFE RETREAT!")
        # Move back up using relative Cartesian motion
        if is_spoon:
            # For spoon, first move straight up
            z_translation = Affine(translation=np.array([0, 0, 0.1]), quaternion=np.array([0, 0, 0, 1]))
            new_pose = z_translation * robot.current_pose.end_effector_pose
            robot.move(CartesianMotion(new_pose))
            
            # Then move back 0.25m in x-axis
            x_translation = Affine(translation=np.array([-0.25, 0, 0]), quaternion=np.array([0, 0, 0, 1]))
            new_pose = x_translation * new_pose
            robot.move(CartesianMotion(new_pose))
        else:
            # For non-spoon, just move straight up
            z_translation = Affine(translation=np.array([0, 0, 0.1]), quaternion=np.array([0, 0, 0, 1]))
            new_pose = z_translation * robot.current_pose.end_effector_pose
            robot.move(CartesianMotion(new_pose))
        
        # Restore original dynamics only at the very end
        robot.relative_dynamics_factor = original_dynamics
        return True
        
    except Exception as e:
        print(f"Error in safe pick/place ({action}): {str(e)}")
        # Restore dynamics in case of error
        if 'original_dynamics' in locals():
            robot.relative_dynamics_factor = original_dynamics
        return False



def main():
    parser = argparse.ArgumentParser(description="Run scooping sequence with force feedback.")
    parser.add_argument("scooping_filename", help="JSON file with Pre/Post-scooping moves.")
    parser.add_argument("positions_filename", help="JSON file with container and spoon positions.")
    parser.add_argument("--depth", type=float, default=0.01, help="Scoop depth (meters).")
    parser.add_argument("--length", type=float, default=0.04, help="Scoop length (meters).")
    parser.add_argument("--speed", type=float, default=0.1, help="Scoop speed (0-1).")
    parser.add_argument("--angle", type=float, default=30.0, help="Spoon pitch angle (degrees).")
    parser.add_argument("--output", type=str, help="Output filename for force telemetry data (without extension)")
    args = parser.parse_args()

    # Ask for container and spoon names
    container_name = input("Enter the name of the container: ").strip()
    spoon_name = input("Enter the name of the spoon: ").strip()

    # container_name = "container1"
    # spoon_name = "spoon1"

    if not container_name or not spoon_name:
        print("Error: Container and spoon names cannot be empty")
        sys.exit(1)

    try:
        # Load scooping movements
        with open(args.scooping_filename) as f:
            scooping_data = json.load(f)
            
        # Load positions file
        with open(args.positions_filename) as f:
            positions_data = json.load(f)
    except Exception as e:
        print(f"ERROR: Failed to load JSON files\n{e}")
        sys.exit(1)

    # Get specific moves from positions file
    moves = positions_data["moves"]
    container_move = get_move_by_name(moves, container_name)
    spoon_move = get_move_by_name(moves, spoon_name)
    endpoint_move = get_move_by_name(moves, "endpoint")
    
    # Validate moves
    if not container_move:
        print(f"Error: Container '{container_name}' not found in positions file")
        sys.exit(1)
    if not spoon_move:
        print(f"Error: Spoon '{spoon_name}' not found in positions file")
        sys.exit(1)
    if not endpoint_move:
        print("Error: 'endpoint' position not found in positions file")
        sys.exit(1)

    # Create category parameters for safe moves
    category_params = {
        "SafeMove": {
            "speed": 0.1,
            "stiffness": [600, 600, 600, 600, 250, 150, 50],
            "damping": [50, 50, 50, 20, 20, 20, 10]
        }
    }

    robot = Robot("10.0.0.1")
    robot.relative_dynamics_factor = 0.01  
    final_modal_outcome = False #Vision scooping detection result

    move_to_home_position(robot)        

    # Activate gripper
    gripper.activate()
    gripper.calibrate(0, 80)
    gripper.open() 

    # 1. Pick up container and move to endpoint
    print("\n=== PICKING UP CONTAINER ===")
    move_to_home_position(robot)
    if not execute_safe_pick_place(robot, container_move, "pick", category_params):
        print("Failed to pick container")
        sys.exit(1)
    move_to_home_position(robot)
    
    print("\n=== PLACING CONTAINER AT ENDPOINT ===")
    if not execute_safe_pick_place(robot, endpoint_move, "place", category_params):
        print("Failed to place container at endpoint")
        sys.exit(1)
    move_to_home_position(robot)

    # 2. Pick up spoon
    print("\n=== PICKING UP SPOON ===")
    if not execute_safe_pick_place(robot, spoon_move, "pick", category_params):
        print("Failed to pick spoon")
        sys.exit(1)
    move_to_home_position(robot)

    # uncomment while
    while (not final_modal_outcome):
    # for i in range(15):
        # print(f"measurement  {i}")

        print("\n=== EXECUTING SCOOPING SEQUENCE ===")
        pre_moves = [m for m in scooping_data["moves"] if m["category"] == "Pre-scooping"]
        if not execute_category(robot, pre_moves, "Pre-scooping", scooping_data["categories"]):
            sys.exit(1)

        if not adjust_spoon_pitch(robot, args.angle):
            sys.exit(1)

        if not execute_scooping_with_force_feedback(robot, args.depth, args.length, args.speed, args.output):
            sys.exit(1)
        
        post_moves = [m for m in scooping_data["moves"] if m["category"] == "Post-scooping"]
        if not adjust_spoon_pitch(robot, 15, 0.02):
            sys.exit(1)

        current_pose = robot.current_cartesian_state
        # Move to camera
        right_translation = Affine(translation=np.array([-0.035, -0.065, 0.04]), quaternion=np.array([0, 0, 0, 1]))
        new_pose = right_translation * current_pose
        robot.move(CartesianMotion(new_pose))

        final_modal_outcome = detect_powder()
        #final_modal_outcome = True   

        print(final_modal_outcome)             
    

    if not execute_category(robot, post_moves, "Post-scooping", scooping_data["categories"]):
        sys.exit(1)

    if not move_to_final_position(robot):
        sys.exit(1)
    time.sleep(10)
    print("Moving backwards")
    robot.relative_dynamics_factor = 0.05
    backwards_from_final = CartesianMotion(Affine([-0.2, 0.0, 0.0]), ReferenceType.Relative)
    robot.move(backwards_from_final)

    move_to_home_position(robot)

    # -- New sequence for returning items --
    # 1. Return spoon
    print("\n=== RETURNING SPOON ===")
    if not execute_safe_pick_place(robot, spoon_move, "return", category_params):
        print("Failed to return spoon")
        sys.exit(1)
    move_to_home_position(robot)

    # 2. Return container to original position
    print("\n=== RETURNING CONTAINER TO ORIGINAL POSITION ===")
    if not execute_safe_pick_place(robot, endpoint_move, "return", category_params):
        print("Failed to pick container from endpoint")
        sys.exit(1)
    move_to_home_position(robot)
    if not execute_safe_pick_place(robot, container_move, "return", category_params):
        print("Failed to return container to original position")
        sys.exit(1)
    move_to_home_position(robot)

    print("\n=== EXPERIMENT COMPLETE ===")

if __name__ == "__main__":
    main()