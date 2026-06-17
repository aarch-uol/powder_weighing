import argparse
import json
import os
import sys
import time

import numpy as np
from scipy.spatial.transform import Rotation as R
from franky import Affine, CartesianMotion, CartesianWaypoint, CartesianWaypointMotion, RelativeDynamicsFactor, Robot

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import scooping_machine
from scooping_machine import ScoopingMachine
from vision_powder_detection_pitchcontrol import TiltReleaseRecorder, detect_powder as detect_powder_pitchcontrol


#  --- Configuration Loader ---
def load_config(config_filename="config.json"):
    """Load configuration from JSON file."""
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(script_dir, config_filename)
        if not os.path.exists(config_path) and not os.path.isabs(config_filename):
            config_path = os.path.join(os.path.dirname(script_dir), config_filename)

        with open(config_path) as f:
            config = json.load(f)
        return config
    except Exception as e:
        print(f"ERROR: Failed to load configuration file '{config_filename}'\n{e}")
        sys.exit(1)


# --- PITCH_CONTROL MODIFICATION START ---
# Replaces the post-scoop SAC incline+shake step with one deterministic
# tilt-release motion: move closer to the base, then pitch from horizontal to 85 deg.
def _tilt_release_dynamics(speed=0.03):
    return RelativeDynamicsFactor(
        velocity=speed,
        acceleration=speed,
        jerk=speed
    )


def tilt_release_after_scoop(
    robot,
    backward_offset=0,
    start_pitch=0.0,
    end_pitch=70.0,
    pitch_step=5.0,
    speed=0.02,
    settle_time=2.0,
    record_tilt_release=True,
    sample_index=None,
    z_offset=0
):
    if pitch_step <= 0:
        raise ValueError("pitch_step must be greater than zero")

    if backward_offset > 0:
        current_pose = robot.current_cartesian_state
        backward_translation = Affine(
            translation=np.array([-backward_offset, 0.0, z_offset]),
            quaternion=np.array([0.0, 0.0, 0.0, 1.0])
        )
        new_pose = backward_translation * current_pose
        backwards_motion = CartesianMotion(
            new_pose,
            relative_dynamics_factor=_tilt_release_dynamics(speed)
        )
        robot.move(backwards_motion)

    end_effector_pose = robot.current_pose.end_effector_pose
    position = end_effector_pose.translation
    euler = R.from_quat(end_effector_pose.quaternion).as_euler('xyz', degrees=True)

    pitch_waypoints = np.linspace(
        start_pitch,
        end_pitch,
        max(2, int(np.ceil(abs(end_pitch - start_pitch) / pitch_step)) + 1)
    )

    waypoints = []
    for pitch in pitch_waypoints:
        euler[1] = float(pitch)
        waypoint_quat = R.from_euler('xyz', euler, degrees=True).as_quat()
        waypoints.append(CartesianWaypoint(Affine(position, waypoint_quat)))

    with TiltReleaseRecorder(enabled=record_tilt_release, sample_index=sample_index):
        robot.move(CartesianWaypointMotion(
            waypoints,
            relative_dynamics_factor=_tilt_release_dynamics(speed)
        ))

        time.sleep(settle_time)
    achieved_pitch = R.from_quat(robot.current_pose.end_effector_pose.quaternion).as_euler('xyz', degrees=True)[1]
    return achieved_pitch
# --- PITCH_CONTROL MODIFICATION END ---


def main():
    parser = argparse.ArgumentParser(description="Run scoop followed by tilt-release.")
    parser.add_argument("scooping_filename", help="JSON file with Pre/Post-scooping moves.")
    parser.add_argument("positions_filename", help="JSON file with container and spoon positions.")
    parser.add_argument("--samples", type=int, help="Number of samples to run", default=1)
    parser.add_argument("--config", help="Path to configuration file", default="config.json")
    parser.add_argument("--no_vision", action="store_true", help="Skip scoop powder detection", default=False)
    parser.add_argument("--no_tilt_recording", action="store_true", help="Skip tilt-release camera recording", default=False)
    args = parser.parse_args()

    config = load_config(args.config)
    print("Configuration loaded successfully:")
    print("Running tilt-release experiment without weighing or shake")

    # Route only this script through the pitch-control vision module.
    scooping_machine.detect_powder = detect_powder_pitchcontrol

    robot = Robot(config["robot_ip"])

    if config["library"] == "franky":
        scooper = ScoopingMachine(
            args.scooping_filename,
            args.positions_filename,
            verbose=False,
            robot=robot,
            config=config
        )
    else:
        scooper = ScoopingMachine(
            args.scooping_filename,
            args.positions_filename,
            verbose=True,
            config=config
        )

    scooper.load_powder()
    scooper.pickup_spoon()

    # --- PITCH_CONTROL MODIFICATION START ---
    # No target weights, no scale, and no CSV weighing results. Each sample only
    # performs scoop -> move closer to base -> continuous tilt-release.
    i = 0
    print(f"Starting tilt-release experiment for {args.samples} samples...")
    while i < args.samples:
        try:
            scoop_success, scoop_angle = scooper.scoop(vision_check=False)
        except:
            robot.recover_from_errors()
            continue

        try:
            achieved_pitch = tilt_release_after_scoop(
                robot,
                record_tilt_release=not args.no_tilt_recording,
                sample_index=i + 1
            )
        except:
            robot.recover_from_errors()
            continue

        i += 1
        print(f"Tilt-release complete: sample={i}, scoop_angle={scoop_angle}, achieved_pitch={achieved_pitch}")
        scooper.reset_scoop_pose()
    # --- PITCH_CONTROL MODIFICATION END ---

    scooper.drop_spoon()
    scooper.unload_powder()


if __name__ == "__main__":
    main()