#Created by: Thomas Little
#Email: sgtlittl@liverpool.ac.uk
import pyrealsense2 as rs
import numpy as np
import cv2
import time
import json
import os
from collections import deque 

# --- Configuration Loader ---
def load_config(config_filename="config.json"):
    """Load configuration from JSON file."""
    try:
        # Try to find config.json in the same directory as this script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(script_dir, config_filename)
        
        with open(config_path) as f:
            config = json.load(f)
        return config
    except Exception as e:
        print(f"ERROR: Failed to load configuration file '{config_filename}'\n{e}")
        # Return default values if config loading fails
        return {
            "vision_detection": {
                "container_roi": [340, 320, 50, 35],
                "spoon_roi": [175, 220, 90, 55],
                "color_match_threshold": 30,
                "empty_spoon_percentage_threshold": 16,
                "container_sample_interval": 1.0,
                "scoop_status_update_interval": 2.0,
                "consecutive_status_to_close": 5,
                "debug_mode": False
            }
        }


# Load configuration
CONFIG = load_config()

# --- Configuration Parameters ---
# Fixed Regions of Interest (ROIs) for the container and spoon.
_vision_config = CONFIG.get("vision_detection", {})
CONTAINER_ROI = tuple(_vision_config.get("container_roi", [340, 320, 50, 35]))
SPOON_ROI = tuple(_vision_config.get("spoon_roi", [175, 220, 90, 55]))

# Color matching threshold: Maximum Euclidean distance for two colors to be considered a match.
COLOR_MATCH_THRESHOLD = _vision_config.get("color_match_threshold", 30)

# Percentage threshold for empty spoon
EMPTY_SPOON_PERCENTAGE_THRESHOLD = _vision_config.get("empty_spoon_percentage_threshold", 16)

# Time interval for sampling container color (in seconds).
CONTAINER_SAMPLE_INTERVAL = _vision_config.get("container_sample_interval", 1.0)

# Time interval for updating SCOOP_STATUS output (in seconds).
SCOOP_STATUS_UPDATE_INTERVAL = _vision_config.get("scoop_status_update_interval", 2.0)

# Number of spaced-out SCOOP_STATUS results to collect for modal analysis before closing.
CONSECUTIVE_STATUS_TO_CLOSE = _vision_config.get("consecutive_status_to_close", 5)

# DEBUG MODE: Set to True to enable continuous camera window until 'q' is pressed
DEBUG_MODE = _vision_config.get("debug_mode", False)

# --- Helper Functions ---
def get_average_color(image, roi):
    """
    Calculates the average color of a specified ROI in an image.
    The input 'image' is expected to be in the color space you are working with (e.g., HSV).
    Args:
        image (np.array): The input image (e.g., HSV).
        roi (tuple): A tuple (x, y, w, h) defining the region of interest.
    Returns:
        np.array: A 3-element array representing the average color.
    """
    x, y, w, h = roi
    # Check for invalid ROI dimensions or out-of-bounds
    if x < 0 or y < 0 or w <= 0 or h <= 0 or x + w > image.shape[1] or y + h > image.shape[0]:
        return np.array([0, 0, 0]) # Return black if ROI is invalid

    roi_image = image[y:y+h, x:x+w]
    if roi_image.size == 0: # Check if the extracted ROI is empty
        return np.array([0, 0, 0])
    return np.mean(roi_image, axis=(0, 1)).astype(int)

def color_distance(color1, color2):
    """
    Calculates the Euclidean distance between two colors.
    The colors are expected to be in the same color space (e.g., HSV).
    Args:
        color1 (np.array): First color.
        color2 (np.array): Second color.
    Returns:
        float: Euclidean distance between the colors.
    """
    return np.linalg.norm(color1 - color2)

def detect_powder():
    """
    Main function to detect powder on spoon and return the final modal outcome.
    Returns:
        bool or None: True if scoop successful, False if empty, None if inconclusive
    """
    # --- Realsense Pipeline Setup ---
    pipeline = rs.pipeline()
    config = rs.config()

    # Enable color stream
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # Start streaming
    try:
        pipeline.start(config)
        print("RealSense camera started.")
    except Exception as e:
        print(f"Failed to start RealSense pipeline: {e}")
        print("Please ensure the camera is connected and drivers are installed.")
        return None

    # --- Variables for continuous sampling ---
    last_container_sample_time = time.time()
    container_avg_color_hsv = np.array([0, 0, 0]) # Initialize with black in HSV
    last_scoop_status_update_time = time.time()
    
    # New: Deque to store the history of SCOOP_STATUS prints for modal analysis
    scoop_status_history = deque(maxlen=CONSECUTIVE_STATUS_TO_CLOSE)
    
    # Variable to store the final modal result before closing
    final_modal_outcome = None
    # average detected coverage
    avg_coverage = 0

    if not DEBUG_MODE:
        print(f"Program will close after {CONSECUTIVE_STATUS_TO_CLOSE} spaced-out SCOOP_STATUS results show a clear modal value.")

    try:
        while True:
            percentage_filled =0
            # Wait for a new camera frame
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            # Convert images to numpy arrays
            color_image = np.asanyarray(color_frame.get_data())
            raw_image=color_image.copy()
            # Convert the color image to HSV for processing
            hsv_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2LAB)

            # --- Container Color Sampling (every second) ---
            current_time = time.time()
            if current_time - last_container_sample_time >= CONTAINER_SAMPLE_INTERVAL:
                if CONTAINER_ROI[2] > 0 and CONTAINER_ROI[3] > 0:
                    container_avg_color_hsv = get_average_color(hsv_image, CONTAINER_ROI)
                    last_container_sample_time = current_time
                else:
                    cv2.putText(color_image, "Invalid CONTAINER ROI dimensions.", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # --- Spoon Detection Logic ---
            spoon_x, spoon_y, spoon_w, spoon_h = SPOON_ROI
            
            current_scoop_successful = False # Default status for current frame
            if (CONTAINER_ROI[2] > 0 and CONTAINER_ROI[3] > 0) and \
               (SPOON_ROI[2] > 0 and SPOON_ROI[3] > 0):

                spoon_roi_image_hsv = hsv_image[spoon_y:spoon_y+spoon_h, spoon_x:spoon_x+spoon_w].copy()

                if spoon_roi_image_hsv.size > 0:
                    match_mask = np.zeros(spoon_roi_image_hsv.shape[:2], dtype=np.uint8)
                    for r_idx in range(spoon_roi_image_hsv.shape[0]):
                        for c_idx in range(spoon_roi_image_hsv.shape[1]):
                            pixel_color_hsv = spoon_roi_image_hsv[r_idx, c_idx]
                            dist_to_container = color_distance(pixel_color_hsv, container_avg_color_hsv)
                            if dist_to_container < COLOR_MATCH_THRESHOLD:
                                match_mask[r_idx, c_idx] = 255

                    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(match_mask, 8, cv2.CV_32S)

                    largest_component_area = 0
                    largest_component_label = 0
                    for i in range(1, num_labels):
                        if stats[i, cv2.CC_STAT_AREA] > largest_component_area:
                            largest_component_area = stats[i, cv2.CC_STAT_AREA]
                            largest_component_label = i

                    effective_spoon_area = spoon_w * spoon_h
                    percentage_filled = (largest_component_area / effective_spoon_area) * 100 if effective_spoon_area > 0 else 0.0

                    is_empty = percentage_filled < EMPTY_SPOON_PERCENTAGE_THRESHOLD
                    current_scoop_successful = not is_empty
                    
                    # Update SCOOP_STATUS output at the specified interval and add to history
                    if current_time - last_scoop_status_update_time >= SCOOP_STATUS_UPDATE_INTERVAL:
                        print(f"SCOOP_STATUS: {current_scoop_successful}")
                        if not DEBUG_MODE:
                            scoop_status_history.append(current_scoop_successful) # Add to history
                            avg_coverage+=percentage_filled
                        last_scoop_status_update_time = current_time

                    # Highlight pixels of the LARGEST detected component in red
                    if largest_component_label > 0:
                        largest_component_mask = np.zeros_like(match_mask, dtype=np.uint8)
                        largest_component_mask[labels == largest_component_label] = 255
                        
                        matched_y_coords, matched_x_coords = np.where(largest_component_mask == 255)
                        for r_idx, c_idx in zip(matched_y_coords, matched_x_coords):
                            if (spoon_y + r_idx < color_image.shape[0] and
                                spoon_x + c_idx < color_image.shape[1]):
                                color_image[spoon_y + r_idx, spoon_x + c_idx] = [0, 0, 255]
                
                    # Display results on the frame (Text color changed to RED)
                    cv2.putText(color_image, f"Container LAB Avg: {container_avg_color_hsv}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    status_text = f"Spoon Filled: {percentage_filled:.2f}% ({'EMPTY' if is_empty else 'SUCCESSFUL SCOOP'})"
                    cv2.putText(color_image, status_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    
                    
                else:
                    cv2.putText(color_image, "Spoon ROI image is empty after crop.", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    current_scoop_successful = False # No successful scoop if ROI is empty
                    if current_time - last_scoop_status_update_time >= SCOOP_STATUS_UPDATE_INTERVAL:
                        print(f"SCOOP_STATUS: {current_scoop_successful}")
                        if not DEBUG_MODE:
                            avg_coverage+=percentage_filled
                            scoop_status_history.append(current_scoop_successful) # Add to history
                        last_scoop_status_update_time = current_time
            else:
                cv2.putText(color_image, "Invalid ROI dimensions.", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                current_scoop_successful = False # No successful scoop if ROIs are invalid
                if current_time - last_scoop_status_update_time >= SCOOP_STATUS_UPDATE_INTERVAL:
                    print(f"SCOOP_STATUS: {current_scoop_successful}")
                    if not DEBUG_MODE:
                        scoop_status_history.append(current_scoop_successful) # Add to history
                        avg_coverage+=percentage_filled
                    last_scoop_status_update_time = current_time

            # --- Logic to close based on modal SCOOP_STATUS history (only in normal mode) ---
            if not DEBUG_MODE and len(scoop_status_history) == CONSECUTIVE_STATUS_TO_CLOSE:
                true_count = scoop_status_history.count(True)
                false_count = scoop_status_history.count(False)
                # Determine the modal outcome
                if true_count > false_count:
                    final_modal_outcome = True
                elif false_count > true_count:
                    final_modal_outcome = False
                else:
                    final_modal_outcome = None # No clear majority/tie

                # If a clear majority exists, close the program
                if final_modal_outcome is not None:
                    avg_coverage=avg_coverage/CONSECUTIVE_STATUS_TO_CLOSE
                    break

            # --- Draw ROIs on the frame ---
            # Container ROI (Blue)
            if CONTAINER_ROI[2] > 0 and CONTAINER_ROI[3] > 0:
                cv2.rectangle(color_image, (CONTAINER_ROI[0], CONTAINER_ROI[1]),
                              (CONTAINER_ROI[0] + CONTAINER_ROI[2], CONTAINER_ROI[1] + CONTAINER_ROI[3]),
                              (255, 0, 0), 2)
                cv2.putText(color_image, "Container", (CONTAINER_ROI[0], CONTAINER_ROI[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            # Spoon ROI (Green)
            if SPOON_ROI[2] > 0 and SPOON_ROI[3] > 0:
                cv2.rectangle(color_image, (SPOON_ROI[0], SPOON_ROI[1]),
                              (SPOON_ROI[0] + SPOON_ROI[2], SPOON_ROI[1] + SPOON_ROI[3]),
                              (0, 255, 0), 2)
                cv2.putText(color_image, "Spoon", (SPOON_ROI[0], SPOON_ROI[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Display the resulting frame
            cv2.imshow('RealSense Spoon Detection', color_image)
            cv2.imshow('Raw image output', raw_image)
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                if DEBUG_MODE:
                    print("Debug mode terminated by user.")
                break

    except Exception as e:
        print(f"An error occurred during streaming: {e}")
        final_modal_outcome = None
    finally:
        # Stop streaming
        pipeline.stop()
        cv2.destroyAllWindows()
        print("RealSense pipeline stopped and windows closed.")
    
    return final_modal_outcome, avg_coverage

def main():
    """Standalone function to run the detection when script is executed directly."""
    if DEBUG_MODE:
        print("Running in DEBUG MODE - Camera window will stay open until 'q' is pressed")
    result = detect_powder()
    print(f"Final result: {result}")
    return result

if __name__ == "__main__":
    main()