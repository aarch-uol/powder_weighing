# Powder Weighing Experiment Script

## Overview

`smart_powder_weighting.py` is an automated script for conducting powder weighing experiments using a robotic arm with reinforcement learning (SAC - Soft Actor-Critic) control. It orchestrates the complete workflow from powder loading to measurement and data collection.

## Prerequisites

### Hardware
- FR3 (10.0.0.1 by default)
- Robotiq 85F gripper
- Scale (Sartorius Entris II or Fisherbrand FPRS22)
- RealSense D405 (for powder detection)
- Nvidia GPU, 3080 or newer. 

### Software
- Ubuntu RT kernel
- Python 3.7+
- Required packages: `torch`, `numpy`, `pyrobotiqgripper`, `franky`
- Pre-trained SAC models (see Models section)
- JSON configuration files for scooping movements and positions

## Input Files

### 1. Scooping Configuration (`scooping_filename`)
JSON file containing pre- and post-scooping movement sequences with gripper and robot actions, defined in the robots Joint space. 

### 2. Positions Configuration (`positions_filename`)
JSON file with container, spoon, and endpoint position definitions for the robotic arm. For accuracy reasons these are defines as grasp points in the robots Joint space. 

## Usage

```bash
python smart_powder_weighting.py \
  <scooping_filename> \
  <positions_filename> \
  <output_directory> \
  <model_name> \
  <powder_name> \
  [--samples N] \
  [--robot_ip IP] \
  [--scale_port PORT] \
  [--gripper_port PORT]
```

### Arguments

| Argument | Type | Description | Default |
|----------|------|-------------|---------|
| `scooping_filename` | string | Path to scooping moves JSON | Required |
| `positions_filename` | string | Path to positions JSON | Required |
| `directory` | string | Directory to save experiment results | Required |
| `model` | string | Model to use (see Models section) | Required |
| `powder` | string | Name of powder being tested | Required |
| `--samples` | int | Number of samples per target weight | 1 |
| `--robot_ip` | string | Robot IP address | 10.0.0.1 |
| `--scale_port` | string | Scale serial port | /dev/ttyACM0 |
| `--gripper_port` | string | Gripper serial port | /dev/ttyUSB0 |

### Example

```bash
python smart_powder_weighting.py \
  scooping_movements.json \
  positions.json \
  ./results \
  curriculum \
  flour \
  --samples 5 \
  --robot_ip 10.0.0.1
```

## Available Models

The script supports these pre-trained SAC models:

- **`curriculum`**:  FLIP Curriculum learning-based model [1]
- **`random`**: FLIP Random model [1]
- **`dr`**: Domain randomization model 
- **`reverse`**: FLIP reverse curriculum model [1]
- **`random_new_reward`**: FLIP Random with new reward_function

Models must be located in the `./models` directory with their respective checkpoint files.



## Output Files

Results are saved as CSV files with naming pattern: `experiment_{powder}_{target_weight}mg.csv`

### CSV Format
```
Final Weight,Target weight,Error,Scoop Angle
15.2,15,0.2,35
14.8,15,0.2,38
...
Average,15,0.2,0.1
```

Each file includes:
- Header row with column names
- Individual measurement rows (Final Weight, Target, Error, Scoop Angle)
- Summary row with average and standard deviation

## Configuration

### Shake Dynamics

The script automatically adjusts shake dynamics based on scoop angle:

```python
SLOW_SHAKE = [0.25, 0.15, 0.15]   # For angles < 40°
FAST_SHAKE = [0.4, 0.22, 0.35]    # For angles >= 40°
```

## Key Classes and Functions

| Class/Function | File | Purpose |
|---|---|---|
| `SACAgent` | SAC/SACAgent.py | RL agent for policy execution |
| `WeighingEnv` | weighing_environment.py | Environment interface with robot and scale |
| `ScoopingMachine` | scooping_mechanism.py | Automated scooping sequences |
| `InterfaceEnvironment` | smart_powder_weighting.py | Wrapper for environment compatibility |

## Troubleshooting

### "Model {name} not known" Error
Ensure the model name matches one in the `models` dictionary at the start of main() and the model file exists.

**Solution:**
```bash
ls ./models/
# Add new model to models dictionary if needed
```

### Robot Connection Issues
- Verify robot IP address (default: 10.0.0.1)
- Check network connectivity:
```bash
ping 10.0.0.1
```
- Ensure Franka desk is in automatic mode

### Serial Port Errors (Scale or Gripper)
- List available serial ports:
```bash
ls /dev/tty*
```
- Verify correct ports are passed via command line arguments
- Check device permissions:
```bash
ls -la /dev/ttyACM0 /dev/ttyUSB0
```

### Scoop Success Failures
The script will:
1. Recover from robot errors automatically
2. Prompt user if scoop is unsuccessful
3. Allow retry or program termination

Press ENTER to continue or close the program if needed.


## Interactive Prompts

The script includes several interactive prompts:

1. **Skip target weight confirmation**: Type `y` to skip, `n` to proceed
2. **Scoop failure confirmation**: Press ENTER to retry or Ctrl+C to exit
3. **System status messages**: Shows scoop angle, shake dynamics, and error metrics

## Data Analysis

Results CSV files can be processed for analysis:
- Average error per powder type
- Success rates by target weight
- Scoop angle effectiveness analysis
- Standard deviation of measurements

## Notes

- Experiments are **interactive** - prompts for confirmation between phases
- Each experiment run creates a new CSV file or appends to existing one
- System automatically recovers from robot errors (except fatal ones)
- Shake dynamics are adjusted dynamically based on scoop angle detection
- All weights are measured in miligrams
- Target weights tested: 10mg, 15mg, 20mg
