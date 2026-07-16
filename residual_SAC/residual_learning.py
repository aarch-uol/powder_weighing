import os
import sys
import argparse
import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from wandb.integration.sb3 import WandbCallback
import wandb 


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# Import your custom modules
from scooping_machine import ScoopingMachine

from weighing_environment import WeighingEnv
from stable_baselines3.common.callbacks import BaseCallback
from residual_env import ResidualAwareWrapper
import json 

from stable_baselines3.common.callbacks import BaseCallback
import wandb


class LogEpisodeErrorCallback(BaseCallback):
    def __init__(self, starting_episode: int = 0, verbose: int = 0):
        super().__init__(verbose)
        # Initialize the counter to whatever the previous run left off at
        self.episode_count = starting_episode 

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info and "final_error" in info:
                self.episode_count += 1
                
                # Log everything as parallel metrics without overriding the step parameter
                wandb.log({
                    "episode/reward": info["episode"]["r"],
                    "episode/final_error": info["final_error"],
                    "episode/number": self.episode_count,
                    "episode/total_timesteps": self.num_timesteps,
                    "episode/raw_weight_obs": info["raw_weight_obs"]
                })
        return True

#   --- Configuration Loader ---
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
        sys.exit(1)

def parse_args():
    parser = argparse.ArgumentParser(description="Residual Learning Material Training Script")  
    parser.add_argument("--config", help="Path to configuration file", default="config.json")
    parser.add_argument("--timesteps", type=int, default=600, help="Timesteps to train on THIS material") 
    parser.add_argument("--seed", type=int, default=1337, help="Random seed (also acts as the unique W&B identifier)")
    parser.add_argument("--material_name", type=str, required=True, help="Name of current material ")
    parser.add_argument("scooping_filename", help="JSON file with Pre/Post-scooping moves.")
    parser.add_argument("positions_filename", help="JSON file with container and spoon positions.")
    # Checkpointing arguments
    parser.add_argument("--load_checkpoint", type=str, default=None, help="Path to an existing SB3 residual checkpoint zip to continue from") 
    parser.add_argument("--load_replay_buffer", type=str, default=None, help="Path to an existing replay buffer (.pkl) to load")
    return parser.parse_args()


def main():
    args = parse_args()

    config = load_config(args.config)
    # STATIC DIRECTORIES
    log_dir = "./logs/residual_sac_checkpoints/"
    tb_log_dir = "./logs/residual_sac_tensorboard/"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(tb_log_dir, exist_ok=True)

    deterministic_run_id = f"residual-scoop-seed-{args.seed}"

    run = wandb.init(
        project="Residual Learning Powder Weighing plain MLP",
        id=deterministic_run_id,
        resume="allow",
        sync_tensorboard=True,
    )
    wandb.define_metric("episode/final_error", step_metric="episode/number")
    wandb.define_metric("episode/reward", step_metric="episode/number")
    wandb.define_metric("episode/raw_weight_obs", step_metric="episode/number")

    # --- NEW: Retrieve the starting episode number ---
    starting_episode = 0
    if run.resumed:
        # Fetch the last recorded episode number from the resumed run's summary
        starting_episode = run.summary.get("episode/number", 0)
        print(f"--> Resuming W&B Run. Starting from episode: {starting_episode}")
        
    # GET ALREADY TRAINED ON MATERIALS
    trained_materials = run.summary.get("trained_materials", [])
    if args.material_name not in trained_materials:
        trained_materials.append(args.material_name)
        run.summary["trained_materials"] = trained_materials
        print(f"--> Adding new material to W&B summary: {args.material_name}")
    
    base_env = WeighingEnv(config["robot_ip"], scale_port=config["scale_port"], gripper_port=config["gripper_port"], pitch_adjustment=False, min_target=10, max_target=20, duration_based_shake=True, new_setup=True, reward_type='error_change', early_stop=False)
    
    scooper = ScoopingMachine(args.scooping_filename, args.positions_filename, verbose=False, robot=base_env.robot.robot, config=config)

    # 3. Wrap the environment for Residual Learning
    env = ResidualAwareWrapper(
        env=base_env,
        scooping_machine=scooper,
        action_scaling_factor=0.2,
        model_path=config["residual_learning"]["baseline_model_path"],
    )
    env = Monitor(env)

    # 4. Handle Model Loading or Initialization
    if args.load_checkpoint:
        print(f"--> Loading existing residual policy from: {args.load_checkpoint}")
        if not os.path.exists(args.load_checkpoint):
            raise FileNotFoundError(f"Specified checkpoint file {args.load_checkpoint} not found.")

        # Load the model and bind it to the newly initialized environment
        residual_model = SAC.load(args.load_checkpoint, env=env)
        
        # Explicitly preserve or reset tensorboard log paths if desired
        residual_model.tensorboard_log = "./logs/residual_sac_tensorboard/"
        if args.load_replay_buffer :
            if not os.path.exists(args.load_replay_buffer):
                raise FileNotFoundError(f"Specified replay buffer file {args.load_replay_buffer} not found.")
            print(f"--> Loading existing replay buffer from: {args.load_replay_buffer}")
            residual_model.load_replay_buffer(args.load_replay_buffer)
        else:
            print("--> WARNING: Resuming model without a replay buffer. Performance may temporarily dip.")
    

         # --- NEW: Print loaded parameters to verify ---
        print("\n=== VERIFYING ACTIVE MODEL PARAMETERS ===")
        print(f"Timesteps (Clock):     {residual_model.num_timesteps}")
        print(f"Batch Size:            {residual_model.batch_size}")
        print(f"Train Freq:            {residual_model.train_freq}")
        print(f"Gradient Steps:        {residual_model.gradient_steps}")
        print(f"Learning Rate:         {residual_model.learning_rate}")
        print(f"Configured ent_coef:   {residual_model.ent_coef}")
        

        if residual_model.replay_buffer is not None:
            current_size = residual_model.replay_buffer.size()
            max_capacity = residual_model.replay_buffer.buffer_size
            print(f"Buffer Transitions:    {current_size} (Max Capacity: {max_capacity})")
        else:
            print("Buffer Transitions:    None (No buffer attached)")
    else:
        print("--> Starting a fresh residual policy from scratch.")
        residual_model = SAC(
            "MlpPolicy",
            env,
            learning_rate=1e-4,
            batch_size=64,
            verbose=1,
            learning_starts=100,
            seed=args.seed,
            ent_coef='auto',
            tensorboard_log="./logs/residual_sac_tensorboard/",
        )

    # 5. Set up a Checkpoint Callback (Saves periodically during this material's run)
    checkpoint_callback = CheckpointCallback(
        save_freq=100,
        save_path=log_dir,
        name_prefix=f"residual_{args.material_name}_{deterministic_run_id}",
    )
    wandb_callback = WandbCallback(verbose=2)

    error_logging_callback = LogEpisodeErrorCallback(starting_episode=starting_episode, verbose=1)


    # 1. Initialize physical hardware components
   

    scooper.load_powder()
    scooper.pickup_spoon()

    print(f"Starting training session for material: {args.material_name}...")

    # 6. Train the model for the designated timesteps
    try:
        residual_model.learn(
            total_timesteps=args.timesteps,
            callback=[checkpoint_callback, wandb_callback, error_logging_callback],
            log_interval=1,
            reset_num_timesteps=False,  # CRITICAL: Keeps the internal step counter continuous across materials
        )
        final_save_path = os.path.join(log_dir, f"final_{trained_materials}_{deterministic_run_id}_policy")
        buffer_save_path = os.path.join(log_dir, f"final_{trained_materials}_{deterministic_run_id}_replay_buffer")
    except Exception as e:
        print(f"\nTraining interrupted. Saving emergency checkpoint... {e}")
        final_save_path = os.path.join(log_dir, f"final_following_exception_{trained_materials}_{deterministic_run_id}_policy")
        buffer_save_path = os.path.join(log_dir, f"final_following_exception_{trained_materials}_{deterministic_run_id}_replay_buffer")
    # 7. Save final model for this material
    
    residual_model.save(final_save_path)
    residual_model.save_replay_buffer(buffer_save_path)

    print(f"Training complete for {trained_materials}. Model saved to {final_save_path}.zip")
    print("You can now safely swap materials and run the next command.")
    scooper.reset_scoop_pose()
    scooper.drop_spoon()
    scooper.unload_powder()

if __name__ == "__main__":
    main()