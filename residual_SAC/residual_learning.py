import os
import argparse
import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from wandb.integration.sb3 import WandbCallback
import wandb 

# Import your custom modules
from scooping_machine import ScoopingMachine
from your_env_file import ResidualAwareWrapper  # Replace with your actual file name


def parse_args():
    parser = argparse.ArgumentParser(description="Residual Learning Material Training Script")  )
    parser.add_argument("--base_model_path",type=str,required=True,help="Path to pre-trained base controller")
    parser.add_argument("--timesteps", type=int, default=300, help="Timesteps to train on THIS material") 
    parser.add_argument("--seed", type=int, default=1337, help="Random seed (also acts as the unique W&B identifier)")
    parser.add_argument("--material_name", type=str, required=True, help="Name of current material ")
    parser.add_argument("scooping_filename", help="JSON file with Pre/Post-scooping moves.")
    parser.add_argument("positions_filename", help="JSON file with container and spoon positions.")
    # Checkpointing arguments
    parser.add_argument("--load_checkpoint", type=str, default=None, help="Path to an existing SB3 residual checkpoint zip to continue from") 
    
    return parser.parse_args()


def main():
    args = parse_args()

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
    
    # 1. Initialize physical hardware components
    scooper = ScoopingMachine(args.scooping_filename, args.positions_filename, verbose=False, robot=env.env.robot.robot, config=config)

    # 2. Initialize the base physical environment
    base_env = gym.make("powder_weighing_envII_small")

    # 3. Wrap the environment for Residual Learning
    env = ResidualAwareWrapper(
        env=base_env,
        scooping_machine=scooper,
        action_scaling_factor=0.2,
        model_path=args.base_model_path,
    )
    env = Monitor(env)

    # 4. Handle Model Loading or Initialization
    if args.load_checkpoint and os.path.exists(args.load_checkpoint):
        print(f"--> Loading existing residual policy from: {args.load_checkpoint}")
        # Load the model and bind it to the newly initialized environment
        residual_model = SAC.load(args.load_checkpoint, env=env)
        
        # Explicitly preserve or reset tensorboard log paths if desired
        residual_model.tensorboard_log = "./logs/residual_sac_tensorboard/"
    else:
        print("--> Starting a fresh residual policy from scratch.")
        residual_model = SAC(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            verbose=1,
            seed=args.seed,
            tensorboard_log="./logs/residual_sac_tensorboard/",
        )

    # 5. Set up a Checkpoint Callback (Saves periodically during this material's run)
    checkpoint_callback = CheckpointCallback(
        save_freq=100,
        save_path=log_dir,
        name_prefix=f"residual_{args.material_name}",
    )
    wandb_callback = WandbCallback(verbose=2)

    print(f"Starting training session for material: {args.material_name}...")

    # 6. Train the model for the designated timesteps
    try:
        residual_model.learn(
            total_timesteps=args.timesteps,
            callback=[checkpoint_callback, wandb_callback],
            log_interval=0,
            reset_num_timesteps=False,  # CRITICAL: Keeps the internal step counter continuous across materials
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving emergency checkpoint...")

    # 7. Save final model for this material
    final_save_path = os.path.join(log_dir, f"final_{args.material_name}_policy")
    residual_model.save(final_save_path)
    print(f"Training complete for {args.material_name}. Model saved to {final_save_path}.zip")
    print("You can now safely swap materials and run the next command.")


if __name__ == "__main__":
    main()