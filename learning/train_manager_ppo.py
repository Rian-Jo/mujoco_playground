"""Training script for manager-based PPO with MuJoCo Playground.

This script provides a complete training pipeline for manager-based environments,
compatible with JAX/Flax/Brax/MJX/MJWarp.

Example usage:
    python learning/train_manager_ppo.py \\
        --env_name Go1-velocity-manager \\
        --num_envs 4096 \\
        --num_timesteps 50000000 \\
        --backend mjx
"""

import functools
import os
from datetime import datetime
from typing import Any, Dict, Optional

from absl import app, flags
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as train_ppo
import jax
from ml_collections import config_dict

from mujoco_playground.manager import ManagerBasedEnv, ManagerBasedConfig
from mujoco_playground._src import wrapper as brax_wrapper

# ==================== Flags ====================

FLAGS = flags.FLAGS

# Environment flags
flags.DEFINE_string("env_name", "Go1-velocity-manager", "Name of the manager-based environment")
flags.DEFINE_string("xml_path", None, "Path to MuJoCo XML (required if not using env_name)")
flags.DEFINE_integer("num_envs", 4096, "Number of parallel environments")
flags.DEFINE_string("backend", "mjx", "Physics backend: mjx or warp")

# Training flags
flags.DEFINE_integer("num_timesteps", 50_000_000, "Total number of training timesteps")
flags.DEFINE_integer("num_evals", 10, "Number of evaluation checkpoints")
flags.DEFINE_integer("episode_length", 1000, "Maximum episode length")
flags.DEFINE_integer("batch_size", 256, "Training batch size")
flags.DEFINE_float("learning_rate", 3e-4, "Learning rate")

# Network flags
flags.DEFINE_list("policy_hidden_layers", [512, 256, 128], "Policy network hidden layer sizes")
flags.DEFINE_list("value_hidden_layers", [512, 256, 128], "Value network hidden layer sizes")

# Logging flags
flags.DEFINE_string("output_dir", "./manager_training_logs", "Directory for training outputs")
flags.DEFINE_bool("use_wandb", False, "Enable Weights & Biases logging")
flags.DEFINE_string("wandb_project", "mujoco-playground-manager", "WandB project name")
flags.DEFINE_string("wandb_entity", None, "WandB entity name")

# Config override flags
flags.DEFINE_string("config_file", None, "Path to manager config file (Python module)")


# ==================== Environment Registry ====================

def get_example_velocity_tracking_config() -> config_dict.ConfigDict:
    """Example configuration for velocity tracking task.

    This demonstrates how to configure a complete manager-based environment.
    """
    from mujoco_playground.manager.observation_manager import (
        ObservationTerm,
        get_joint_positions,
        get_joint_velocities,
        get_base_orientation,
        get_base_linear_velocity,
        get_base_angular_velocity,
        get_last_action,
        get_current_command,
    )
    from mujoco_playground.manager.reward_manager import (
        RewardTerm,
        reward_alive,
        reward_energy_penalty,
        reward_velocity_tracking,
        reward_orientation_penalty,
    )
    from mujoco_playground.manager.termination_manager import (
        TerminationTerm,
        termination_bad_orientation,
        termination_height_limit,
    )
    from mujoco_playground.manager.command_manager import command_velocity_tracking

    config = config_dict.create(
        # Physics settings
        ctrl_dt=0.02,
        sim_dt=0.004,

        # Action settings
        action_dim=12,  # For quadruped
        action_space="position",
        action_scale=0.5,
        action_clip=(-1.0, 1.0),
        default_pose=None,  # Will use current pose

        # Observation settings
        observation_terms=config_dict.create(
            policy=config_dict.create(
                joint_pos=config_dict.create(
                    func=get_joint_positions,
                    noise_config={"std": 0.01},
                    clip_range=(-10.0, 10.0),
                    scale=1.0,
                    history_len=1,
                ),
                joint_vel=config_dict.create(
                    func=get_joint_velocities,
                    noise_config={"std": 0.1},
                    clip_range=(-20.0, 20.0),
                    scale=0.1,
                    history_len=1,
                ),
                base_orientation=config_dict.create(
                    func=get_base_orientation,
                    noise_config={"std": 0.05},
                    scale=1.0,
                ),
                base_lin_vel=config_dict.create(
                    func=get_base_linear_velocity,
                    noise_config={"std": 0.1},
                    scale=1.0,
                ),
                base_ang_vel=config_dict.create(
                    func=get_base_angular_velocity,
                    noise_config={"std": 0.1},
                    scale=0.1,
                ),
                last_action=config_dict.create(
                    func=get_last_action,
                    scale=1.0,
                ),
                command=config_dict.create(
                    func=get_current_command,
                    scale=1.0,
                ),
            )
        ),
        concatenate_groups=True,
        add_noise=True,

        # Reward settings
        reward_terms=config_dict.create(
            alive=config_dict.create(
                func=reward_alive,
                weight=0.1,
            ),
            velocity_tracking=config_dict.create(
                func=reward_velocity_tracking,
                weight=2.0,
            ),
            orientation=config_dict.create(
                func=reward_orientation_penalty,
                weight=0.5,
            ),
            energy=config_dict.create(
                func=reward_energy_penalty,
                weight=0.01,
            ),
        ),

        # Termination settings
        termination_terms=config_dict.create(
            bad_orientation=config_dict.create(
                func=termination_bad_orientation,
                time_out=False,
                params={"threshold": 0.5},
            ),
            height_limit=config_dict.create(
                func=termination_height_limit,
                time_out=False,
                params={"min_height": 0.2, "max_height": 2.0},
            ),
        ),
        episode_length=1000,

        # Command settings
        command_func=command_velocity_tracking,
        command_dim=3,
        resample_interval=500,
        resample_on_reset=True,
        command_ranges=config_dict.create(
            lin_vel_x=(-1.0, 1.0),
            lin_vel_y=(-0.5, 0.5),
            ang_vel_z=(-1.0, 1.0),
        ),

        # Event settings (empty for now)
        event_terms={},

        # Curriculum settings (empty for now)
        curriculum_terms={},
        curriculum_mode="threshold",
        curriculum_threshold=0.8,
    )

    return config


# ==================== Training Functions ====================

def create_manager_env(
    xml_path: str,
    config: config_dict.ConfigDict,
    num_envs: int,
    backend: str = "mjx",
) -> ManagerBasedEnv:
    """Create a manager-based environment.

    Args:
        xml_path: Path to MuJoCo XML model
        config: Manager configuration
        num_envs: Number of parallel environments
        backend: Physics backend

    Returns:
        ManagerBasedEnv instance
    """
    # Add environment metadata to config
    full_config = config.copy()
    full_config.xml_path = xml_path
    full_config.num_envs = num_envs
    full_config.backend = backend

    # Separate manager configs
    full_config.action_config = config.copy()
    full_config.observation_config = config.copy()
    full_config.reward_config = config.copy()
    full_config.termination_config = config.copy()
    full_config.command_config = config.copy()
    full_config.event_config = config.copy()
    full_config.curriculum_config = config.copy()

    env = ManagerBasedEnv(full_config)

    return env


def wrap_manager_env_for_training(
    env: ManagerBasedEnv,
    episode_length: int,
    num_envs: int,
) -> Any:
    """Wrap manager-based environment for Brax training.

    Args:
        env: Manager-based environment
        episode_length: Maximum episode length
        num_envs: Number of parallel environments

    Returns:
        Wrapped environment compatible with Brax training
    """
    # TODO: Create a proper wrapper that converts ManagerBasedEnv to Brax env
    # For now, we'll note that this needs to be implemented
    # The wrapper should:
    # 1. Convert ManagerBasedState to brax.envs.State
    # 2. Handle reset() and step() conversions
    # 3. Extract observations from info dict

    # Placeholder - would need actual implementation
    return env


def train_manager_ppo(
    env_name: str,
    xml_path: Optional[str] = None,
    num_envs: int = 4096,
    num_timesteps: int = 50_000_000,
    num_evals: int = 10,
    episode_length: int = 1000,
    backend: str = "mjx",
    policy_hidden_layers: tuple = (512, 256, 128),
    value_hidden_layers: tuple = (512, 256, 128),
    learning_rate: float = 3e-4,
    batch_size: int = 256,
    output_dir: str = "./manager_training_logs",
    use_wandb: bool = False,
    wandb_project: str = "mujoco-playground-manager",
    wandb_entity: Optional[str] = None,
    **kwargs,
) -> tuple:
    """Train a manager-based environment with PPO.

    Args:
        env_name: Name of the environment configuration
        xml_path: Path to MuJoCo XML (if not using env_name)
        num_envs: Number of parallel environments
        num_timesteps: Total training timesteps
        num_evals: Number of evaluation checkpoints
        episode_length: Maximum episode length
        backend: Physics backend
        policy_hidden_layers: Policy network architecture
        value_hidden_layers: Value network architecture
        learning_rate: Learning rate
        batch_size: Training batch size
        output_dir: Directory for outputs
        use_wandb: Enable WandB logging
        wandb_project: WandB project name
        wandb_entity: WandB entity name
        **kwargs: Additional arguments

    Returns:
        Tuple of (inference_fn, params, metrics)
    """
    print(f"Training manager-based environment: {env_name}")
    print(f"Backend: {backend}")
    print(f"Num envs: {num_envs}")
    print(f"Num timesteps: {num_timesteps}")

    # Load configuration
    if env_name == "Go1-velocity-manager":
        config = get_example_velocity_tracking_config()
        if xml_path is None:
            # Default to Go1 model
            xml_path = "path/to/go1.xml"  # TODO: Update with actual path
    else:
        raise ValueError(f"Unknown environment: {env_name}")

    # Override config with any additional kwargs
    config.update(kwargs)

    # Create environment
    print("Creating manager-based environment...")
    env = create_manager_env(xml_path, config, num_envs, backend)

    print("Note: Full Brax integration wrapper needs to be implemented.")
    print("This is a placeholder for the training pipeline.")

    # TODO: Implement full training pipeline
    # Steps needed:
    # 1. Create Brax-compatible wrapper
    # 2. Setup network factory
    # 3. Run PPO training
    # 4. Save checkpoints
    # 5. Evaluate policy

    return None, None, None


def main(argv):
    """Main training function."""
    del argv  # Unused

    # Parse hidden layer sizes
    policy_hidden_layers = tuple(int(x) for x in FLAGS.policy_hidden_layers)
    value_hidden_layers = tuple(int(x) for x in FLAGS.value_hidden_layers)

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(FLAGS.output_dir, f"{FLAGS.env_name}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    print(f"Output directory: {output_dir}")

    # Train
    inference_fn, params, metrics = train_manager_ppo(
        env_name=FLAGS.env_name,
        xml_path=FLAGS.xml_path,
        num_envs=FLAGS.num_envs,
        num_timesteps=FLAGS.num_timesteps,
        num_evals=FLAGS.num_evals,
        episode_length=FLAGS.episode_length,
        backend=FLAGS.backend,
        policy_hidden_layers=policy_hidden_layers,
        value_hidden_layers=value_hidden_layers,
        learning_rate=FLAGS.learning_rate,
        batch_size=FLAGS.batch_size,
        output_dir=output_dir,
        use_wandb=FLAGS.use_wandb,
        wandb_project=FLAGS.wandb_project,
        wandb_entity=FLAGS.wandb_entity,
    )

    print("Training complete!")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    app.run(main)
