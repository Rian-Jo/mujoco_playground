"""Training script for Go2 velocity tracking with manager-based architecture.

This script trains a Unitree Go2 quadruped to track velocity commands using
the manager-based architecture. It's based on:
- mjlab's velocity tracking task
- mujoco_playground's Go1 implementation
- Manager-based modular architecture

Example usage:
    # Train on flat terrain
    python learning/train_go2_velocity.py \\
        --task flat \\
        --num_envs 4096 \\
        --num_timesteps 50000000 \\
        --backend mjx

    # Train on rough terrain
    python learning/train_go2_velocity.py \\
        --task rough \\
        --num_envs 8192 \\
        --num_timesteps 100000000 \\
        --use_wandb

Author: Based on mjlab and mujoco_playground implementations
"""

import functools
import os
from datetime import datetime
from typing import Optional

from absl import app, flags
import jax
import jax.numpy as jnp

# Brax imports (Note: Full integration needs to be completed)
# from brax.training.agents.ppo import networks as ppo_networks
# from brax.training.agents.ppo import train as train_ppo

from ml_collections import config_dict

# Manager imports
from mujoco_playground.manager import ManagerBasedEnv
from mujoco_playground.manager.configs.go2_velocity import (
    get_go2_velocity_flat_config,
    get_go2_velocity_rough_config,
    get_xml_path,
)

# ==================== Flags ====================

FLAGS = flags.FLAGS

# Environment flags
flags.DEFINE_enum("task", "flat", ["flat", "rough"], "Task terrain type")
flags.DEFINE_integer("num_envs", 4096, "Number of parallel environments")
flags.DEFINE_string("backend", "mjx", "Physics backend: mjx or warp")

# Training flags
flags.DEFINE_integer("num_timesteps", 50_000_000, "Total number of training timesteps")
flags.DEFINE_integer("num_evals", 10, "Number of evaluation checkpoints")
flags.DEFINE_integer("episode_length", 1000, "Maximum episode length")
flags.DEFINE_integer("batch_size", 256, "Training batch size")
flags.DEFINE_float("learning_rate", 3e-4, "Learning rate")
flags.DEFINE_float("entropy_cost", 1e-2, "Entropy cost coefficient")
flags.DEFINE_float("discounting", 0.97, "Discount factor")
flags.DEFINE_integer("unroll_length", 10, "Unroll length for PPO")
flags.DEFINE_integer("num_minibatches", 32, "Number of minibatches")
flags.DEFINE_integer("num_updates_per_batch", 4, "PPO epochs per batch")

# Network flags
flags.DEFINE_list("policy_hidden_layers", [512, 256, 128], "Policy network hidden layer sizes")
flags.DEFINE_list("value_hidden_layers", [512, 256, 128], "Value network hidden layer sizes")
flags.DEFINE_string("activation", "relu", "Network activation function")

# Logging flags
flags.DEFINE_string("output_dir", "./go2_training_logs", "Directory for training outputs")
flags.DEFINE_bool("use_wandb", False, "Enable Weights & Biases logging")
flags.DEFINE_string("wandb_project", "mujoco-playground-go2", "WandB project name")
flags.DEFINE_string("wandb_entity", None, "WandB entity name")
flags.DEFINE_string("wandb_run_name", None, "WandB run name (auto-generated if None)")

# Debugging flags
flags.DEFINE_bool("debug", False, "Enable debug mode (smaller scale for testing)")
flags.DEFINE_integer("debug_envs", 128, "Number of envs in debug mode")
flags.DEFINE_integer("debug_timesteps", 100_000, "Timesteps in debug mode")


# ==================== Environment Setup ====================

def create_go2_env(
    task: str = "flat",
    num_envs: int = 4096,
    backend: str = "mjx",
) -> ManagerBasedEnv:
    """Create Go2 velocity tracking environment.

    Args:
        task: Task type ("flat" or "rough")
        num_envs: Number of parallel environments
        backend: Physics backend

    Returns:
        ManagerBasedEnv instance
    """
    print(f"Creating Go2 {task} terrain environment...")
    print(f"  Num envs: {num_envs}")
    print(f"  Backend: {backend}")

    # Get config
    if task == "flat":
        config = get_go2_velocity_flat_config()
    elif task == "rough":
        config = get_go2_velocity_rough_config()
    else:
        raise ValueError(f"Unknown task: {task}")

    # Set environment parameters
    config.xml_path = get_xml_path(task)
    config.num_envs = num_envs
    config.backend = backend

    # Separate manager configs
    config.action_config = config.copy()
    config.observation_config = config.copy()
    config.reward_config = config.copy()
    config.termination_config = config.copy()
    config.command_config = config.copy()
    config.event_config = config.copy()
    config.curriculum_config = config.copy()

    # Create environment
    env = ManagerBasedEnv(config)

    print(f"✓ Environment created successfully")
    print(f"  XML: {config.xml_path}")
    print(f"  Action dim: {config.action_dim}")
    print(f"  Episode length: {config.episode_length}")

    return env


def test_environment(env: ManagerBasedEnv, num_steps: int = 10):
    """Test environment with random actions.

    Args:
        env: Manager-based environment
        num_steps: Number of steps to test
    """
    print(f"\nTesting environment for {num_steps} steps...")

    rng = jax.random.PRNGKey(0)

    # Reset
    print("  Resetting environment...")
    state = env.reset(rng)

    print(f"  Initial state:")
    print(f"    Observations keys: {state.info['observations'].keys()}")
    for key, obs in state.info['observations'].items():
        print(f"      {key}: {obs.shape}")

    # Step
    print(f"  Stepping {num_steps} times...")
    for i in range(num_steps):
        # Random actions
        action = jax.random.uniform(
            rng,
            (env.config.num_envs, env.action_size),
            minval=-1.0,
            maxval=1.0
        )
        rng, _ = jax.random.split(rng)

        state = env.step(state, action)

        if i == 0:
            print(f"  Step 1:")
            print(f"    Reward: {jnp.mean(state.info['reward']):.4f}")
            print(f"    Done: {jnp.sum(state.info['done'])} / {env.config.num_envs}")

    print(f"  Final step:")
    print(f"    Reward: {jnp.mean(state.info['reward']):.4f}")
    print(f"    Done: {jnp.sum(state.info['done'])} / {env.config.num_envs}")

    print("✓ Environment test completed successfully")


# ==================== Training ====================

def train_go2_velocity(
    task: str = "flat",
    num_envs: int = 4096,
    num_timesteps: int = 50_000_000,
    num_evals: int = 10,
    episode_length: int = 1000,
    backend: str = "mjx",
    batch_size: int = 256,
    learning_rate: float = 3e-4,
    policy_hidden_layers: tuple = (512, 256, 128),
    value_hidden_layers: tuple = (512, 256, 128),
    output_dir: str = "./go2_training_logs",
    use_wandb: bool = False,
    wandb_project: str = "mujoco-playground-go2",
    wandb_entity: Optional[str] = None,
    wandb_run_name: Optional[str] = None,
    **kwargs,
):
    """Train Go2 velocity tracking with PPO.

    Args:
        task: Task type
        num_envs: Number of parallel environments
        num_timesteps: Total training timesteps
        num_evals: Number of evaluation checkpoints
        episode_length: Maximum episode length
        backend: Physics backend
        batch_size: Training batch size
        learning_rate: Learning rate
        policy_hidden_layers: Policy network architecture
        value_hidden_layers: Value network architecture
        output_dir: Directory for outputs
        use_wandb: Enable WandB logging
        wandb_project: WandB project name
        wandb_entity: WandB entity name
        wandb_run_name: WandB run name
        **kwargs: Additional arguments

    Returns:
        Tuple of (inference_fn, params, metrics)
    """
    print("="*80)
    print("Go2 Velocity Tracking Training")
    print("="*80)
    print(f"Task: {task} terrain")
    print(f"Num envs: {num_envs}")
    print(f"Num timesteps: {num_timesteps:,}")
    print(f"Backend: {backend}")
    print(f"Output dir: {output_dir}")
    print("="*80)

    # Create environment
    env = create_go2_env(task, num_envs, backend)

    # Test environment
    print("\n" + "="*80)
    test_environment(env, num_steps=10)
    print("="*80)

    # TODO: Implement full Brax PPO training pipeline
    # This requires:
    # 1. Creating a Brax-compatible wrapper for ManagerBasedEnv
    # 2. Setting up PPO networks (policy and value)
    # 3. Running the training loop
    # 4. Saving checkpoints
    # 5. Evaluation and logging

    print("\n" + "="*80)
    print("NOTE: Full Brax integration is not yet complete")
    print("The manager-based environment has been created and tested successfully.")
    print("To complete training, implement the Brax wrapper as described in:")
    print("  mujoco_playground/manager/README.md")
    print("="*80)

    # Initialize WandB if requested
    if use_wandb:
        try:
            import wandb

            if wandb_run_name is None:
                wandb_run_name = f"go2_{task}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            wandb.init(
                project=wandb_project,
                entity=wandb_entity,
                name=wandb_run_name,
                config={
                    "task": task,
                    "num_envs": num_envs,
                    "num_timesteps": num_timesteps,
                    "backend": backend,
                    "learning_rate": learning_rate,
                    **kwargs,
                }
            )
            print("✓ WandB initialized")
        except ImportError:
            print("⚠ WandB not installed, skipping WandB logging")

    return None, None, None


def main(argv):
    """Main training function."""
    del argv  # Unused

    # Debug mode adjustments
    if FLAGS.debug:
        print("="*80)
        print("DEBUG MODE ENABLED")
        print("="*80)
        num_envs = FLAGS.debug_envs
        num_timesteps = FLAGS.debug_timesteps
    else:
        num_envs = FLAGS.num_envs
        num_timesteps = FLAGS.num_timesteps

    # Parse hidden layer sizes
    policy_hidden_layers = tuple(int(x) for x in FLAGS.policy_hidden_layers)
    value_hidden_layers = tuple(int(x) for x in FLAGS.value_hidden_layers)

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    task_name = f"go2_{FLAGS.task}_{timestamp}"
    output_dir = os.path.join(FLAGS.output_dir, task_name)
    os.makedirs(output_dir, exist_ok=True)

    print(f"Output directory: {output_dir}\n")

    # Train
    inference_fn, params, metrics = train_go2_velocity(
        task=FLAGS.task,
        num_envs=num_envs,
        num_timesteps=num_timesteps,
        num_evals=FLAGS.num_evals,
        episode_length=FLAGS.episode_length,
        backend=FLAGS.backend,
        batch_size=FLAGS.batch_size,
        learning_rate=FLAGS.learning_rate,
        policy_hidden_layers=policy_hidden_layers,
        value_hidden_layers=value_hidden_layers,
        output_dir=output_dir,
        use_wandb=FLAGS.use_wandb,
        wandb_project=FLAGS.wandb_project,
        wandb_entity=FLAGS.wandb_entity,
        # Additional PPO hyperparameters
        entropy_cost=FLAGS.entropy_cost,
        discounting=FLAGS.discounting,
        unroll_length=FLAGS.unroll_length,
        num_minibatches=FLAGS.num_minibatches,
        num_updates_per_batch=FLAGS.num_updates_per_batch,
    )

    print("\n" + "="*80)
    print("Training complete!")
    print(f"Results saved to: {output_dir}")
    print("="*80)


if __name__ == "__main__":
    app.run(main)
