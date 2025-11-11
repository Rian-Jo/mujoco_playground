"""Reward manager for computing modular reward terms."""

from typing import Callable, Dict, Optional, Tuple

import flax.struct
import jax
import jax.numpy as jnp
from ml_collections import config_dict
from mujoco import mjx

from mujoco_playground.manager.manager_base import ManagerBase


@flax.struct.dataclass
class RewardState:
    """State for reward manager.

    Attributes:
        episode_rewards: Cumulative rewards for current episode per term
            Shape: (num_envs, num_terms)
        step_rewards: Rewards from last step per term
            Shape: (num_envs, num_terms)
        total_episode_reward: Total cumulative reward for current episode
            Shape: (num_envs,)
    """
    episode_rewards: jax.Array
    step_rewards: jax.Array
    total_episode_reward: jax.Array


@flax.struct.dataclass
class RewardTerm:
    """Configuration for a single reward term.

    Attributes:
        func: Function to compute reward
            Signature: func(env_state, action, config, **params) -> jax.Array
        weight: Weight multiplier for this reward term
        params: Additional parameters passed to the reward function
    """
    func: Callable
    weight: float = 1.0
    params: Optional[Dict[str, any]] = None


@flax.struct.dataclass
class RewardConfig:
    """Configuration for reward manager.

    Attributes:
        reward_terms: Dict of reward terms {term_name: RewardTerm}
        normalize_rewards: If True, normalize rewards to [-1, 1] range
        clip_rewards: Optional (min, max) tuple for clipping total reward
    """
    reward_terms: Dict[str, RewardTerm]
    normalize_rewards: bool = False
    clip_rewards: Optional[Tuple[float, float]] = None


class RewardManager(ManagerBase):
    """Manages modular reward computation.

    The RewardManager handles:
    - Computing individual reward terms
    - Weighting and combining terms
    - Tracking per-term and total rewards
    - Episodic reward accumulation
    - Optional normalization and clipping

    All operations are JIT-compilable and VMAP-compatible.
    """

    def init_state(
        self,
        rng: jax.Array,
        num_envs: int,
        config: config_dict.ConfigDict,
    ) -> RewardState:
        """Initialize reward state.

        Args:
            rng: JAX random key (unused for reward manager)
            num_envs: Number of parallel environments
            config: Configuration dict containing reward_terms

        Returns:
            Initial RewardState with zero rewards
        """
        reward_terms = config.get("reward_terms", {})
        num_terms = len(reward_terms)

        episode_rewards = jnp.zeros((num_envs, num_terms))
        step_rewards = jnp.zeros((num_envs, num_terms))
        total_episode_reward = jnp.zeros(num_envs)

        return RewardState(
            episode_rewards=episode_rewards,
            step_rewards=step_rewards,
            total_episode_reward=total_episode_reward,
        )

    def reset(
        self,
        state: RewardState,
        env_ids: jax.Array,
        rng: jax.Array,
        config: config_dict.ConfigDict,
    ) -> RewardState:
        """Reset reward state for specified environments.

        Args:
            state: Current reward state
            env_ids: Boolean mask of environments to reset
            rng: JAX random key (unused for reward manager)
            config: Configuration dict

        Returns:
            Updated reward state with reset environments
        """
        # Reset episode rewards
        zero_rewards = jnp.zeros_like(state.episode_rewards)
        new_episode_rewards = jnp.where(
            env_ids[:, None],
            zero_rewards,
            state.episode_rewards
        )

        # Reset total episode reward
        zero_total = jnp.zeros_like(state.total_episode_reward)
        new_total_episode_reward = jnp.where(
            env_ids,
            zero_total,
            state.total_episode_reward
        )

        # Note: step_rewards are not reset as they represent the last step

        return state.replace(
            episode_rewards=new_episode_rewards,
            total_episode_reward=new_total_episode_reward,
        )

    def compute(
        self,
        state: RewardState,
        env_state: mjx.Data,
        config: config_dict.ConfigDict,
        action: jax.Array,
        action_state: Optional[any] = None,
        command_state: Optional[any] = None,
    ) -> Tuple[RewardState, Tuple[jax.Array, Dict[str, jax.Array]]]:
        """Compute weighted sum of all reward terms.

        Args:
            state: Current reward state
            env_state: Current MJX physics state
            config: Configuration dict with reward_terms
            action: Current action (num_envs, action_dim)
            action_state: Optional action state for action-based rewards
            command_state: Optional command state for command tracking rewards

        Returns:
            Tuple of (new_state, (total_reward, term_rewards)):
            - new_state: Updated reward state
            - total_reward: Weighted sum of all terms (num_envs,)
            - term_rewards: Dict of individual term rewards {name: array}
        """
        reward_terms = config.get("reward_terms", {})
        dt = config.get("ctrl_dt", 0.02)

        num_envs = env_state.qpos.shape[0]
        total_reward = jnp.zeros(num_envs)
        term_rewards_dict = {}
        term_rewards_array = []

        # Compute each reward term
        for term_idx, (term_name, term_config) in enumerate(reward_terms.items()):
            # Get term parameters
            params = term_config.params or {}

            # Compute term reward
            term_reward = term_config.func(
                env_state,
                action,
                config,
                action_state=action_state,
                command_state=command_state,
                **params
            )

            # Apply weight and dt
            weighted_reward = term_reward * term_config.weight * dt

            # Accumulate
            total_reward += weighted_reward
            term_rewards_dict[term_name] = term_reward
            term_rewards_array.append(term_reward)

        # Stack term rewards for state tracking
        step_rewards_array = jnp.stack(term_rewards_array, axis=-1)

        # Optionally clip total reward
        if config.get("clip_rewards"):
            clip_min, clip_max = config.clip_rewards
            total_reward = jnp.clip(total_reward, clip_min, clip_max)

        # Optionally normalize rewards
        if config.get("normalize_rewards", False):
            # Simple normalization: tanh squashing
            total_reward = jnp.tanh(total_reward)

        # Update state
        new_episode_rewards = state.episode_rewards + step_rewards_array
        new_total_episode_reward = state.total_episode_reward + total_reward

        new_state = state.replace(
            episode_rewards=new_episode_rewards,
            step_rewards=step_rewards_array,
            total_episode_reward=new_total_episode_reward,
        )

        return new_state, (total_reward, term_rewards_dict)

    def get_episodic_rewards(
        self,
        state: RewardState,
        config: config_dict.ConfigDict,
    ) -> Dict[str, jax.Array]:
        """Get episodic rewards for logging.

        Args:
            state: Current reward state
            config: Configuration dict

        Returns:
            Dict of episodic rewards per term
        """
        reward_terms = config.get("reward_terms", {})
        episodic_rewards = {}

        for term_idx, term_name in enumerate(reward_terms.keys()):
            episodic_rewards[term_name] = state.episode_rewards[:, term_idx]

        episodic_rewards["total"] = state.total_episode_reward

        return episodic_rewards


# ==================== Common Reward Functions ====================

def reward_alive(
    env_state: mjx.Data,
    action: jax.Array,
    config: config_dict.ConfigDict,
    **kwargs,
) -> jax.Array:
    """Reward for staying alive.

    Args:
        env_state: MJX state
        action: Current action
        config: Configuration dict
        **kwargs: Additional arguments

    Returns:
        Alive reward (num_envs,)
    """
    num_envs = env_state.qpos.shape[0]
    return jnp.ones(num_envs)


def reward_energy_penalty(
    env_state: mjx.Data,
    action: jax.Array,
    config: config_dict.ConfigDict,
    action_state=None,
    **kwargs,
) -> jax.Array:
    """Penalty for energy consumption (action magnitude).

    Args:
        env_state: MJX state
        action: Current action
        config: Configuration dict
        action_state: Action state (unused)
        **kwargs: Additional arguments

    Returns:
        Energy penalty (num_envs,) - negative values
    """
    # L2 norm of actions squared
    energy = jnp.sum(action ** 2, axis=-1)
    return -energy


def reward_velocity_tracking(
    env_state: mjx.Data,
    action: jax.Array,
    config: config_dict.ConfigDict,
    command_state=None,
    velocity_indices: Tuple[int, ...] = (0, 1, 5),
    **kwargs,
) -> jax.Array:
    """Reward for tracking velocity commands.

    Args:
        env_state: MJX state
        action: Current action
        config: Configuration dict
        command_state: Command state with target velocities
        velocity_indices: Indices of velocity components to track
            Default: (0, 1, 5) for (vx, vy, wz)
        **kwargs: Additional arguments

    Returns:
        Velocity tracking reward (num_envs,)
    """
    if command_state is None:
        num_envs = env_state.qpos.shape[0]
        return jnp.zeros(num_envs)

    # Get current velocities
    current_vel = env_state.qvel[:, list(velocity_indices)]

    # Get command velocities
    command_vel = command_state.current_command

    # Compute tracking error
    error = jnp.sum((current_vel - command_vel) ** 2, axis=-1)

    # Reward is negative exponential of error
    reward = jnp.exp(-error)

    return reward


def reward_orientation_penalty(
    env_state: mjx.Data,
    action: jax.Array,
    config: config_dict.ConfigDict,
    **kwargs,
) -> jax.Array:
    """Penalty for deviating from upright orientation.

    Args:
        env_state: MJX state
        action: Current action
        config: Configuration dict
        **kwargs: Additional arguments

    Returns:
        Orientation penalty (num_envs,) - negative for bad orientation
    """
    # Get base orientation quaternion
    quat = env_state.qpos[:, 3:7]

    # Convert to up vector (z-axis in world frame)
    # For quaternion [w, x, y, z], up vector z-component is:
    # 2(xz + wy)
    up_z = 2 * (quat[:, 1] * quat[:, 3] + quat[:, 0] * quat[:, 2])

    # Reward for being upright (up_z close to 1)
    # Penalty for being tilted (up_z far from 1)
    orientation_reward = up_z

    return orientation_reward


def reward_smoothness(
    env_state: mjx.Data,
    action: jax.Array,
    config: config_dict.ConfigDict,
    action_state=None,
    **kwargs,
) -> jax.Array:
    """Reward for smooth actions (penalize action changes).

    Args:
        env_state: MJX state
        action: Current action
        config: Configuration dict
        action_state: Action state with last action
        **kwargs: Additional arguments

    Returns:
        Smoothness reward (num_envs,) - negative for jerky motion
    """
    if action_state is None:
        num_envs = env_state.qpos.shape[0]
        return jnp.zeros(num_envs)

    # Compute action difference
    action_diff = action - action_state.last_action

    # Penalty for large differences
    smoothness_penalty = -jnp.sum(action_diff ** 2, axis=-1)

    return smoothness_penalty


def reward_joint_velocity_penalty(
    env_state: mjx.Data,
    action: jax.Array,
    config: config_dict.ConfigDict,
    skip_dofs: int = 6,
    **kwargs,
) -> jax.Array:
    """Penalty for excessive joint velocities.

    Args:
        env_state: MJX state
        action: Current action
        config: Configuration dict
        skip_dofs: Number of DOFs to skip (freejoint)
        **kwargs: Additional arguments

    Returns:
        Joint velocity penalty (num_envs,) - negative values
    """
    # Get joint velocities (skip freejoint)
    joint_vel = env_state.qvel[:, skip_dofs:]

    # L2 penalty
    penalty = -jnp.sum(joint_vel ** 2, axis=-1)

    return penalty


def reward_joint_acceleration_penalty(
    env_state: mjx.Data,
    action: jax.Array,
    config: config_dict.ConfigDict,
    skip_dofs: int = 6,
    **kwargs,
) -> jax.Array:
    """Penalty for excessive joint accelerations.

    Args:
        env_state: MJX state
        action: Current action
        config: Configuration dict
        skip_dofs: Number of DOFs to skip (freejoint)
        **kwargs: Additional arguments

    Returns:
        Joint acceleration penalty (num_envs,) - negative values
    """
    # Get joint accelerations (skip freejoint)
    joint_acc = env_state.qacc[:, skip_dofs:]

    # L2 penalty
    penalty = -jnp.sum(joint_acc ** 2, axis=-1)

    return penalty


def reward_height_penalty(
    env_state: mjx.Data,
    action: jax.Array,
    config: config_dict.ConfigDict,
    target_height: float = 0.5,
    **kwargs,
) -> jax.Array:
    """Penalty for deviating from target height.

    Args:
        env_state: MJX state
        action: Current action
        config: Configuration dict
        target_height: Target height in meters
        **kwargs: Additional arguments

    Returns:
        Height penalty (num_envs,) - negative for deviation
    """
    # Get base height (z position)
    height = env_state.qpos[:, 2]

    # Penalty for deviation from target
    height_error = (height - target_height) ** 2
    penalty = -height_error

    return penalty


def create_reward_config(
    reward_terms: Dict[str, RewardTerm],
    normalize_rewards: bool = False,
    clip_rewards: Optional[Tuple[float, float]] = None,
) -> config_dict.ConfigDict:
    """Create a reward configuration dict.

    Args:
        reward_terms: Dict of reward terms
        normalize_rewards: If True, normalize rewards
        clip_rewards: Optional (min, max) for clipping

    Returns:
        ConfigDict with reward configuration
    """
    return config_dict.create(
        reward_terms=reward_terms,
        normalize_rewards=normalize_rewards,
        clip_rewards=clip_rewards,
    )
