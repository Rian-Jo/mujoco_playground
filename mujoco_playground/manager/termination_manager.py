"""Termination manager for checking episode termination conditions."""

from typing import Callable, Dict, Optional, Tuple

import flax.struct
import jax
import jax.numpy as jnp
from ml_collections import config_dict
from mujoco import mjx

from mujoco_playground.manager.manager_base import ManagerBase


@flax.struct.dataclass
class TerminationState:
    """State for termination manager.

    Attributes:
        episode_length: Number of steps in current episode (num_envs,)
        terminated: Whether episode terminated due to failure (num_envs,)
        truncated: Whether episode truncated due to time limit (num_envs,)
    """
    episode_length: jax.Array
    terminated: jax.Array
    truncated: jax.Array


@flax.struct.dataclass
class TerminationTerm:
    """Configuration for a termination condition.

    Attributes:
        func: Function to check termination
            Signature: func(env_state, config, **params) -> jax.Array (bool)
        time_out: If True, counts as truncation not termination
        params: Additional parameters passed to the termination function
    """
    func: Callable
    time_out: bool = False
    params: Optional[Dict[str, any]] = None


@flax.struct.dataclass
class TerminationConfig:
    """Configuration for termination manager.

    Attributes:
        termination_terms: Dict of termination terms {term_name: TerminationTerm}
        episode_length: Maximum episode length before truncation
    """
    termination_terms: Dict[str, TerminationTerm]
    episode_length: int = 1000


class TerminationManager(ManagerBase):
    """Manages episode termination conditions.

    The TerminationManager handles:
    - Checking multiple termination conditions
    - Distinguishing between termination (failure) and truncation (timeout)
    - Tracking episode length
    - Time limit handling

    All operations are JIT-compilable and VMAP-compatible.
    """

    def init_state(
        self,
        rng: jax.Array,
        num_envs: int,
        config: config_dict.ConfigDict,
    ) -> TerminationState:
        """Initialize termination state.

        Args:
            rng: JAX random key (unused for termination manager)
            num_envs: Number of parallel environments
            config: Configuration dict containing termination_terms

        Returns:
            Initial TerminationState with zero lengths and no terminations
        """
        episode_length = jnp.zeros(num_envs, dtype=jnp.int32)
        terminated = jnp.zeros(num_envs, dtype=bool)
        truncated = jnp.zeros(num_envs, dtype=bool)

        return TerminationState(
            episode_length=episode_length,
            terminated=terminated,
            truncated=truncated,
        )

    def reset(
        self,
        state: TerminationState,
        env_ids: jax.Array,
        rng: jax.Array,
        config: config_dict.ConfigDict,
    ) -> TerminationState:
        """Reset termination state for specified environments.

        Args:
            state: Current termination state
            env_ids: Boolean mask of environments to reset
            rng: JAX random key (unused for termination manager)
            config: Configuration dict

        Returns:
            Updated termination state with reset environments
        """
        # Reset episode length
        new_episode_length = jnp.where(
            env_ids,
            jnp.zeros_like(state.episode_length),
            state.episode_length
        )

        # Reset termination flags
        new_terminated = jnp.where(
            env_ids,
            jnp.zeros_like(state.terminated),
            state.terminated
        )

        new_truncated = jnp.where(
            env_ids,
            jnp.zeros_like(state.truncated),
            state.truncated
        )

        return state.replace(
            episode_length=new_episode_length,
            terminated=new_terminated,
            truncated=new_truncated,
        )

    def compute(
        self,
        state: TerminationState,
        env_state: mjx.Data,
        config: config_dict.ConfigDict,
    ) -> Tuple[TerminationState, Dict[str, jax.Array]]:
        """Check all termination conditions.

        Args:
            state: Current termination state
            env_state: Current MJX physics state
            config: Configuration dict with termination_terms

        Returns:
            Tuple of (new_state, termination_info):
            - new_state: Updated termination state
            - termination_info: Dict with keys:
                - done: Combined termination/truncation flag
                - terminated: Termination due to failure
                - truncated: Truncation due to time limit
                - term_<name>: Individual term flags
        """
        termination_terms = config.get("termination_terms", {})
        episode_length_limit = config.get("episode_length", 1000)

        num_envs = env_state.qpos.shape[0]
        terminated = jnp.zeros(num_envs, dtype=bool)
        truncated = jnp.zeros(num_envs, dtype=bool)

        termination_info = {}

        # Increment episode length
        new_episode_length = state.episode_length + 1

        # Check time limit
        time_limit_exceeded = new_episode_length >= episode_length_limit
        truncated = jnp.logical_or(truncated, time_limit_exceeded)
        termination_info["time_limit"] = time_limit_exceeded

        # Check termination terms
        for term_name, term_config in termination_terms.items():
            # Get term parameters
            params = term_config.params or {}

            # Compute term condition
            term_done = term_config.func(
                env_state,
                config,
                **params
            )

            # Add to appropriate category
            if term_config.time_out:
                truncated = jnp.logical_or(truncated, term_done)
            else:
                terminated = jnp.logical_or(terminated, term_done)

            # Store individual term result
            termination_info[f"term_{term_name}"] = term_done

        # Combined done flag
        done = jnp.logical_or(terminated, truncated)

        # Create info dict
        termination_info.update({
            "done": done,
            "terminated": terminated,
            "truncated": truncated,
        })

        # Update state
        new_state = state.replace(
            episode_length=new_episode_length,
            terminated=terminated,
            truncated=truncated,
        )

        return new_state, termination_info


# ==================== Common Termination Functions ====================

def termination_bad_orientation(
    env_state: mjx.Data,
    config: config_dict.ConfigDict,
    threshold: float = 0.5,
    **kwargs,
) -> jax.Array:
    """Terminate if robot orientation is too far from upright.

    Args:
        env_state: MJX state
        config: Configuration dict
        threshold: Minimum z-component of up vector (0.5 = 60 degrees)
        **kwargs: Additional arguments

    Returns:
        Termination flag (num_envs,) bool
    """
    # Get base orientation quaternion
    quat = env_state.qpos[:, 3:7]

    # Convert to up vector z-component
    up_z = 2 * (quat[:, 1] * quat[:, 3] + quat[:, 0] * quat[:, 2])

    # Terminate if too tilted
    bad_orientation = up_z < threshold

    return bad_orientation


def termination_height_limit(
    env_state: mjx.Data,
    config: config_dict.ConfigDict,
    min_height: float = 0.2,
    max_height: float = 2.0,
    **kwargs,
) -> jax.Array:
    """Terminate if robot height is outside valid range.

    Args:
        env_state: MJX state
        config: Configuration dict
        min_height: Minimum valid height in meters
        max_height: Maximum valid height in meters
        **kwargs: Additional arguments

    Returns:
        Termination flag (num_envs,) bool
    """
    # Get base height (z position)
    height = env_state.qpos[:, 2]

    # Terminate if outside valid range
    too_low = height < min_height
    too_high = height > max_height
    invalid_height = jnp.logical_or(too_low, too_high)

    return invalid_height


def termination_joint_limit(
    env_state: mjx.Data,
    config: config_dict.ConfigDict,
    joint_limits: Optional[Tuple[jax.Array, jax.Array]] = None,
    margin: float = 0.05,
    skip_dofs: int = 7,
    **kwargs,
) -> jax.Array:
    """Terminate if any joint exceeds its limits.

    Args:
        env_state: MJX state
        config: Configuration dict
        joint_limits: Optional tuple of (lower, upper) limits
        margin: Safety margin (radians or meters)
        skip_dofs: Number of DOFs to skip (freejoint)
        **kwargs: Additional arguments

    Returns:
        Termination flag (num_envs,) bool
    """
    # Get joint positions
    joint_pos = env_state.qpos[:, skip_dofs:]

    if joint_limits is None:
        # TODO: Extract from MuJoCo model
        # For now, assume no termination
        num_envs = env_state.qpos.shape[0]
        return jnp.zeros(num_envs, dtype=bool)

    lower_limits, upper_limits = joint_limits

    # Check if any joint exceeds limits
    below_lower = jnp.any(joint_pos < (lower_limits + margin), axis=-1)
    above_upper = jnp.any(joint_pos > (upper_limits - margin), axis=-1)

    limit_exceeded = jnp.logical_or(below_lower, above_upper)

    return limit_exceeded


def termination_contact_limit(
    env_state: mjx.Data,
    config: config_dict.ConfigDict,
    contact_geoms: Optional[Tuple[int, ...]] = None,
    force_threshold: float = 1.0,
    **kwargs,
) -> jax.Array:
    """Terminate if specific geoms make contact (e.g., knees, elbows).

    Args:
        env_state: MJX state
        config: Configuration dict
        contact_geoms: Tuple of geom IDs that should not contact
        force_threshold: Minimum contact force to count as contact
        **kwargs: Additional arguments

    Returns:
        Termination flag (num_envs,) bool
    """
    if contact_geoms is None:
        # No contact termination
        num_envs = env_state.qpos.shape[0]
        return jnp.zeros(num_envs, dtype=bool)

    # TODO: Check contact forces for specific geoms
    # This requires access to MJX contact data
    # For now, return no termination
    num_envs = env_state.qpos.shape[0]
    return jnp.zeros(num_envs, dtype=bool)


def termination_velocity_limit(
    env_state: mjx.Data,
    config: config_dict.ConfigDict,
    max_velocity: float = 10.0,
    skip_dofs: int = 6,
    **kwargs,
) -> jax.Array:
    """Terminate if any joint velocity exceeds limit.

    Args:
        env_state: MJX state
        config: Configuration dict
        max_velocity: Maximum allowed joint velocity
        skip_dofs: Number of DOFs to skip (freejoint)
        **kwargs: Additional arguments

    Returns:
        Termination flag (num_envs,) bool
    """
    # Get joint velocities
    joint_vel = env_state.qvel[:, skip_dofs:]

    # Check if any velocity exceeds limit
    velocity_exceeded = jnp.any(jnp.abs(joint_vel) > max_velocity, axis=-1)

    return velocity_exceeded


def termination_nan_check(
    env_state: mjx.Data,
    config: config_dict.ConfigDict,
    **kwargs,
) -> jax.Array:
    """Terminate if any state values are NaN or Inf.

    Args:
        env_state: MJX state
        config: Configuration dict
        **kwargs: Additional arguments

    Returns:
        Termination flag (num_envs,) bool
    """
    # Check qpos for NaN/Inf
    qpos_invalid = jnp.logical_or(
        jnp.any(jnp.isnan(env_state.qpos), axis=-1),
        jnp.any(jnp.isinf(env_state.qpos), axis=-1)
    )

    # Check qvel for NaN/Inf
    qvel_invalid = jnp.logical_or(
        jnp.any(jnp.isnan(env_state.qvel), axis=-1),
        jnp.any(jnp.isinf(env_state.qvel), axis=-1)
    )

    invalid_state = jnp.logical_or(qpos_invalid, qvel_invalid)

    return invalid_state


def create_termination_config(
    termination_terms: Dict[str, TerminationTerm],
    episode_length: int = 1000,
) -> config_dict.ConfigDict:
    """Create a termination configuration dict.

    Args:
        termination_terms: Dict of termination terms
        episode_length: Maximum episode length

    Returns:
        ConfigDict with termination configuration
    """
    return config_dict.create(
        termination_terms=termination_terms,
        episode_length=episode_length,
    )
