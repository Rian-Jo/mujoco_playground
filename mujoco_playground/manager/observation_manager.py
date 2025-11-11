"""Observation manager for collecting and processing observations."""

from typing import Callable, Dict, Optional, Tuple

import flax.struct
import jax
import jax.numpy as jnp
from ml_collections import config_dict
from mujoco import mjx

from mujoco_playground.manager.manager_base import (
    ManagerBase,
    init_circular_buffer,
    update_circular_buffer,
    flatten_history_buffer,
)


@flax.struct.dataclass
class ObservationState:
    """State for observation manager.

    Attributes:
        history_buffers: Dict of history buffers for each observation term
            Keys are term names, values are arrays of shape (num_envs, obs_dim, history_len)
        step_counter: Step counter for each environment (num_envs,)
    """
    history_buffers: Dict[str, jax.Array]
    step_counter: jax.Array


@flax.struct.dataclass
class ObservationTerm:
    """Configuration for a single observation term.

    Attributes:
        func: Function to extract observation from env_state
            Signature: func(env_state, config) -> jax.Array
        noise_config: Optional noise configuration dict with keys:
            - std: Standard deviation of Gaussian noise
            - type: "gaussian" or "uniform" (default: "gaussian")
        clip_range: Optional (min, max) tuple for clipping
        scale: Scaling factor applied after clipping
        history_len: Number of timesteps to keep in history
        flatten_history: If True, flatten history into single vector
        group: Group name for this observation ("policy", "value", etc.)
    """
    func: Callable
    noise_config: Optional[Dict[str, float]] = None
    clip_range: Optional[Tuple[float, float]] = None
    scale: float = 1.0
    history_len: int = 1
    flatten_history: bool = True
    group: str = "policy"


@flax.struct.dataclass
class ObservationConfig:
    """Configuration for observation manager.

    Attributes:
        observation_terms: Dict of observation terms grouped by type
            Structure: {group_name: {term_name: ObservationTerm}}
            Example: {"policy": {"joint_pos": ObservationTerm(...)}}
        concatenate_groups: If True, concatenate all terms in each group
        add_noise: If False, disable noise injection globally
    """
    observation_terms: Dict[str, Dict[str, ObservationTerm]]
    concatenate_groups: bool = True
    add_noise: bool = True


class ObservationManager(ManagerBase):
    """Manages observation collection and processing.

    The ObservationManager handles:
    - Extracting observations from MJX state via term functions
    - Applying noise, clipping, and scaling
    - Managing observation history buffers
    - Grouping observations (policy, value, privileged)
    - Flattening and concatenating observations

    All operations are JIT-compilable and VMAP-compatible.
    """

    def init_state(
        self,
        rng: jax.Array,
        num_envs: int,
        config: config_dict.ConfigDict,
    ) -> ObservationState:
        """Initialize observation state.

        Args:
            rng: JAX random key (unused for observation manager)
            num_envs: Number of parallel environments
            config: Configuration dict containing observation_terms

        Returns:
            Initial ObservationState with zero buffers
        """
        history_buffers = {}

        observation_terms = config.get("observation_terms", {})

        for group_name, group_terms in observation_terms.items():
            for term_name, term_config in group_terms.items():
                if term_config.history_len > 1:
                    # Create history buffer for this term
                    # We'll initialize with proper size after first compute
                    # For now, use placeholder
                    buffer_key = f"{group_name}.{term_name}"
                    history_buffers[buffer_key] = None

        step_counter = jnp.zeros(num_envs, dtype=jnp.int32)

        return ObservationState(
            history_buffers=history_buffers,
            step_counter=step_counter,
        )

    def reset(
        self,
        state: ObservationState,
        env_ids: jax.Array,
        rng: jax.Array,
        config: config_dict.ConfigDict,
    ) -> ObservationState:
        """Reset observation state for specified environments.

        Args:
            state: Current observation state
            env_ids: Boolean mask of environments to reset
            rng: JAX random key (unused for observation manager)
            config: Configuration dict

        Returns:
            Updated observation state with reset environments
        """
        # Reset step counter
        new_step_counter = jnp.where(
            env_ids,
            jnp.zeros_like(state.step_counter),
            state.step_counter
        )

        # Reset history buffers
        new_history_buffers = {}
        for key, buffer in state.history_buffers.items():
            if buffer is not None:
                zero_buffer = jnp.zeros_like(buffer)
                new_buffer = jnp.where(
                    env_ids[:, None, None],
                    zero_buffer,
                    buffer
                )
                new_history_buffers[key] = new_buffer
            else:
                new_history_buffers[key] = None

        return state.replace(
            history_buffers=new_history_buffers,
            step_counter=new_step_counter,
        )

    def compute(
        self,
        state: ObservationState,
        env_state: mjx.Data,
        config: config_dict.ConfigDict,
        rng: jax.Array,
        action_state: Optional[any] = None,
        command_state: Optional[any] = None,
    ) -> Tuple[ObservationState, Dict[str, jax.Array]]:
        """Compute all observation groups.

        Args:
            state: Current observation state
            env_state: Current MJX physics state
            config: Configuration dict with observation_terms
            rng: JAX random key for noise injection
            action_state: Optional action state for action-based observations
            command_state: Optional command state for command-based observations

        Returns:
            Tuple of (new_state, observations):
            - new_state: Updated observation state
            - observations: Dict of {group_name: observation_array}
        """
        observation_terms = config.get("observation_terms", {})
        add_noise = config.get("add_noise", True)
        concatenate_groups = config.get("concatenate_groups", True)

        observations = {}
        new_history_buffers = dict(state.history_buffers)

        for group_name, group_terms in observation_terms.items():
            group_observations = [] if concatenate_groups else {}

            for term_name, term_config in group_terms.items():
                # 1. Extract raw observation
                raw_obs = term_config.func(
                    env_state,
                    config,
                    action_state=action_state,
                    command_state=command_state,
                )

                # 2. Apply noise
                if add_noise and term_config.noise_config:
                    rng, noise_key = jax.random.split(rng)
                    raw_obs = self._add_noise(
                        raw_obs, noise_key, term_config.noise_config
                    )

                # 3. Clip
                if term_config.clip_range:
                    raw_obs = jnp.clip(
                        raw_obs,
                        term_config.clip_range[0],
                        term_config.clip_range[1]
                    )

                # 4. Scale
                scaled_obs = raw_obs * term_config.scale

                # 5. Handle history
                if term_config.history_len > 1:
                    buffer_key = f"{group_name}.{term_name}"

                    # Initialize buffer if needed
                    if new_history_buffers[buffer_key] is None:
                        num_envs = scaled_obs.shape[0]
                        obs_dim = scaled_obs.shape[1] if len(scaled_obs.shape) > 1 else 1
                        new_history_buffers[buffer_key] = jnp.zeros(
                            (num_envs, obs_dim, term_config.history_len)
                        )

                    # Update buffer
                    current_buffer = new_history_buffers[buffer_key]
                    updated_buffer = update_circular_buffer(
                        current_buffer, scaled_obs
                    )
                    new_history_buffers[buffer_key] = updated_buffer

                    # Get observation from buffer
                    if term_config.flatten_history:
                        term_obs = flatten_history_buffer(updated_buffer)
                    else:
                        term_obs = updated_buffer
                else:
                    term_obs = scaled_obs

                # 6. Add to group
                if concatenate_groups:
                    group_observations.append(term_obs)
                else:
                    group_observations[term_name] = term_obs

            # 7. Concatenate or package group observations
            if concatenate_groups:
                observations[group_name] = jnp.concatenate(
                    group_observations, axis=-1
                )
            else:
                observations[group_name] = group_observations

        # Update step counter
        new_step_counter = state.step_counter + 1

        new_state = state.replace(
            history_buffers=new_history_buffers,
            step_counter=new_step_counter,
        )

        return new_state, observations

    def _add_noise(
        self,
        obs: jax.Array,
        rng: jax.Array,
        noise_config: Dict[str, float],
    ) -> jax.Array:
        """Add noise to observations.

        Args:
            obs: Observation array
            rng: JAX random key
            noise_config: Noise configuration dict

        Returns:
            Noisy observation
        """
        noise_type = noise_config.get("type", "gaussian")
        noise_std = noise_config.get("std", 0.01)

        if noise_type == "gaussian":
            noise = jax.random.normal(rng, obs.shape) * noise_std
        elif noise_type == "uniform":
            noise = jax.random.uniform(rng, obs.shape, minval=-noise_std, maxval=noise_std)
        else:
            raise ValueError(f"Unknown noise type: {noise_type}")

        return obs + noise

    def get_observation_size(
        self,
        config: config_dict.ConfigDict,
        group: str = "policy",
    ) -> int:
        """Compute the size of observations for a given group.

        This is useful for creating network architectures.

        Args:
            config: Configuration dict with observation_terms
            group: Group name to compute size for

        Returns:
            Total observation size for the group
        """
        observation_terms = config.get("observation_terms", {})
        group_terms = observation_terms.get(group, {})

        total_size = 0
        for term_name, term_config in group_terms.items():
            # We need to actually call the function to get the size
            # This is a placeholder - in practice, you'd specify sizes in config
            # or call the function once with a dummy state
            term_size = term_config.get("size", 0)

            # Account for history
            if term_config.history_len > 1 and term_config.flatten_history:
                term_size *= term_config.history_len

            total_size += term_size

        return total_size


# ==================== Common Observation Functions ====================

def get_joint_positions(
    env_state: mjx.Data,
    config: config_dict.ConfigDict,
    **kwargs,
) -> jax.Array:
    """Extract joint positions from environment state.

    Args:
        env_state: MJX state
        config: Configuration dict
        **kwargs: Additional arguments (unused)

    Returns:
        Joint positions (num_envs, n_joints)
    """
    # Skip freejoint if present (first 7 DOFs)
    skip_dofs = config.get("skip_freejoint_dofs", 7)
    return env_state.qpos[:, skip_dofs:]


def get_joint_velocities(
    env_state: mjx.Data,
    config: config_dict.ConfigDict,
    **kwargs,
) -> jax.Array:
    """Extract joint velocities from environment state.

    Args:
        env_state: MJX state
        config: Configuration dict
        **kwargs: Additional arguments (unused)

    Returns:
        Joint velocities (num_envs, n_joints)
    """
    skip_dofs = config.get("skip_freejoint_dofs", 6)
    return env_state.qvel[:, skip_dofs:]


def get_base_orientation(
    env_state: mjx.Data,
    config: config_dict.ConfigDict,
    **kwargs,
) -> jax.Array:
    """Extract base orientation (quaternion) from environment state.

    Args:
        env_state: MJX state
        config: Configuration dict
        **kwargs: Additional arguments (unused)

    Returns:
        Base quaternion (num_envs, 4)
    """
    return env_state.qpos[:, 3:7]


def get_base_linear_velocity(
    env_state: mjx.Data,
    config: config_dict.ConfigDict,
    **kwargs,
) -> jax.Array:
    """Extract base linear velocity from environment state.

    Args:
        env_state: MJX state
        config: Configuration dict
        **kwargs: Additional arguments (unused)

    Returns:
        Base linear velocity (num_envs, 3)
    """
    return env_state.qvel[:, :3]


def get_base_angular_velocity(
    env_state: mjx.Data,
    config: config_dict.ConfigDict,
    **kwargs,
) -> jax.Array:
    """Extract base angular velocity from environment state.

    Args:
        env_state: MJX state
        config: Configuration dict
        **kwargs: Additional arguments (unused)

    Returns:
        Base angular velocity (num_envs, 3)
    """
    return env_state.qvel[:, 3:6]


def get_last_action(
    env_state: mjx.Data,
    config: config_dict.ConfigDict,
    action_state=None,
    **kwargs,
) -> jax.Array:
    """Extract last action from action state.

    Args:
        env_state: MJX state (unused)
        config: Configuration dict
        action_state: Action manager state
        **kwargs: Additional arguments (unused)

    Returns:
        Last action (num_envs, action_dim)
    """
    if action_state is None:
        # Return zeros if no action state available
        num_envs = env_state.qpos.shape[0]
        action_dim = config.get("action_dim", 12)
        return jnp.zeros((num_envs, action_dim))

    return action_state.last_action


def get_current_command(
    env_state: mjx.Data,
    config: config_dict.ConfigDict,
    command_state=None,
    **kwargs,
) -> jax.Array:
    """Extract current command from command state.

    Args:
        env_state: MJX state (unused)
        config: Configuration dict
        command_state: Command manager state
        **kwargs: Additional arguments (unused)

    Returns:
        Current command (num_envs, command_dim)
    """
    if command_state is None:
        # Return zeros if no command state available
        num_envs = env_state.qpos.shape[0]
        command_dim = config.get("command_dim", 3)
        return jnp.zeros((num_envs, command_dim))

    return command_state.current_command


def create_observation_config(
    observation_terms: Dict[str, Dict[str, ObservationTerm]],
    concatenate_groups: bool = True,
    add_noise: bool = True,
) -> config_dict.ConfigDict:
    """Create an observation configuration dict.

    Args:
        observation_terms: Dict of observation terms grouped by type
        concatenate_groups: If True, concatenate all terms in each group
        add_noise: If False, disable noise injection globally

    Returns:
        ConfigDict with observation configuration
    """
    return config_dict.create(
        observation_terms=observation_terms,
        concatenate_groups=concatenate_groups,
        add_noise=add_noise,
    )
