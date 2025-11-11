"""Command manager for generating and tracking command signals."""

from typing import Callable, Dict, Optional, Tuple

import flax.struct
import jax
import jax.numpy as jnp
from ml_collections import config_dict
from mujoco import mjx

from mujoco_playground.manager.manager_base import ManagerBase


@flax.struct.dataclass
class CommandState:
    """State for command manager.

    Attributes:
        current_command: Current command signal (num_envs, command_dim)
        command_counter: Steps since last resample (num_envs,)
        resample_time: Next time to resample commands (num_envs,)
    """
    current_command: jax.Array
    command_counter: jax.Array
    resample_time: jax.Array


@flax.struct.dataclass
class CommandConfig:
    """Configuration for command manager.

    Attributes:
        command_func: Function to generate commands
            Signature: func(rng, config) -> jax.Array (num_envs, command_dim)
        command_dim: Dimension of command signal
        resample_interval: Steps between command resampling
        resample_on_reset: If True, resample commands on environment reset
        command_ranges: Optional dict of {command_name: (min, max)} for clipping
    """
    command_func: Callable
    command_dim: int = 3
    resample_interval: int = 500
    resample_on_reset: bool = True
    command_ranges: Optional[Dict[str, Tuple[float, float]]] = None


class CommandManager(ManagerBase):
    """Manages command generation and tracking.

    The CommandManager handles:
    - Generating command signals (velocity targets, poses, etc.)
    - Automatic resampling at intervals
    - Per-environment command tracking
    - Curriculum-aware command generation

    All operations are JIT-compilable and VMAP-compatible.
    """

    def init_state(
        self,
        rng: jax.Array,
        num_envs: int,
        config: config_dict.ConfigDict,
    ) -> CommandState:
        """Initialize command state.

        Args:
            rng: JAX random key for initial command generation
            num_envs: Number of parallel environments
            config: Configuration dict with command_func

        Returns:
            Initial CommandState with random commands
        """
        command_func = config.get("command_func")
        if command_func is None:
            # No command generation - use zero commands
            command_dim = config.get("command_dim", 3)
            current_command = jnp.zeros((num_envs, command_dim))
        else:
            # Generate initial commands
            current_command = command_func(rng, config, num_envs)

        command_counter = jnp.zeros(num_envs, dtype=jnp.int32)
        resample_interval = config.get("resample_interval", 500)
        resample_time = jnp.ones(num_envs, dtype=jnp.int32) * resample_interval

        return CommandState(
            current_command=current_command,
            command_counter=command_counter,
            resample_time=resample_time,
        )

    def reset(
        self,
        state: CommandState,
        env_ids: jax.Array,
        rng: jax.Array,
        config: config_dict.ConfigDict,
    ) -> CommandState:
        """Reset command state for specified environments.

        Args:
            state: Current command state
            env_ids: Boolean mask of environments to reset
            rng: JAX random key for command generation
            config: Configuration dict

        Returns:
            Updated command state with reset environments
        """
        # Optionally resample commands on reset
        resample_on_reset = config.get("resample_on_reset", True)

        if resample_on_reset:
            command_func = config.get("command_func")
            if command_func is not None:
                # Generate new commands
                num_envs = state.current_command.shape[0]
                new_commands = command_func(rng, config, num_envs)

                # Update only reset environments
                new_current_command = jnp.where(
                    env_ids[:, None],
                    new_commands,
                    state.current_command
                )
            else:
                new_current_command = state.current_command
        else:
            new_current_command = state.current_command

        # Reset counter for reset environments
        new_command_counter = jnp.where(
            env_ids,
            jnp.zeros_like(state.command_counter),
            state.command_counter
        )

        return state.replace(
            current_command=new_current_command,
            command_counter=new_command_counter,
        )

    def compute(
        self,
        state: CommandState,
        env_state: mjx.Data,
        config: config_dict.ConfigDict,
        rng: jax.Array,
    ) -> Tuple[CommandState, jax.Array]:
        """Update commands, resampling if needed.

        Args:
            state: Current command state
            env_state: Current MJX physics state
            config: Configuration dict
            rng: JAX random key for command generation

        Returns:
            Tuple of (new_state, current_command):
            - new_state: Updated command state
            - current_command: Command signal (num_envs, command_dim)
        """
        command_func = config.get("command_func")
        resample_interval = config.get("resample_interval", 500)

        # Check if resample needed
        should_resample = state.command_counter >= resample_interval

        if command_func is not None:
            # Generate new commands
            num_envs = state.current_command.shape[0]
            new_commands = command_func(rng, config, num_envs)

            # Conditionally update based on resample flag
            current_command = jnp.where(
                should_resample[:, None],
                new_commands,
                state.current_command
            )
        else:
            current_command = state.current_command

        # Update counter (reset when resampled)
        command_counter = jnp.where(
            should_resample,
            jnp.zeros_like(state.command_counter),
            state.command_counter + 1
        )

        new_state = state.replace(
            current_command=current_command,
            command_counter=command_counter,
        )

        return new_state, current_command


# ==================== Common Command Generation Functions ====================

def command_velocity_tracking(
    rng: jax.Array,
    config: config_dict.ConfigDict,
    num_envs: int,
) -> jax.Array:
    """Generate random velocity tracking commands.

    Generates commands for (vx, vy, wz) - linear x, linear y, angular z velocities.

    Args:
        rng: JAX random key
        config: Configuration dict with command_ranges
        num_envs: Number of environments

    Returns:
        Velocity commands (num_envs, 3)
    """
    command_ranges = config.get("command_ranges", {})

    # Default ranges
    vx_range = command_ranges.get("lin_vel_x", (-1.0, 1.0))
    vy_range = command_ranges.get("lin_vel_y", (-0.5, 0.5))
    wz_range = command_ranges.get("ang_vel_z", (-1.0, 1.0))

    # Generate random commands
    rng, vx_key, vy_key, wz_key = jax.random.split(rng, 4)

    vx = jax.random.uniform(vx_key, (num_envs,), minval=vx_range[0], maxval=vx_range[1])
    vy = jax.random.uniform(vy_key, (num_envs,), minval=vy_range[0], maxval=vy_range[1])
    wz = jax.random.uniform(wz_key, (num_envs,), minval=wz_range[0], maxval=wz_range[1])

    commands = jnp.stack([vx, vy, wz], axis=-1)

    return commands


def command_position_tracking(
    rng: jax.Array,
    config: config_dict.ConfigDict,
    num_envs: int,
) -> jax.Array:
    """Generate random position tracking commands.

    Generates target positions in (x, y, z).

    Args:
        rng: JAX random key
        config: Configuration dict with command_ranges
        num_envs: Number of environments

    Returns:
        Position commands (num_envs, 3)
    """
    command_ranges = config.get("command_ranges", {})

    # Default ranges
    x_range = command_ranges.get("pos_x", (-2.0, 2.0))
    y_range = command_ranges.get("pos_y", (-2.0, 2.0))
    z_range = command_ranges.get("pos_z", (0.5, 1.5))

    # Generate random commands
    rng, x_key, y_key, z_key = jax.random.split(rng, 4)

    x = jax.random.uniform(x_key, (num_envs,), minval=x_range[0], maxval=x_range[1])
    y = jax.random.uniform(y_key, (num_envs,), minval=y_range[0], maxval=y_range[1])
    z = jax.random.uniform(z_key, (num_envs,), minval=z_range[0], maxval=z_range[1])

    commands = jnp.stack([x, y, z], axis=-1)

    return commands


def command_heading_tracking(
    rng: jax.Array,
    config: config_dict.ConfigDict,
    num_envs: int,
) -> jax.Array:
    """Generate random heading tracking commands.

    Generates target heading angles.

    Args:
        rng: JAX random key
        config: Configuration dict with command_ranges
        num_envs: Number of environments

    Returns:
        Heading commands (num_envs, 1)
    """
    command_ranges = config.get("command_ranges", {})

    # Default range: [-pi, pi]
    heading_range = command_ranges.get("heading", (-jnp.pi, jnp.pi))

    # Generate random headings
    heading = jax.random.uniform(
        rng,
        (num_envs, 1),
        minval=heading_range[0],
        maxval=heading_range[1]
    )

    return heading


def command_zero(
    rng: jax.Array,
    config: config_dict.ConfigDict,
    num_envs: int,
) -> jax.Array:
    """Generate zero commands (for testing or simple tasks).

    Args:
        rng: JAX random key (unused)
        config: Configuration dict
        num_envs: Number of environments

    Returns:
        Zero commands (num_envs, command_dim)
    """
    command_dim = config.get("command_dim", 3)
    return jnp.zeros((num_envs, command_dim))


def create_command_config(
    command_func: Optional[Callable] = None,
    command_dim: int = 3,
    resample_interval: int = 500,
    resample_on_reset: bool = True,
    command_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
) -> config_dict.ConfigDict:
    """Create a command configuration dict.

    Args:
        command_func: Function to generate commands
        command_dim: Dimension of command signal
        resample_interval: Steps between resampling
        resample_on_reset: If True, resample on reset
        command_ranges: Optional command ranges

    Returns:
        ConfigDict with command configuration
    """
    return config_dict.create(
        command_func=command_func,
        command_dim=command_dim,
        resample_interval=resample_interval,
        resample_on_reset=resample_on_reset,
        command_ranges=command_ranges or {},
    )
