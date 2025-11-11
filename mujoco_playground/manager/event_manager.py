"""Event manager for handling environment events and modifications."""

from typing import Callable, Dict, Optional, Tuple

import flax.struct
import jax
import jax.numpy as jnp
from ml_collections import config_dict
from mujoco import mjx

from mujoco_playground.manager.manager_base import ManagerBase


@flax.struct.dataclass
class EventState:
    """State for event manager.

    Attributes:
        event_counters: Counter for each event (num_events, num_envs)
        last_trigger_step: Last step when each event triggered (num_events, num_envs)
    """
    event_counters: jax.Array
    last_trigger_step: jax.Array


@flax.struct.dataclass
class EventTerm:
    """Configuration for an event.

    Attributes:
        func: Event handler function
            Signature: func(env_state, rng, config, **params) -> env_state
        mode: Trigger mode ("interval", "reset", "condition")
        interval: Steps between triggers (for "interval" mode)
        condition_func: Condition check function (for "condition" mode)
            Signature: func(env_state, config) -> bool array
        params: Additional parameters passed to the event function
    """
    func: Callable
    mode: str = "interval"
    interval: int = 1000
    condition_func: Optional[Callable] = None
    params: Optional[Dict[str, any]] = None


@flax.struct.dataclass
class EventConfig:
    """Configuration for event manager.

    Attributes:
        event_terms: Dict of event terms {event_name: EventTerm}
    """
    event_terms: Dict[str, EventTerm]


class EventManager(ManagerBase):
    """Manages environment events and modifications.

    The EventManager handles:
    - Interval-based events (e.g., domain randomization every N steps)
    - Condition-based triggers (e.g., spawn obstacle when robot reaches position)
    - Reset events (trigger on environment reset)
    - Environment state modifications

    All operations are JIT-compilable and VMAP-compatible.
    """

    def init_state(
        self,
        rng: jax.Array,
        num_envs: int,
        config: config_dict.ConfigDict,
    ) -> EventState:
        """Initialize event state.

        Args:
            rng: JAX random key (unused for event manager)
            num_envs: Number of parallel environments
            config: Configuration dict containing event_terms

        Returns:
            Initial EventState with zero counters
        """
        event_terms = config.get("event_terms", {})
        num_events = len(event_terms)

        event_counters = jnp.zeros((num_events, num_envs), dtype=jnp.int32)
        last_trigger_step = jnp.zeros((num_events, num_envs), dtype=jnp.int32)

        return EventState(
            event_counters=event_counters,
            last_trigger_step=last_trigger_step,
        )

    def reset(
        self,
        state: EventState,
        env_ids: jax.Array,
        rng: jax.Array,
        config: config_dict.ConfigDict,
    ) -> EventState:
        """Reset event state for specified environments.

        Also triggers "reset" mode events.

        Args:
            state: Current event state
            env_ids: Boolean mask of environments to reset
            rng: JAX random key for event execution
            config: Configuration dict

        Returns:
            Updated event state with reset environments
        """
        # Reset counters for reset environments
        zero_counters = jnp.zeros_like(state.event_counters)
        new_event_counters = jnp.where(
            env_ids[None, :],
            zero_counters,
            state.event_counters
        )

        # Reset last trigger step
        zero_triggers = jnp.zeros_like(state.last_trigger_step)
        new_last_trigger_step = jnp.where(
            env_ids[None, :],
            zero_triggers,
            state.last_trigger_step
        )

        return state.replace(
            event_counters=new_event_counters,
            last_trigger_step=new_last_trigger_step,
        )

    def compute(
        self,
        state: EventState,
        env_state: mjx.Data,
        config: config_dict.ConfigDict,
        rng: jax.Array,
    ) -> Tuple[EventState, mjx.Data]:
        """Execute events and modify environment state.

        Args:
            state: Current event state
            env_state: Current MJX physics state
            config: Configuration dict with event_terms
            rng: JAX random key for event execution

        Returns:
            Tuple of (new_state, modified_env_state):
            - new_state: Updated event state
            - modified_env_state: Environment state after event modifications
        """
        event_terms = config.get("event_terms", {})
        modified_env_state = env_state
        new_event_counters = state.event_counters.copy()

        for event_idx, (event_name, event_config) in enumerate(event_terms.items()):
            # Check if event should trigger
            should_trigger = self._check_trigger(
                state, event_idx, event_config, env_state, config
            )

            # Execute event if triggered
            if jnp.any(should_trigger):
                rng, event_key = jax.random.split(rng)
                params = event_config.params or {}

                # Apply event function
                modified_env_state = event_config.func(
                    modified_env_state,
                    event_key,
                    config,
                    should_trigger,  # Pass mask to apply selectively
                    **params
                )

                # Reset counter for triggered environments
                new_event_counters = new_event_counters.at[event_idx].set(
                    jnp.where(should_trigger, 0, new_event_counters[event_idx])
                )

        # Increment all counters
        new_event_counters = new_event_counters + 1

        new_state = state.replace(
            event_counters=new_event_counters,
        )

        return new_state, modified_env_state

    def _check_trigger(
        self,
        state: EventState,
        event_idx: int,
        event_config: EventTerm,
        env_state: mjx.Data,
        config: config_dict.ConfigDict,
    ) -> jax.Array:
        """Check if event should trigger.

        Args:
            state: Current event state
            event_idx: Index of the event
            event_config: Event configuration
            env_state: Current environment state
            config: Configuration dict

        Returns:
            Boolean mask of environments where event should trigger
        """
        num_envs = env_state.qpos.shape[0]

        if event_config.mode == "interval":
            # Trigger every N steps
            elapsed = state.event_counters[event_idx]
            should_trigger = elapsed >= event_config.interval

        elif event_config.mode == "condition":
            # Trigger when condition is met
            if event_config.condition_func is not None:
                should_trigger = event_config.condition_func(env_state, config)
            else:
                should_trigger = jnp.zeros(num_envs, dtype=bool)

        elif event_config.mode == "reset":
            # Handled separately in reset() method
            should_trigger = jnp.zeros(num_envs, dtype=bool)

        else:
            # Unknown mode - no trigger
            should_trigger = jnp.zeros(num_envs, dtype=bool)

        return should_trigger


# ==================== Common Event Functions ====================

def event_randomize_friction(
    env_state: mjx.Data,
    rng: jax.Array,
    config: config_dict.ConfigDict,
    mask: jax.Array,
    friction_range: Tuple[float, float] = (0.5, 1.5),
    **kwargs,
) -> mjx.Data:
    """Randomize friction coefficients.

    Args:
        env_state: MJX state
        rng: JAX random key
        config: Configuration dict
        mask: Boolean mask of environments to apply randomization
        friction_range: (min, max) range for friction multiplier
        **kwargs: Additional arguments

    Returns:
        Modified MJX state with randomized friction
    """
    # TODO: Implement friction randomization via MJX model parameters
    # This requires modifying geom friction parameters
    # For now, return unchanged state
    return env_state


def event_randomize_mass(
    env_state: mjx.Data,
    rng: jax.Array,
    config: config_dict.ConfigDict,
    mask: jax.Array,
    mass_range: Tuple[float, float] = (0.8, 1.2),
    **kwargs,
) -> mjx.Data:
    """Randomize link masses.

    Args:
        env_state: MJX state
        rng: JAX random key
        config: Configuration dict
        mask: Boolean mask of environments to apply randomization
        mass_range: (min, max) range for mass multiplier
        **kwargs: Additional arguments

    Returns:
        Modified MJX state with randomized masses
    """
    # TODO: Implement mass randomization via MJX model parameters
    # For now, return unchanged state
    return env_state


def event_push_robot(
    env_state: mjx.Data,
    rng: jax.Array,
    config: config_dict.ConfigDict,
    mask: jax.Array,
    push_force_range: Tuple[float, float] = (-50.0, 50.0),
    **kwargs,
) -> mjx.Data:
    """Apply random push force to robot base.

    Args:
        env_state: MJX state
        rng: JAX random key
        config: Configuration dict
        mask: Boolean mask of environments to apply push
        push_force_range: (min, max) range for push force
        **kwargs: Additional arguments

    Returns:
        Modified MJX state with applied forces
    """
    num_envs = env_state.qpos.shape[0]

    # Generate random push forces
    rng, fx_key, fy_key = jax.random.split(rng, 3)

    fx = jax.random.uniform(
        fx_key, (num_envs,), minval=push_force_range[0], maxval=push_force_range[1]
    )
    fy = jax.random.uniform(
        fy_key, (num_envs,), minval=push_force_range[0], maxval=push_force_range[1]
    )

    # Apply force by modifying base velocity
    # This is a simplified version - proper implementation would use xfrc_applied
    base_vel = env_state.qvel[:, :2]  # Base x, y velocities

    push_vel = jnp.stack([fx, fy], axis=-1) * 0.001  # Scale force to velocity

    # Apply only to masked environments
    new_base_vel = jnp.where(
        mask[:, None],
        base_vel + push_vel,
        base_vel
    )

    # Update qvel
    new_qvel = env_state.qvel.at[:, :2].set(new_base_vel)

    return env_state.replace(qvel=new_qvel)


def event_reset_to_random_pose(
    env_state: mjx.Data,
    rng: jax.Array,
    config: config_dict.ConfigDict,
    mask: jax.Array,
    **kwargs,
) -> mjx.Data:
    """Reset robot to random pose.

    Args:
        env_state: MJX state
        rng: JAX random key
        config: Configuration dict
        mask: Boolean mask of environments to reset
        **kwargs: Additional arguments

    Returns:
        Modified MJX state with random poses
    """
    # TODO: Implement random pose generation
    # This would involve sampling valid joint configurations
    # For now, return unchanged state
    return env_state


def create_event_config(
    event_terms: Dict[str, EventTerm],
) -> config_dict.ConfigDict:
    """Create an event configuration dict.

    Args:
        event_terms: Dict of event terms

    Returns:
        ConfigDict with event configuration
    """
    return config_dict.create(
        event_terms=event_terms,
    )
