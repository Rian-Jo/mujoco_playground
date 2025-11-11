"""Action manager for processing and applying actions."""

from typing import Callable, Optional, Tuple

import flax.struct
import jax
import jax.numpy as jnp
from ml_collections import config_dict
from mujoco import mjx

from mujoco_playground.manager.manager_base import ManagerBase


@flax.struct.dataclass
class ActionState:
    """State for action manager.

    Attributes:
        last_action: Previous action taken (num_envs, action_dim)
        action_history: History of actions (num_envs, action_dim, history_len)
    """
    last_action: jax.Array
    action_history: Optional[jax.Array] = None


@flax.struct.dataclass
class ActionConfig:
    """Configuration for action manager.

    Attributes:
        action_space: Type of action space ("position", "velocity", "torque")
        action_scale: Scaling factor for actions
        action_clip: Tuple of (min, max) for clipping actions
        default_pose: Default joint positions for position control
        history_len: Number of previous actions to keep in history
        process_func: Optional custom action processing function
    """
    action_space: str = "position"
    action_scale: float = 0.5
    action_clip: Tuple[float, float] = (-1.0, 1.0)
    default_pose: Optional[jax.Array] = None
    history_len: int = 1
    process_func: Optional[Callable] = None


class ActionManager(ManagerBase):
    """Manages action processing and application to the environment.

    The ActionManager handles:
    - Action clipping and scaling
    - Conversion between action spaces (position, velocity, torque)
    - Action history tracking
    - Custom action processing functions

    All operations are JIT-compilable and VMAP-compatible.
    """

    def init_state(
        self,
        rng: jax.Array,
        num_envs: int,
        config: config_dict.ConfigDict,
    ) -> ActionState:
        """Initialize action state.

        Args:
            rng: JAX random key (unused for action manager)
            num_envs: Number of parallel environments
            config: Configuration dict containing ActionConfig fields

        Returns:
            Initial ActionState with zero actions
        """
        action_dim = config.get("action_dim", 0)

        last_action = jnp.zeros((num_envs, action_dim))

        action_history = None
        if config.get("history_len", 1) > 1:
            action_history = jnp.zeros(
                (num_envs, action_dim, config.history_len)
            )

        return ActionState(
            last_action=last_action,
            action_history=action_history,
        )

    def reset(
        self,
        state: ActionState,
        env_ids: jax.Array,
        rng: jax.Array,
        config: config_dict.ConfigDict,
    ) -> ActionState:
        """Reset action state for specified environments.

        Args:
            state: Current action state
            env_ids: Boolean mask of environments to reset
            rng: JAX random key (unused for action manager)
            config: Configuration dict

        Returns:
            Updated action state with reset environments
        """
        # Reset last action to zero
        zero_action = jnp.zeros_like(state.last_action)
        new_last_action = jnp.where(
            env_ids[:, None],
            zero_action,
            state.last_action
        )

        # Reset action history if it exists
        new_action_history = state.action_history
        if state.action_history is not None:
            zero_history = jnp.zeros_like(state.action_history)
            new_action_history = jnp.where(
                env_ids[:, None, None],
                zero_history,
                state.action_history
            )

        return state.replace(
            last_action=new_last_action,
            action_history=new_action_history,
        )

    def compute(
        self,
        state: ActionState,
        env_state: mjx.Data,
        config: config_dict.ConfigDict,
        action: jax.Array,
    ) -> Tuple[ActionState, jax.Array]:
        """Process raw actions into actuator commands.

        Args:
            state: Current action state
            env_state: Current MJX physics state
            config: Configuration dict with ActionConfig fields
            action: Raw actions from policy (num_envs, action_dim)

        Returns:
            Tuple of (new_state, ctrl):
            - new_state: Updated action state
            - ctrl: Actuator control signals (num_envs, nu)
        """
        # 1. Clip raw actions
        action_clip = config.get("action_clip", (-1.0, 1.0))
        clipped_action = jnp.clip(action, action_clip[0], action_clip[1])

        # 2. Apply custom processing if provided
        process_func = config.get("process_func", None)
        if process_func is not None:
            processed_action = process_func(
                clipped_action, state, env_state, config
            )
        else:
            processed_action = clipped_action

        # 3. Scale actions
        action_scale = config.get("action_scale", 1.0)
        scaled_action = processed_action * action_scale

        # 4. Convert to actuator commands based on action space
        ctrl = self._action_to_control(
            scaled_action, env_state, config
        )

        # 5. Update action history
        new_action_history = state.action_history
        if state.action_history is not None:
            # Shift history and append new action
            shifted = state.action_history[..., 1:]
            new_action_expanded = clipped_action[..., None]
            new_action_history = jnp.concatenate(
                [shifted, new_action_expanded], axis=-1
            )

        # 6. Create new state
        new_state = state.replace(
            last_action=clipped_action,
            action_history=new_action_history,
        )

        return new_state, ctrl

    def _action_to_control(
        self,
        action: jax.Array,
        env_state: mjx.Data,
        config: config_dict.ConfigDict,
    ) -> jax.Array:
        """Convert scaled action to actuator control signal.

        Args:
            action: Scaled action (num_envs, action_dim)
            env_state: Current MJX physics state
            config: Configuration dict

        Returns:
            Control signal (num_envs, nu)
        """
        action_space = config.get("action_space", "position")

        if action_space == "position":
            # Position control: action is offset from default pose
            default_pose = config.get("default_pose", None)
            if default_pose is None:
                # Use current position as default
                default_pose = env_state.qpos[:, 7:]  # Skip freejoint if present

            ctrl = default_pose + action

        elif action_space == "velocity":
            # Velocity control: action is desired velocity
            ctrl = action

        elif action_space == "torque":
            # Direct torque control
            ctrl = action

        elif action_space == "delta_position":
            # Delta position: action is change from current position
            current_pos = env_state.qpos[:, 7:]  # Skip freejoint
            ctrl = current_pos + action

        else:
            raise ValueError(
                f"Unknown action space: {action_space}. "
                f"Must be one of: position, velocity, torque, delta_position"
            )

        return ctrl


def create_action_config(
    action_dim: int,
    action_space: str = "position",
    action_scale: float = 0.5,
    action_clip: Tuple[float, float] = (-1.0, 1.0),
    default_pose: Optional[jax.Array] = None,
    history_len: int = 1,
    process_func: Optional[Callable] = None,
) -> config_dict.ConfigDict:
    """Create an action configuration dict.

    Args:
        action_dim: Dimension of action space
        action_space: Type of action space
        action_scale: Scaling factor for actions
        action_clip: Tuple of (min, max) for clipping
        default_pose: Default joint positions for position control
        history_len: Number of previous actions to keep
        process_func: Optional custom processing function

    Returns:
        ConfigDict with action configuration
    """
    return config_dict.create(
        action_dim=action_dim,
        action_space=action_space,
        action_scale=action_scale,
        action_clip=action_clip,
        default_pose=default_pose,
        history_len=history_len,
        process_func=process_func,
    )
