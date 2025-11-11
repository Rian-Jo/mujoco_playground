"""Base classes for manager-based architecture.

This module provides the abstract base class and common types for all managers.
All managers follow a functional, JAX-native design with immutable state.
"""

import abc
from typing import Any, Callable, Dict, Optional, Tuple

import flax.struct
import jax
import jax.numpy as jnp
from ml_collections import config_dict
from mujoco import mjx


@flax.struct.dataclass
class ManagerState:
    """Base dataclass for manager states.

    All manager-specific states should inherit from this or use flax.struct.dataclass.
    This ensures immutability and compatibility with JAX transformations.
    """
    pass


class ManagerBase(abc.ABC):
    """Abstract base class for all managers.

    Key design principles:
    - Stateless/functional: All methods are pure functions
    - JAX-compatible: Works with JIT, VMAP, GRAD
    - Immutable state: State is passed in and returned, never modified in place
    - Type-safe: Uses type hints for all methods

    Lifecycle:
    1. init_state: Initialize manager state
    2. reset: Reset state for specific environments
    3. compute: Main computation (observations, rewards, etc.)
    """

    @abc.abstractmethod
    def init_state(
        self,
        rng: jax.Array,
        num_envs: int,
        config: config_dict.ConfigDict,
    ) -> Any:
        """Initialize manager state.

        Args:
            rng: JAX random key for initialization
            num_envs: Number of parallel environments
            config: Configuration dict for this manager

        Returns:
            Initial manager state (typically a flax.struct.dataclass)
        """
        pass

    @abc.abstractmethod
    def reset(
        self,
        state: Any,
        env_ids: jax.Array,
        rng: jax.Array,
        config: config_dict.ConfigDict,
    ) -> Any:
        """Reset manager state for specified environments.

        Args:
            state: Current manager state
            env_ids: Boolean mask of environments to reset (num_envs,)
            rng: JAX random key for reset
            config: Configuration dict for this manager

        Returns:
            Updated manager state with reset environments
        """
        pass

    @abc.abstractmethod
    def compute(
        self,
        state: Any,
        env_state: mjx.Data,
        config: config_dict.ConfigDict,
        **kwargs: Any,
    ) -> Tuple[Any, Any]:
        """Compute manager output.

        Args:
            state: Current manager state
            env_state: Current MJX physics state
            config: Configuration dict for this manager
            **kwargs: Additional arguments specific to the manager

        Returns:
            Tuple of (new_state, output)
            - new_state: Updated manager state
            - output: Manager-specific output (observations, rewards, etc.)
        """
        pass


class ManagerTermBase:
    """Base class for manager terms (reward terms, observation terms, etc.).

    Terms are the building blocks of managers. Each term computes a specific
    component (e.g., one reward signal, one observation feature) that the
    manager combines.
    """

    def __init__(
        self,
        func: Callable,
        weight: float = 1.0,
        params: Optional[Dict[str, Any]] = None,
    ):
        """Initialize a manager term.

        Args:
            func: Callable that computes the term value
            weight: Weight for this term (used in rewards)
            params: Additional parameters passed to the function
        """
        self.func = func
        self.weight = weight
        self.params = params or {}

    def __call__(self, *args: Any, **kwargs: Any) -> jax.Array:
        """Compute the term value.

        Args:
            *args: Positional arguments passed to the term function
            **kwargs: Keyword arguments (merged with self.params)

        Returns:
            Term value as JAX array
        """
        merged_kwargs = {**self.params, **kwargs}
        return self.func(*args, **merged_kwargs)


def create_term_from_config(
    term_config: Dict[str, Any],
    term_class: type = ManagerTermBase,
) -> ManagerTermBase:
    """Create a manager term from a configuration dict.

    Args:
        term_config: Dict with keys: func, weight, params, etc.
        term_class: Class to instantiate (default: ManagerTermBase)

    Returns:
        Instantiated term object
    """
    func = term_config.get("func")
    if func is None:
        raise ValueError("Term config must have 'func' key")

    weight = term_config.get("weight", 1.0)
    params = term_config.get("params", {})

    return term_class(func=func, weight=weight, params=params)


def batch_reset_state(
    state: flax.struct.PyTreeNode,
    reset_ids: jax.Array,
    reset_values: flax.struct.PyTreeNode,
) -> flax.struct.PyTreeNode:
    """Helper function to reset specific environments in a batched state.

    This is useful for auto-resetting terminated environments while keeping
    others unchanged. Works with any flax.struct.dataclass.

    Args:
        state: Current state pytree
        reset_ids: Boolean mask of environments to reset (num_envs,)
        reset_values: Values to use for reset environments

    Returns:
        Updated state with reset environments

    Example:
        >>> state = RewardState(episode_rewards=jnp.array([1.0, 2.0, 3.0]))
        >>> reset_ids = jnp.array([False, True, False])
        >>> reset_values = RewardState(episode_rewards=jnp.zeros(3))
        >>> new_state = batch_reset_state(state, reset_ids, reset_values)
        >>> new_state.episode_rewards  # [1.0, 0.0, 3.0]
    """
    # Expand reset_ids to match state structure
    reset_mask = reset_ids

    def _reset_leaf(old_leaf: jax.Array, new_leaf: jax.Array) -> jax.Array:
        """Reset leaf values where reset_ids is True."""
        # Handle scalar leaves
        if old_leaf.shape == ():
            return old_leaf

        # Expand mask to match leaf shape
        expanded_mask = reset_mask
        for _ in range(len(old_leaf.shape) - 1):
            expanded_mask = expanded_mask[..., None]

        return jnp.where(expanded_mask, new_leaf, old_leaf)

    # Apply reset to all leaves in the state pytree
    return jax.tree_map(_reset_leaf, state, reset_values)


def init_circular_buffer(
    shape: Tuple[int, ...],
    history_len: int,
    dtype: jnp.dtype = jnp.float32,
) -> jax.Array:
    """Initialize a circular buffer for observation history.

    Args:
        shape: Shape of a single observation (e.g., (num_envs, obs_dim))
        history_len: Number of timesteps to keep in history
        dtype: Data type for the buffer

    Returns:
        Zero-initialized buffer of shape (*shape, history_len)
    """
    buffer_shape = (*shape, history_len)
    return jnp.zeros(buffer_shape, dtype=dtype)


def update_circular_buffer(
    buffer: jax.Array,
    new_value: jax.Array,
) -> jax.Array:
    """Update circular buffer with new value.

    Shifts the buffer left and appends the new value at the end.

    Args:
        buffer: Current buffer of shape (..., history_len)
        new_value: New value to append of shape (...)

    Returns:
        Updated buffer with new value at the end

    Example:
        >>> buffer = jnp.array([[1, 2, 3], [4, 5, 6]])  # (2, 3)
        >>> new_value = jnp.array([7, 8])  # (2,)
        >>> updated = update_circular_buffer(buffer, new_value)
        >>> updated  # [[2, 3, 7], [5, 6, 8]]
    """
    # Shift left: drop oldest, keep the rest
    shifted = buffer[..., 1:]

    # Append new value
    new_value_expanded = new_value[..., None]
    return jnp.concatenate([shifted, new_value_expanded], axis=-1)


def flatten_history_buffer(buffer: jax.Array) -> jax.Array:
    """Flatten history buffer for use as observations.

    Args:
        buffer: History buffer of shape (num_envs, obs_dim, history_len)

    Returns:
        Flattened buffer of shape (num_envs, obs_dim * history_len)
    """
    num_envs = buffer.shape[0]
    flattened_dim = buffer.shape[1] * buffer.shape[2]
    return buffer.reshape(num_envs, flattened_dim)
