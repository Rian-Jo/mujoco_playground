"""Brax compatibility wrapper for manager-based environments.

This module provides wrappers to make ManagerBasedEnv compatible with Brax's
training infrastructure. It converts between ManagerBasedState and brax.envs.State,
handles observation formatting, and provides the necessary interfaces for PPO training.

Example:
    >>> from mujoco_playground.manager.configs.go2_velocity import create_go2_velocity_env
    >>> from mujoco_playground.manager.brax_wrapper import wrap_for_brax
    >>>
    >>> # Create manager-based environment
    >>> env = create_go2_velocity_env(task="flat", num_envs=4096)
    >>>
    >>> # Wrap for Brax training
    >>> brax_env = wrap_for_brax(env)
    >>>
    >>> # Now can use with Brax PPO
    >>> from brax.training.agents.ppo import train as train_ppo
    >>> train_fn = train_ppo.train(brax_env, ...)
"""

from typing import Any, Dict, Optional

import jax
import jax.numpy as jnp
from brax import envs
from brax.envs import State as BraxState
import flax.struct

from mujoco_playground.manager.manager_based_env import ManagerBasedEnv, ManagerBasedState


@flax.struct.dataclass
class BraxCompatibleState(BraxState):
    """Brax-compatible state that wraps ManagerBasedState.

    This extends Brax's State to include the full ManagerBasedState,
    allowing seamless conversion between the two representations.

    Attributes:
        pipeline_state: Not used (for Brax compatibility)
        obs: Observations dict
        reward: Scalar reward
        done: Done flag
        metrics: Metrics dict
        info: Info dict (contains full ManagerBasedState)
    """
    # Store full manager state in info for state continuity
    manager_state: Optional[ManagerBasedState] = None


class BraxManagerWrapper(envs.Env):
    """Wrapper to make ManagerBasedEnv compatible with Brax training.

    This wrapper:
    - Converts ManagerBasedState to brax.envs.State
    - Extracts policy observations from observation dict
    - Handles reset and step interfaces
    - Maintains state continuity through info dict

    Args:
        env: ManagerBasedEnv instance
        observation_key: Which observation group to use for policy ("policy" or "privileged")
        backend: Brax backend (not used, for compatibility)
    """

    def __init__(
        self,
        env: ManagerBasedEnv,
        observation_key: str = "policy",
        backend: Optional[str] = None,
    ):
        """Initialize Brax wrapper.

        Args:
            env: ManagerBasedEnv to wrap
            observation_key: Observation group to use
            backend: Ignored (for Brax compatibility)
        """
        self._env = env
        self._observation_key = observation_key
        self._backend = backend

    @property
    def observation_size(self) -> int:
        """Size of observation vector."""
        # This should be computed from the observation config
        # For now, return a default value
        # TODO: Compute actual size from observation terms
        return 48

    @property
    def action_size(self) -> int:
        """Size of action vector."""
        return self._env.action_size

    @property
    def backend(self) -> Optional[str]:
        """Backend name."""
        return self._backend

    def reset(self, rng: jax.Array) -> BraxCompatibleState:
        """Reset environment.

        Args:
            rng: JAX random key

        Returns:
            BraxCompatibleState with initial observations
        """
        # Reset manager-based environment
        manager_state = self._env.reset(rng)

        # Convert to Brax state
        brax_state = self._manager_to_brax_state(manager_state)

        return brax_state

    def step(
        self,
        state: BraxCompatibleState,
        action: jax.Array
    ) -> BraxCompatibleState:
        """Step environment.

        Args:
            state: Current Brax-compatible state
            action: Actions to take

        Returns:
            New BraxCompatibleState after step
        """
        # Extract manager state
        manager_state = state.manager_state

        # Step manager-based environment
        new_manager_state = self._env.step(manager_state, action)

        # Convert to Brax state
        brax_state = self._manager_to_brax_state(new_manager_state)

        return brax_state

    def _manager_to_brax_state(
        self,
        manager_state: ManagerBasedState
    ) -> BraxCompatibleState:
        """Convert ManagerBasedState to BraxCompatibleState.

        Args:
            manager_state: Manager-based state

        Returns:
            Brax-compatible state
        """
        # Extract observations
        observations = manager_state.info.get("observations", {})

        # Get policy observations
        if self._observation_key in observations:
            obs = observations[self._observation_key]
        else:
            # Fallback: concatenate all observations
            obs = jnp.concatenate([
                v for v in observations.values() if isinstance(v, jax.Array)
            ], axis=-1)

        # Get reward, done from info
        reward = manager_state.info.get("reward", jnp.zeros(obs.shape[0]))
        done = manager_state.info.get("done", jnp.zeros(obs.shape[0], dtype=bool))

        # Build metrics dict (for logging)
        metrics = {}
        for key, value in manager_state.info.items():
            if key.startswith("reward/") or key in ["done", "terminated", "truncated"]:
                metrics[key] = value

        # Create Brax-compatible state
        brax_state = BraxCompatibleState(
            pipeline_state=None,  # Not used
            obs=obs,
            reward=reward,
            done=done,
            metrics=metrics,
            info=manager_state.info,
            manager_state=manager_state,
        )

        return brax_state


def wrap_for_brax(
    env: ManagerBasedEnv,
    observation_key: str = "policy",
    backend: Optional[str] = None,
) -> BraxManagerWrapper:
    """Wrap ManagerBasedEnv for Brax training.

    Args:
        env: ManagerBasedEnv instance
        observation_key: Which observation group to use ("policy" or "privileged")
        backend: Brax backend (optional)

    Returns:
        BraxManagerWrapper compatible with Brax training

    Example:
        >>> env = create_go2_velocity_env(task="flat", num_envs=4096)
        >>> brax_env = wrap_for_brax(env)
        >>>
        >>> # Use with Brax PPO
        >>> from brax.training.agents.ppo import train as train_ppo
        >>> from brax.training.agents.ppo import networks as ppo_networks
        >>>
        >>> network_factory = ppo_networks.make_ppo_networks
        >>> train_fn = train_ppo.train(
        ...     brax_env,
        ...     num_timesteps=50_000_000,
        ...     num_evals=10,
        ...     episode_length=1000,
        ...     network_factory=network_factory,
        ... )
    """
    return BraxManagerWrapper(env, observation_key, backend)


# Convenience function for wrapping with auto-reset
def wrap_for_training(
    env: ManagerBasedEnv,
    episode_length: int = 1000,
    observation_key: str = "policy",
    action_repeat: int = 1,
) -> BraxManagerWrapper:
    """Wrap ManagerBasedEnv with training-specific wrappers.

    This is a higher-level wrapper that applies common training wrappers:
    - Brax compatibility
    - Episode length handling (already handled by ManagerBasedEnv)
    - Action repeat (if needed)

    Args:
        env: ManagerBasedEnv instance
        episode_length: Maximum episode length (should match env config)
        observation_key: Observation group to use
        action_repeat: Number of times to repeat each action

    Returns:
        Wrapped environment ready for training

    Example:
        >>> env = create_go2_velocity_env(task="flat", num_envs=4096)
        >>> train_env = wrap_for_training(env, episode_length=1000)
        >>>
        >>> # Train with Brax PPO
        >>> train_fn = train_ppo.train(train_env, ...)
    """
    # Wrap for Brax
    brax_env = wrap_for_brax(env, observation_key=observation_key)

    # Note: Episode length and auto-reset are already handled by ManagerBasedEnv
    # Action repeat could be added here if needed

    if action_repeat > 1:
        # TODO: Implement action repeat wrapper if needed
        pass

    return brax_env


# Integration with existing Brax wrappers
def wrap_with_brax_wrappers(
    env: ManagerBasedEnv,
    episode_length: int = 1000,
    action_repeat: int = 1,
    **kwargs,
) -> envs.Env:
    """Wrap with both manager wrapper and standard Brax wrappers.

    This provides maximum compatibility with Brax training scripts.

    Args:
        env: ManagerBasedEnv instance
        episode_length: Maximum episode length
        action_repeat: Action repeat
        **kwargs: Additional arguments for Brax wrappers

    Returns:
        Fully wrapped environment
    """
    from brax.envs.wrappers import training as training_wrappers

    # First wrap with manager wrapper
    brax_env = wrap_for_brax(env)

    # Then apply standard Brax training wrappers if needed
    # Note: Episode wrapping is already handled by ManagerBasedEnv

    # Could add:
    # - EpisodeWrapper (already handled)
    # - AutoResetWrapper (already handled)
    # - VmapWrapper (already vectorized)

    return brax_env
