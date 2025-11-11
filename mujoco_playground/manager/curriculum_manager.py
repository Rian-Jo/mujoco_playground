"""Curriculum manager for managing training progression and difficulty."""

from typing import Callable, Dict, Optional, Tuple

import flax.struct
import jax
import jax.numpy as jnp
from ml_collections import config_dict

from mujoco_playground.manager.manager_base import ManagerBase


@flax.struct.dataclass
class CurriculumState:
    """State for curriculum manager.

    Attributes:
        difficulty_level: Current difficulty level (num_envs,) in [0, 1]
        success_rate: Recent success rate (num_envs,)
        curriculum_step: Global curriculum step counter
        success_buffer: Rolling buffer of recent success flags (num_envs, buffer_size)
        buffer_index: Current index in success buffer (num_envs,)
    """
    difficulty_level: jax.Array
    success_rate: jax.Array
    curriculum_step: int
    success_buffer: jax.Array
    buffer_index: jax.Array


@flax.struct.dataclass
class CurriculumTerm:
    """Configuration for curriculum progression on a single parameter.

    Attributes:
        param_name: Name of config parameter to modify
        start_value: Value at difficulty=0
        end_value: Value at difficulty=1
        schedule: Progression schedule ("linear", "exponential", "threshold")
    """
    param_name: str
    start_value: float
    end_value: float
    schedule: str = "linear"


@flax.struct.dataclass
class CurriculumConfig:
    """Configuration for curriculum manager.

    Attributes:
        curriculum_terms: Dict of curriculum terms {term_name: CurriculumTerm}
        curriculum_mode: Mode for difficulty progression
            - "linear": Linear progression over time
            - "threshold": Increase when success rate exceeds threshold
            - "adaptive": Adaptive based on recent performance
        curriculum_length: Total steps for linear progression
        curriculum_threshold: Success rate threshold for threshold mode
        success_buffer_size: Size of rolling success buffer
        update_interval: Steps between curriculum updates
    """
    curriculum_terms: Dict[str, CurriculumTerm]
    curriculum_mode: str = "threshold"
    curriculum_length: int = 10_000_000
    curriculum_threshold: float = 0.8
    success_buffer_size: int = 100
    update_interval: int = 1000


class CurriculumManager(ManagerBase):
    """Manages curriculum learning progression.

    The CurriculumManager handles:
    - Adjusting task difficulty over training
    - Modifying config parameters based on performance
    - Tracking learning progress metrics
    - Multiple curriculum strategies

    All operations are JIT-compilable and VMAP-compatible.
    """

    def init_state(
        self,
        rng: jax.Array,
        num_envs: int,
        config: config_dict.ConfigDict,
    ) -> CurriculumState:
        """Initialize curriculum state.

        Args:
            rng: JAX random key (unused for curriculum manager)
            num_envs: Number of parallel environments
            config: Configuration dict containing curriculum_terms

        Returns:
            Initial CurriculumState starting at difficulty=0
        """
        buffer_size = config.get("success_buffer_size", 100)

        difficulty_level = jnp.zeros(num_envs)
        success_rate = jnp.zeros(num_envs)
        curriculum_step = 0
        success_buffer = jnp.zeros((num_envs, buffer_size), dtype=bool)
        buffer_index = jnp.zeros(num_envs, dtype=jnp.int32)

        return CurriculumState(
            difficulty_level=difficulty_level,
            success_rate=success_rate,
            curriculum_step=curriculum_step,
            success_buffer=success_buffer,
            buffer_index=buffer_index,
        )

    def reset(
        self,
        state: CurriculumState,
        env_ids: jax.Array,
        rng: jax.Array,
        config: config_dict.ConfigDict,
    ) -> CurriculumState:
        """Reset curriculum state for specified environments.

        Note: Difficulty level is NOT reset - it persists across episodes.
        Only episodic metrics are reset.

        Args:
            state: Current curriculum state
            env_ids: Boolean mask of environments to reset
            rng: JAX random key (unused for curriculum manager)
            config: Configuration dict

        Returns:
            Updated curriculum state with reset episodic metrics
        """
        # Don't reset difficulty_level - it should persist
        # Only reset episodic metrics if needed

        # Could optionally reset success buffer for reset environments
        # For now, keep all state persistent

        return state

    def compute(
        self,
        state: CurriculumState,
        metrics: Dict[str, jax.Array],
        config: config_dict.ConfigDict,
    ) -> Tuple[CurriculumState, config_dict.ConfigDict]:
        """Update curriculum and return modified config.

        Args:
            state: Current curriculum state
            metrics: Dict of performance metrics (must include "success")
            config: Configuration dict

        Returns:
            Tuple of (new_state, updated_config):
            - new_state: Updated curriculum state
            - updated_config: Config with modified parameters
        """
        curriculum_mode = config.get("curriculum_mode", "threshold")
        update_interval = config.get("update_interval", 1000)

        # Check if update needed
        should_update = (state.curriculum_step % update_interval) == 0

        if should_update:
            # Update success buffer
            success = metrics.get("success", jnp.zeros(state.success_rate.shape[0]))
            new_success_buffer, new_buffer_index = self._update_success_buffer(
                state.success_buffer,
                state.buffer_index,
                success
            )

            # Compute new success rate
            new_success_rate = jnp.mean(new_success_buffer, axis=-1)

            # Update difficulty based on mode
            new_difficulty = self._compute_difficulty(
                state, new_success_rate, config
            )
        else:
            new_success_buffer = state.success_buffer
            new_buffer_index = state.buffer_index
            new_success_rate = state.success_rate
            new_difficulty = state.difficulty_level

        # Apply curriculum to config
        updated_config = self._apply_curriculum(new_difficulty, config)

        # Update state
        new_state = state.replace(
            difficulty_level=new_difficulty,
            success_rate=new_success_rate,
            curriculum_step=state.curriculum_step + 1,
            success_buffer=new_success_buffer,
            buffer_index=new_buffer_index,
        )

        return new_state, updated_config

    def _update_success_buffer(
        self,
        buffer: jax.Array,
        index: jax.Array,
        success: jax.Array,
    ) -> Tuple[jax.Array, jax.Array]:
        """Update rolling success buffer.

        Args:
            buffer: Current success buffer (num_envs, buffer_size)
            index: Current buffer indices (num_envs,)
            success: New success flags (num_envs,)

        Returns:
            Tuple of (new_buffer, new_index)
        """
        buffer_size = buffer.shape[1]

        # Update buffer at current index for each environment
        new_buffer = buffer
        for i in range(buffer.shape[0]):
            idx = index[i]
            new_buffer = new_buffer.at[i, idx].set(success[i])

        # Increment index (wrap around)
        new_index = (index + 1) % buffer_size

        return new_buffer, new_index

    def _compute_difficulty(
        self,
        state: CurriculumState,
        success_rate: jax.Array,
        config: config_dict.ConfigDict,
    ) -> jax.Array:
        """Compute new difficulty level.

        Args:
            state: Current curriculum state
            success_rate: Recent success rate
            config: Configuration dict

        Returns:
            New difficulty level (num_envs,)
        """
        curriculum_mode = config.get("curriculum_mode", "threshold")

        if curriculum_mode == "linear":
            # Linear progression over time
            curriculum_length = config.get("curriculum_length", 10_000_000)
            progress = jnp.minimum(state.curriculum_step / curriculum_length, 1.0)
            new_difficulty = jnp.ones_like(state.difficulty_level) * progress

        elif curriculum_mode == "threshold":
            # Increase when success rate exceeds threshold
            threshold = config.get("curriculum_threshold", 0.8)
            step_size = config.get("curriculum_step_size", 0.1)

            # Increase difficulty if success rate above threshold
            should_increase = success_rate > threshold
            should_decrease = success_rate < (threshold - 0.2)  # Hysteresis

            difficulty_delta = jnp.where(
                should_increase,
                step_size,
                jnp.where(should_decrease, -step_size, 0.0)
            )

            new_difficulty = jnp.clip(
                state.difficulty_level + difficulty_delta,
                0.0,
                1.0
            )

        elif curriculum_mode == "adaptive":
            # Adaptive based on success rate
            # Target success rate around 0.5-0.7 for optimal learning
            target_success = config.get("target_success_rate", 0.6)
            adaptation_rate = config.get("adaptation_rate", 0.01)

            # Increase difficulty if too easy, decrease if too hard
            error = success_rate - target_success
            difficulty_delta = error * adaptation_rate

            new_difficulty = jnp.clip(
                state.difficulty_level + difficulty_delta,
                0.0,
                1.0
            )

        else:
            # Unknown mode - no change
            new_difficulty = state.difficulty_level

        return new_difficulty

    def _apply_curriculum(
        self,
        difficulty: jax.Array,
        config: config_dict.ConfigDict,
    ) -> config_dict.ConfigDict:
        """Apply curriculum by modifying config parameters.

        Args:
            difficulty: Current difficulty level (num_envs,)
            config: Configuration dict

        Returns:
            Modified config dict
        """
        curriculum_terms = config.get("curriculum_terms", {})

        # Create a copy of config to modify
        updated_config = config.copy()

        # Take mean difficulty across environments for config modification
        mean_difficulty = jnp.mean(difficulty)

        for term_name, term_config in curriculum_terms.items():
            # Interpolate between start and end values
            if term_config.schedule == "linear":
                value = (
                    term_config.start_value +
                    mean_difficulty * (term_config.end_value - term_config.start_value)
                )
            elif term_config.schedule == "exponential":
                # Exponential interpolation
                log_start = jnp.log(term_config.start_value + 1e-8)
                log_end = jnp.log(term_config.end_value + 1e-8)
                log_value = log_start + mean_difficulty * (log_end - log_start)
                value = jnp.exp(log_value)
            else:
                # Default to linear
                value = (
                    term_config.start_value +
                    mean_difficulty * (term_config.end_value - term_config.start_value)
                )

            # Update config parameter
            updated_config[term_config.param_name] = float(value)

        return updated_config


# ==================== Helper Functions ====================

def create_curriculum_config(
    curriculum_terms: Dict[str, CurriculumTerm],
    curriculum_mode: str = "threshold",
    curriculum_length: int = 10_000_000,
    curriculum_threshold: float = 0.8,
    success_buffer_size: int = 100,
    update_interval: int = 1000,
) -> config_dict.ConfigDict:
    """Create a curriculum configuration dict.

    Args:
        curriculum_terms: Dict of curriculum terms
        curriculum_mode: Mode for difficulty progression
        curriculum_length: Total steps for linear progression
        curriculum_threshold: Success rate threshold
        success_buffer_size: Size of rolling success buffer
        update_interval: Steps between updates

    Returns:
        ConfigDict with curriculum configuration
    """
    return config_dict.create(
        curriculum_terms=curriculum_terms,
        curriculum_mode=curriculum_mode,
        curriculum_length=curriculum_length,
        curriculum_threshold=curriculum_threshold,
        success_buffer_size=success_buffer_size,
        update_interval=update_interval,
    )


# ==================== Example Curriculum Functions ====================

def curriculum_command_scale(
    start_scale: float = 0.2,
    end_scale: float = 1.0,
) -> CurriculumTerm:
    """Create curriculum term for command scaling.

    Gradually increase command difficulty (e.g., velocity range).

    Args:
        start_scale: Initial scale (easy)
        end_scale: Final scale (hard)

    Returns:
        CurriculumTerm for command scaling
    """
    return CurriculumTerm(
        param_name="command_scale",
        start_value=start_scale,
        end_value=end_scale,
        schedule="linear"
    )


def curriculum_terrain_difficulty(
    start_difficulty: float = 0.0,
    end_difficulty: float = 1.0,
) -> CurriculumTerm:
    """Create curriculum term for terrain difficulty.

    Gradually increase terrain complexity.

    Args:
        start_difficulty: Initial difficulty (flat terrain)
        end_difficulty: Final difficulty (rough terrain)

    Returns:
        CurriculumTerm for terrain difficulty
    """
    return CurriculumTerm(
        param_name="terrain_difficulty",
        start_value=start_difficulty,
        end_value=end_difficulty,
        schedule="linear"
    )


def curriculum_disturbance_intensity(
    start_intensity: float = 0.0,
    end_intensity: float = 1.0,
) -> CurriculumTerm:
    """Create curriculum term for disturbance intensity.

    Gradually increase push forces and perturbations.

    Args:
        start_intensity: Initial intensity (no disturbances)
        end_intensity: Final intensity (strong disturbances)

    Returns:
        CurriculumTerm for disturbance intensity
    """
    return CurriculumTerm(
        param_name="disturbance_intensity",
        start_value=start_intensity,
        end_value=end_intensity,
        schedule="linear"
    )
