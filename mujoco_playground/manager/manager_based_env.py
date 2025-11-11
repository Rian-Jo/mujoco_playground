"""Manager-based environment integrating all managers into a unified training interface."""

from typing import Any, Dict, Optional, Tuple

import flax.struct
import jax
import jax.numpy as jnp
import mujoco
from ml_collections import config_dict
from mujoco import mjx

from mujoco_playground.manager.action_manager import ActionManager, ActionState
from mujoco_playground.manager.command_manager import CommandManager, CommandState
from mujoco_playground.manager.curriculum_manager import CurriculumManager, CurriculumState
from mujoco_playground.manager.event_manager import EventManager, EventState
from mujoco_playground.manager.observation_manager import ObservationManager, ObservationState
from mujoco_playground.manager.reward_manager import RewardManager, RewardState
from mujoco_playground.manager.termination_manager import TerminationManager, TerminationState


@flax.struct.dataclass
class ManagerBasedState:
    """Complete state for manager-based environment.

    This dataclass holds all state needed for a manager-based environment,
    including physics state and all manager states.

    Attributes:
        env_state: MJX physics state
        obs_state: Observation manager state
        reward_state: Reward manager state
        termination_state: Termination manager state
        action_state: Action manager state
        command_state: Command manager state
        event_state: Event manager state
        curriculum_state: Curriculum manager state
        info: Dict of additional information for logging
        rng: Random key for next step
    """
    env_state: mjx.Data
    obs_state: ObservationState
    reward_state: RewardState
    termination_state: TerminationState
    action_state: ActionState
    command_state: CommandState
    event_state: EventState
    curriculum_state: CurriculumState
    info: Dict[str, Any]
    rng: jax.Array


@flax.struct.dataclass
class ManagerBasedConfig:
    """Configuration for manager-based environment.

    This bundles all manager configurations into a single config object.
    """
    # Environment settings
    xml_path: str
    num_envs: int = 1
    backend: str = "mjx"  # "mjx" or "warp"

    # Physics settings
    ctrl_dt: float = 0.02
    sim_dt: float = 0.004
    n_substeps: int = 5

    # Manager configurations
    action_config: Optional[config_dict.ConfigDict] = None
    observation_config: Optional[config_dict.ConfigDict] = None
    reward_config: Optional[config_dict.ConfigDict] = None
    termination_config: Optional[config_dict.ConfigDict] = None
    command_config: Optional[config_dict.ConfigDict] = None
    event_config: Optional[config_dict.ConfigDict] = None
    curriculum_config: Optional[config_dict.ConfigDict] = None


class ManagerBasedEnv:
    """Manager-based RL environment for MuJoCo Playground.

    This class integrates all managers into a unified training interface that
    is compatible with JAX/Flax/Brax/MJX/MJWarp.

    The environment follows a functional design:
    - All state is immutable and passed explicitly
    - All operations are JIT-compilable
    - Supports VMAP for parallelization
    - Compatible with Brax training infrastructure

    Example:
        >>> config = ManagerBasedConfig(
        ...     xml_path="path/to/model.xml",
        ...     num_envs=4096,
        ...     action_config=...,
        ...     observation_config=...,
        ...     reward_config=...,
        ... )
        >>> env = ManagerBasedEnv(config)
        >>> rng = jax.random.PRNGKey(0)
        >>> state = env.reset(rng)
        >>> action = jnp.zeros((config.num_envs, env.action_size))
        >>> state = env.step(state, action)
    """

    def __init__(self, config: config_dict.ConfigDict):
        """Initialize manager-based environment.

        Args:
            config: Configuration dict with all environment and manager settings
        """
        self.config = config

        # Load MuJoCo model
        self.mj_model = mujoco.MjModel.from_xml_path(config.xml_path)

        # Create backend-specific model
        backend = config.get("backend", "mjx")
        if backend == "mjx":
            self.model = mjx.put_model(self.mj_model)
        elif backend == "warp":
            # MJWarp integration (requires warp and mjwarp)
            try:
                import warp as wp
                import mjwarp
                wp.init()
                self.model = mjwarp.put_model(self.mj_model)
            except ImportError:
                raise ImportError(
                    "MJWarp backend requires 'warp' and 'mjwarp' packages. "
                    "Install with: pip install warp mjwarp"
                )
        else:
            raise ValueError(f"Unknown backend: {backend}. Must be 'mjx' or 'warp'")

        # Compute derived config values
        self.config.n_substeps = int(config.ctrl_dt / config.sim_dt)

        # Initialize managers
        self.obs_manager = ObservationManager()
        self.reward_manager = RewardManager()
        self.termination_manager = TerminationManager()
        self.action_manager = ActionManager()
        self.command_manager = CommandManager()
        self.event_manager = EventManager()
        self.curriculum_manager = CurriculumManager()

    @property
    def action_size(self) -> int:
        """Action space size."""
        return self.config.get("action_dim", self.mj_model.nu)

    @property
    def observation_size(self) -> Dict[str, int]:
        """Observation space sizes per group."""
        # This would need to be computed from observation config
        # For now, return placeholder
        return {"policy": 48, "value": 48}

    def reset(self, rng: jax.Array) -> ManagerBasedState:
        """Reset environment and all managers.

        Args:
            rng: JAX random key for initialization

        Returns:
            Initial ManagerBasedState
        """
        # Split RNG for different components
        rngs = jax.random.split(rng, 9)

        # Reset physics to default state
        env_state = self._reset_physics(rngs[0])

        # Get number of environments
        num_envs = self.config.get("num_envs", 1)

        # Initialize all manager states
        obs_state = self.obs_manager.init_state(
            rngs[1], num_envs, self.config.get("observation_config", {})
        )
        reward_state = self.reward_manager.init_state(
            rngs[2], num_envs, self.config.get("reward_config", {})
        )
        termination_state = self.termination_manager.init_state(
            rngs[3], num_envs, self.config.get("termination_config", {})
        )
        action_state = self.action_manager.init_state(
            rngs[4], num_envs, self.config.get("action_config", {})
        )
        command_state = self.command_manager.init_state(
            rngs[5], num_envs, self.config.get("command_config", {})
        )
        event_state = self.event_manager.init_state(
            rngs[6], num_envs, self.config.get("event_config", {})
        )
        curriculum_state = self.curriculum_manager.init_state(
            rngs[7], num_envs, self.config.get("curriculum_config", {})
        )

        # Compute initial observations
        obs_state, observations = self.obs_manager.compute(
            obs_state,
            env_state,
            self.config.get("observation_config", {}),
            rngs[8],
            action_state=action_state,
            command_state=command_state,
        )

        return ManagerBasedState(
            env_state=env_state,
            obs_state=obs_state,
            reward_state=reward_state,
            termination_state=termination_state,
            action_state=action_state,
            command_state=command_state,
            event_state=event_state,
            curriculum_state=curriculum_state,
            info={"observations": observations},
            rng=rngs[0],
        )

    def step(
        self,
        state: ManagerBasedState,
        action: jax.Array,
    ) -> ManagerBasedState:
        """Execute one environment step with all managers.

        This method orchestrates all managers to:
        1. Process actions
        2. Update commands
        3. Step physics
        4. Handle events
        5. Compute observations
        6. Compute rewards
        7. Check terminations
        8. Update curriculum
        9. Auto-reset terminated environments

        Args:
            state: Current manager-based state
            action: Actions from policy (num_envs, action_dim)

        Returns:
            New manager-based state after step
        """
        # Split RNG for different operations
        rngs = jax.random.split(state.rng, 6)

        # 1. Process action
        action_state, ctrl = self.action_manager.compute(
            state.action_state,
            state.env_state,
            self.config.get("action_config", self.config),
            action,
        )

        # 2. Update commands
        command_state, command = self.command_manager.compute(
            state.command_state,
            state.env_state,
            self.config.get("command_config", self.config),
            rngs[0],
        )

        # 3. Step physics
        env_state = self._step_physics(state.env_state, ctrl)

        # 4. Handle events
        event_state, env_state = self.event_manager.compute(
            state.event_state,
            env_state,
            self.config.get("event_config", self.config),
            rngs[1],
        )

        # 5. Compute observations
        obs_state, observations = self.obs_manager.compute(
            state.obs_state,
            env_state,
            self.config.get("observation_config", self.config),
            rngs[2],
            action_state=action_state,
            command_state=command_state,
        )

        # 6. Compute rewards
        reward_state, (reward, reward_info) = self.reward_manager.compute(
            state.reward_state,
            env_state,
            self.config.get("reward_config", self.config),
            action,
            action_state=action_state,
            command_state=command_state,
        )

        # 7. Check terminations
        termination_state, termination_info = self.termination_manager.compute(
            state.termination_state,
            env_state,
            self.config.get("termination_config", self.config),
        )

        # 8. Update curriculum
        curriculum_metrics = {
            "success": reward_info.get("success", jnp.zeros(self.config.get("num_envs", 1))),
            "reward": reward,
        }
        curriculum_state, updated_config = self.curriculum_manager.compute(
            state.curriculum_state,
            curriculum_metrics,
            self.config.get("curriculum_config", self.config),
        )

        # 9. Auto-reset terminated environments
        should_reset = termination_info["done"]
        if jnp.any(should_reset):
            env_state, obs_state, reward_state, termination_state, action_state, command_state = (
                self._auto_reset(
                    env_state,
                    obs_state,
                    reward_state,
                    termination_state,
                    action_state,
                    command_state,
                    should_reset,
                    rngs[3],
                )
            )

        # 10. Collect info
        info = {
            "reward": reward,
            "observations": observations,
            "command": command,
            **reward_info,
            **termination_info,
        }

        return ManagerBasedState(
            env_state=env_state,
            obs_state=obs_state,
            reward_state=reward_state,
            termination_state=termination_state,
            action_state=action_state,
            command_state=command_state,
            event_state=event_state,
            curriculum_state=curriculum_state,
            info=info,
            rng=rngs[4],
        )

    def _reset_physics(self, rng: jax.Array) -> mjx.Data:
        """Reset physics simulation to default state.

        Args:
            rng: JAX random key for randomization

        Returns:
            Initial MJX data state
        """
        num_envs = self.config.get("num_envs", 1)

        # Create default MJX data
        data = mjx.make_data(self.model)

        # Replicate for multiple environments
        if num_envs > 1:
            # Stack data for vectorization
            data = jax.tree_map(
                lambda x: jnp.tile(x[None], (num_envs,) + (1,) * len(x.shape)),
                data
            )

        return data

    def _step_physics(self, env_state: mjx.Data, ctrl: jax.Array) -> mjx.Data:
        """Step physics simulation.

        Args:
            env_state: Current MJX state
            ctrl: Control inputs (num_envs, nu)

        Returns:
            Updated MJX state after stepping
        """
        backend = self.config.get("backend", "mjx")

        # Set control
        env_state = env_state.replace(ctrl=ctrl)

        # Sub-stepping for simulation stability
        for _ in range(self.config.n_substeps):
            if backend == "mjx":
                env_state = mjx.step(self.model, env_state)
            elif backend == "warp":
                import mjwarp
                env_state = mjwarp.step(self.model, env_state)

        return env_state

    def _auto_reset(
        self,
        env_state: mjx.Data,
        obs_state: ObservationState,
        reward_state: RewardState,
        termination_state: TerminationState,
        action_state: ActionState,
        command_state: CommandState,
        reset_mask: jax.Array,
        rng: jax.Array,
    ) -> Tuple[mjx.Data, ObservationState, RewardState, TerminationState, ActionState, CommandState]:
        """Auto-reset terminated environments.

        Args:
            env_state: Current environment state
            obs_state: Current observation state
            reward_state: Current reward state
            termination_state: Current termination state
            action_state: Current action state
            command_state: Current command state
            reset_mask: Boolean mask of environments to reset
            rng: JAX random key

        Returns:
            Tuple of reset states
        """
        # Split RNG
        rngs = jax.random.split(rng, 6)

        # Reset physics for masked environments
        reset_env_state = self._reset_physics(rngs[0])
        env_state = jax.tree_map(
            lambda new, old: jnp.where(reset_mask[:, None], new, old),
            reset_env_state,
            env_state
        )

        # Reset manager states
        obs_state = self.obs_manager.reset(
            obs_state,
            reset_mask,
            rngs[1],
            self.config.get("observation_config", {}),
        )

        reward_state = self.reward_manager.reset(
            reward_state,
            reset_mask,
            rngs[2],
            self.config.get("reward_config", {}),
        )

        termination_state = self.termination_manager.reset(
            termination_state,
            reset_mask,
            rngs[3],
            self.config.get("termination_config", {}),
        )

        action_state = self.action_manager.reset(
            action_state,
            reset_mask,
            rngs[4],
            self.config.get("action_config", {}),
        )

        command_state = self.command_manager.reset(
            command_state,
            reset_mask,
            rngs[5],
            self.config.get("command_config", {}),
        )

        return (
            env_state,
            obs_state,
            reward_state,
            termination_state,
            action_state,
            command_state,
        )


def create_manager_based_config(
    xml_path: str,
    num_envs: int = 4096,
    backend: str = "mjx",
    **kwargs,
) -> config_dict.ConfigDict:
    """Create a manager-based environment configuration.

    Args:
        xml_path: Path to MuJoCo XML model
        num_envs: Number of parallel environments
        backend: Physics backend ("mjx" or "warp")
        **kwargs: Additional configuration parameters

    Returns:
        ConfigDict with complete environment configuration
    """
    config = config_dict.create(
        xml_path=xml_path,
        num_envs=num_envs,
        backend=backend,
        ctrl_dt=0.02,
        sim_dt=0.004,
        **kwargs
    )

    return config
