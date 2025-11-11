"""Manager-based architecture for MuJoCo Playground.

This module provides a manager-based training architecture inspired by mjlab,
fully adapted for JAX/Flax/Brax/MJX/MJWarp compatibility with functional
programming paradigms.

The architecture includes:
- ManagerBase: Abstract base class for all managers
- ActionManager: Processes actions and applies them to the environment
- ObservationManager: Collects and processes observations
- RewardManager: Computes modular reward terms
- TerminationManager: Checks termination conditions
- CommandManager: Generates and tracks commands
- EventManager: Handles environment events and modifications
- CurriculumManager: Manages training progression
- ManagerBasedEnv: Integrates all managers into a unified environment
"""

from mujoco_playground.manager.manager_base import (
    ManagerBase,
    ManagerState,
)

from mujoco_playground.manager.action_manager import (
    ActionManager,
    ActionState,
    ActionConfig,
)

from mujoco_playground.manager.observation_manager import (
    ObservationManager,
    ObservationState,
    ObservationTerm,
    ObservationConfig,
)

from mujoco_playground.manager.reward_manager import (
    RewardManager,
    RewardState,
    RewardTerm,
    RewardConfig,
)

from mujoco_playground.manager.termination_manager import (
    TerminationManager,
    TerminationState,
    TerminationTerm,
    TerminationConfig,
)

from mujoco_playground.manager.command_manager import (
    CommandManager,
    CommandState,
    CommandConfig,
)

from mujoco_playground.manager.event_manager import (
    EventManager,
    EventState,
    EventTerm,
    EventConfig,
)

from mujoco_playground.manager.curriculum_manager import (
    CurriculumManager,
    CurriculumState,
    CurriculumTerm,
    CurriculumConfig,
)

from mujoco_playground.manager.manager_based_env import (
    ManagerBasedEnv,
    ManagerBasedState,
    ManagerBasedConfig,
)

from mujoco_playground.manager.brax_wrapper import (
    BraxManagerWrapper,
    BraxCompatibleState,
    wrap_for_brax,
    wrap_for_training,
    wrap_with_brax_wrappers,
)

__all__ = [
    # Base
    "ManagerBase",
    "ManagerState",
    # Action
    "ActionManager",
    "ActionState",
    "ActionConfig",
    # Observation
    "ObservationManager",
    "ObservationState",
    "ObservationTerm",
    "ObservationConfig",
    # Reward
    "RewardManager",
    "RewardState",
    "RewardTerm",
    "RewardConfig",
    # Termination
    "TerminationManager",
    "TerminationState",
    "TerminationTerm",
    "TerminationConfig",
    # Command
    "CommandManager",
    "CommandState",
    "CommandConfig",
    # Event
    "EventManager",
    "EventState",
    "EventTerm",
    "EventConfig",
    # Curriculum
    "CurriculumManager",
    "CurriculumState",
    "CurriculumTerm",
    "CurriculumConfig",
    # Environment
    "ManagerBasedEnv",
    "ManagerBasedState",
    "ManagerBasedConfig",
    # Brax Integration
    "BraxManagerWrapper",
    "BraxCompatibleState",
    "wrap_for_brax",
    "wrap_for_training",
    "wrap_with_brax_wrappers",
]
