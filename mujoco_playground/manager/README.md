# Manager-Based Architecture for MuJoCo Playground

This directory contains a manager-based training architecture inspired by mjlab, fully adapted for JAX/Flax/Brax/MJX/MJWarp compatibility with functional programming paradigms.

## Overview

The manager-based architecture organizes RL environment functionality into modular, composable managers that handle different aspects of training:

- **ActionManager**: Processes actions from the policy and converts them to actuator commands
- **ObservationManager**: Collects and processes observations with noise, clipping, and history
- **RewardManager**: Computes modular reward terms with configurable weights
- **TerminationManager**: Checks termination conditions and manages episode resets
- **CommandManager**: Generates and tracks command signals (e.g., velocity targets)
- **EventManager**: Handles environment events and modifications (e.g., domain randomization)
- **CurriculumManager**: Manages training progression and difficulty adaptation
- **ManagerBasedEnv**: Integrates all managers into a unified training environment

## Key Features

### ✅ JAX-Native Design
- All operations are pure functions
- Fully JIT-compilable for maximum performance
- VMAP-ready for parallelization across environments
- Immutable state management with flax.struct

### ✅ Modular & Composable
- Separate concerns into independent managers
- Easy to add/remove/modify individual terms
- Reusable components across different tasks
- Configuration-driven task definition

### ✅ Performance Optimized
- Full GPU acceleration via JAX
- Efficient parallelization with VMAP
- Compatible with Brax's vectorized training
- Support for both MJX and MJWarp backends

### ✅ Flexible & Extensible
- Easy to add custom managers and terms
- Plugin-based architecture
- Works with existing mujoco_playground infrastructure
- Backward compatible with non-manager environments

## Architecture

```
ManagerBasedEnv
├── ActionManager        → Process actions
├── ObservationManager   → Collect observations
├── RewardManager        → Compute rewards
├── TerminationManager   → Check terminations
├── CommandManager       → Generate commands
├── EventManager         → Handle events
└── CurriculumManager    → Adapt difficulty
```

Each manager:
1. Has an immutable state (flax.struct.dataclass)
2. Implements `init_state()`, `reset()`, and `compute()` methods
3. Takes state as input and returns new state + output
4. Is fully JIT-compilable and VMAP-compatible

## Quick Start

### 1. Define Manager Configuration

```python
from ml_collections import config_dict
from mujoco_playground.manager.observation_manager import ObservationTerm, get_joint_positions
from mujoco_playground.manager.reward_manager import RewardTerm, reward_velocity_tracking

config = config_dict.create(
    # Observation terms
    observation_terms={
        "policy": {
            "joint_pos": ObservationTerm(
                func=get_joint_positions,
                noise_config={"std": 0.01},
                scale=1.0,
            ),
            # ... more observation terms
        }
    },

    # Reward terms
    reward_terms={
        "velocity_tracking": RewardTerm(
            func=reward_velocity_tracking,
            weight=2.0,
        ),
        # ... more reward terms
    },

    # Termination terms
    termination_terms={
        "bad_orientation": TerminationTerm(
            func=termination_bad_orientation,
            params={"threshold": 0.5},
        ),
    },

    # Command generation
    command_func=command_velocity_tracking,
    command_resample_interval=500,
)
```

### 2. Create Environment

```python
from mujoco_playground.manager import ManagerBasedEnv

env = ManagerBasedEnv(config)

# Reset environment
rng = jax.random.PRNGKey(0)
state = env.reset(rng)

# Step environment
action = jnp.zeros((num_envs, action_dim))
state = env.step(state, action)

# Access results
observations = state.info["observations"]
reward = state.info["reward"]
done = state.info["done"]
```

### 3. Train with Brax PPO

```python
# See learning/train_manager_ppo.py for complete example

python learning/train_manager_ppo.py \
    --env_name Go1-velocity-manager \
    --num_envs 4096 \
    --num_timesteps 50000000 \
    --backend mjx
```

## Manager Details

### ObservationManager

Collects and processes observations from environment state.

**Features:**
- Modular observation terms
- Noise injection (Gaussian or uniform)
- Clipping and scaling
- Observation history buffers
- Grouped observations (policy, value, privileged)

**Example:**
```python
from mujoco_playground.manager.observation_manager import (
    ObservationManager,
    ObservationTerm,
    get_joint_positions,
)

obs_config = {
    "observation_terms": {
        "policy": {
            "joint_pos": ObservationTerm(
                func=get_joint_positions,
                noise_config={"std": 0.01, "type": "gaussian"},
                clip_range=(-10.0, 10.0),
                scale=1.0,
                history_len=3,  # Keep 3 timesteps
                flatten_history=True,
            )
        }
    }
}

obs_manager = ObservationManager()
obs_state = obs_manager.init_state(rng, num_envs, obs_config)
obs_state, observations = obs_manager.compute(
    obs_state, env_state, obs_config, rng
)
```

**Common Observation Functions:**
- `get_joint_positions`: Joint angles
- `get_joint_velocities`: Joint velocities
- `get_base_orientation`: Base orientation quaternion
- `get_base_linear_velocity`: Base linear velocity
- `get_base_angular_velocity`: Base angular velocity
- `get_last_action`: Previous action
- `get_current_command`: Current command signal

### RewardManager

Computes modular reward terms and combines them with weights.

**Features:**
- Composable reward terms
- Per-term weights
- Episodic reward tracking
- Optional reward normalization/clipping

**Example:**
```python
from mujoco_playground.manager.reward_manager import (
    RewardManager,
    RewardTerm,
    reward_velocity_tracking,
    reward_energy_penalty,
)

reward_config = {
    "reward_terms": {
        "tracking": RewardTerm(
            func=reward_velocity_tracking,
            weight=2.0,
        ),
        "energy": RewardTerm(
            func=reward_energy_penalty,
            weight=-0.01,
        ),
    }
}

reward_manager = RewardManager()
reward_state = reward_manager.init_state(rng, num_envs, reward_config)
reward_state, (total_reward, term_rewards) = reward_manager.compute(
    reward_state, env_state, reward_config, action
)
```

**Common Reward Functions:**
- `reward_alive`: Staying alive bonus
- `reward_velocity_tracking`: Track velocity commands
- `reward_energy_penalty`: Penalize energy consumption
- `reward_orientation_penalty`: Penalize bad orientation
- `reward_smoothness`: Penalize jerky motions
- `reward_joint_velocity_penalty`: Penalize high joint velocities

### TerminationManager

Checks termination conditions and manages episode resets.

**Features:**
- Multiple termination conditions
- Distinction between termination (failure) and truncation (timeout)
- Episode length tracking
- Per-term termination flags

**Example:**
```python
from mujoco_playground.manager.termination_manager import (
    TerminationManager,
    TerminationTerm,
    termination_bad_orientation,
    termination_height_limit,
)

termination_config = {
    "termination_terms": {
        "orientation": TerminationTerm(
            func=termination_bad_orientation,
            time_out=False,  # Counts as failure
            params={"threshold": 0.5},
        ),
        "height": TerminationTerm(
            func=termination_height_limit,
            params={"min_height": 0.2, "max_height": 2.0},
        ),
    },
    "episode_length": 1000,
}

termination_manager = TerminationManager()
termination_state = termination_manager.init_state(rng, num_envs, termination_config)
termination_state, term_info = termination_manager.compute(
    termination_state, env_state, termination_config
)

done = term_info["done"]
terminated = term_info["terminated"]  # Failed
truncated = term_info["truncated"]    # Timeout
```

**Common Termination Functions:**
- `termination_bad_orientation`: Check if robot tipped over
- `termination_height_limit`: Check if height is valid
- `termination_joint_limit`: Check joint limits
- `termination_velocity_limit`: Check velocity limits
- `termination_nan_check`: Check for NaN/Inf values

### ActionManager

Processes raw policy actions into actuator commands.

**Features:**
- Action clipping and scaling
- Multiple action spaces (position, velocity, torque)
- Action history tracking
- Custom processing functions

**Example:**
```python
from mujoco_playground.manager.action_manager import ActionManager

action_config = {
    "action_dim": 12,
    "action_space": "position",  # or "velocity", "torque"
    "action_scale": 0.5,
    "action_clip": (-1.0, 1.0),
    "default_pose": default_joint_positions,
}

action_manager = ActionManager()
action_state = action_manager.init_state(rng, num_envs, action_config)
action_state, ctrl = action_manager.compute(
    action_state, env_state, action_config, action
)
```

### CommandManager

Generates and tracks command signals for the robot to follow.

**Features:**
- Random command generation
- Automatic resampling at intervals
- Command ranges and constraints
- Custom command generation functions

**Example:**
```python
from mujoco_playground.manager.command_manager import (
    CommandManager,
    command_velocity_tracking,
)

command_config = {
    "command_func": command_velocity_tracking,
    "command_dim": 3,  # (vx, vy, wz)
    "resample_interval": 500,
    "command_ranges": {
        "lin_vel_x": (-1.0, 1.0),
        "lin_vel_y": (-0.5, 0.5),
        "ang_vel_z": (-1.0, 1.0),
    },
}

command_manager = CommandManager()
command_state = command_manager.init_state(rng, num_envs, command_config)
command_state, command = command_manager.compute(
    command_state, env_state, command_config, rng
)
```

**Common Command Functions:**
- `command_velocity_tracking`: Random velocity targets
- `command_position_tracking`: Random position targets
- `command_heading_tracking`: Random heading angles
- `command_zero`: Zero commands for testing

### EventManager

Handles environment events and state modifications.

**Features:**
- Interval-based events
- Condition-based triggers
- Domain randomization
- Environment state modifications

**Example:**
```python
from mujoco_playground.manager.event_manager import (
    EventManager,
    EventTerm,
    event_push_robot,
)

event_config = {
    "event_terms": {
        "push": EventTerm(
            func=event_push_robot,
            mode="interval",
            interval=1000,  # Every 1000 steps
            params={"push_force_range": (-50.0, 50.0)},
        ),
    }
}

event_manager = EventManager()
event_state = event_manager.init_state(rng, num_envs, event_config)
event_state, modified_env_state = event_manager.compute(
    event_state, env_state, event_config, rng
)
```

### CurriculumManager

Manages training progression and difficulty adaptation.

**Features:**
- Linear, threshold, or adaptive progression
- Parameter scheduling
- Performance-based adaptation
- Success rate tracking

**Example:**
```python
from mujoco_playground.manager.curriculum_manager import (
    CurriculumManager,
    CurriculumTerm,
)

curriculum_config = {
    "curriculum_terms": {
        "command_scale": CurriculumTerm(
            param_name="command_scale",
            start_value=0.2,  # Easy
            end_value=1.0,    # Hard
            schedule="linear",
        ),
    },
    "curriculum_mode": "threshold",
    "curriculum_threshold": 0.8,
}

curriculum_manager = CurriculumManager()
curriculum_state = curriculum_manager.init_state(rng, num_envs, curriculum_config)
curriculum_state, updated_config = curriculum_manager.compute(
    curriculum_state, {"success": success_flags}, curriculum_config
)
```

## Creating Custom Managers

You can create custom managers by inheriting from `ManagerBase`:

```python
from mujoco_playground.manager.manager_base import ManagerBase
import flax.struct

@flax.struct.dataclass
class MyManagerState:
    custom_data: jax.Array

class MyCustomManager(ManagerBase):
    def init_state(self, rng, num_envs, config):
        return MyManagerState(
            custom_data=jnp.zeros(num_envs)
        )

    def reset(self, state, env_ids, rng, config):
        # Reset logic
        return state

    def compute(self, state, env_state, config, **kwargs):
        # Computation logic
        new_state = state
        output = compute_something(env_state)
        return new_state, output
```

## Creating Custom Terms

Custom observation/reward/termination functions follow a simple signature:

```python
# Observation term
def my_observation_func(
    env_state: mjx.Data,
    config: config_dict.ConfigDict,
    **kwargs,
) -> jax.Array:
    # Extract observation from env_state
    return observation

# Reward term
def my_reward_func(
    env_state: mjx.Data,
    action: jax.Array,
    config: config_dict.ConfigDict,
    **kwargs,
) -> jax.Array:
    # Compute reward
    return reward

# Termination term
def my_termination_func(
    env_state: mjx.Data,
    config: config_dict.ConfigDict,
    **kwargs,
) -> jax.Array:
    # Check termination condition
    return done_flags
```

## Performance Tips

1. **JIT Compilation**: All manager operations are JIT-compiled by default
2. **VMAP Parallelization**: Use `jax.vmap` for additional parallelization
3. **Batch Size**: Larger batch sizes (4096+) improve GPU utilization
4. **History Buffers**: Minimize history length to reduce memory usage
5. **MJWarp Backend**: Consider MJWarp for potentially better performance

## Examples

See `learning/train_manager_ppo.py` for complete training examples.

## Design Document

See `MANAGER_ARCHITECTURE_DESIGN.md` in the repository root for detailed architecture documentation.

## Comparison with mjlab

| Feature | mjlab | mujoco_playground |
|---------|-------|-------------------|
| Language | Python/JAX | Python/JAX |
| State Management | Object-oriented | Functional |
| Managers | 7 managers | 7 managers |
| JIT Compatible | Yes | Yes |
| VMAP Compatible | Yes | Yes |
| Backend | MuJoCo Warp | MJX + MJWarp |
| Integration | Isaac Lab API | Brax training |

## Contributing

To add new managers or terms:

1. Follow the functional design patterns
2. Ensure JIT and VMAP compatibility
3. Add comprehensive docstrings
4. Include example usage
5. Add tests for new functionality

## License

Same as mujoco_playground main repository.
