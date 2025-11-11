# Manager-Based Task Configurations

This directory contains complete task configurations for the manager-based architecture. Each configuration defines observation terms, reward terms, termination conditions, commands, events, and curriculum for a specific robot and task.

## Available Configurations

### Go2 Velocity Tracking

**File**: `go2_velocity.py`

**Description**: Unitree Go2 quadruped performing velocity command tracking on flat or rough terrain.

**Tasks**:
- `flat`: Velocity tracking on flat terrain
- `rough`: Velocity tracking on rough terrain with obstacles

**Features**:
- 12-DOF quadruped control (3 joints per leg)
- 3D velocity command tracking (vx, vy, wz)
- Modular reward structure (tracking, energy, smoothness, etc.)
- Automatic command resampling
- Push disturbance events
- Curriculum learning support

**Quick Start**:
```python
from mujoco_playground.manager.configs.go2_velocity import create_go2_velocity_env
import jax

# Create environment
env = create_go2_velocity_env(task="flat", num_envs=4096)

# Reset and step
rng = jax.random.PRNGKey(0)
state = env.reset(rng)
action = jax.random.uniform(rng, (4096, 12), minval=-1.0, maxval=1.0)
state = env.step(state, action)

# Access results
observations = state.info["observations"]
reward = state.info["reward"]
done = state.info["done"]
command = state.info["command"]
```

**Training**:
```bash
# Flat terrain
python learning/train_go2_velocity.py --task flat --num_envs 4096

# Rough terrain
python learning/train_go2_velocity.py --task rough --num_envs 8192 --use_wandb

# Debug mode
python learning/train_go2_velocity.py --task flat --debug
```

## Configuration Structure

Each configuration file should provide:

### 1. **Default Configuration Function**
```python
def get_{robot}_{task}_config() -> config_dict.ConfigDict:
    """Get complete manager-based configuration."""
    return config_dict.create(
        # Environment settings
        xml_path=...,
        num_envs=...,
        backend=...,

        # Physics settings
        ctrl_dt=...,
        sim_dt=...,

        # Action settings
        action_dim=...,
        action_space=...,
        action_scale=...,

        # Observation terms
        observation_terms=...,

        # Reward terms
        reward_terms=...,

        # Termination terms
        termination_terms=...,

        # Command settings
        command_func=...,
        command_ranges=...,

        # Event settings
        event_terms=...,

        # Curriculum settings
        curriculum_terms=...,
    )
```

### 2. **Helper Functions**
```python
def get_xml_path(task: str) -> str:
    """Get XML path for task variant."""
    ...

def create_{robot}_{task}_env(**kwargs):
    """Create environment with config."""
    ...
```

### 3. **Task Variants**
- Provide multiple configurations for different task variants
- Example: flat terrain, rough terrain, stairs, etc.

## Observation Terms

Observation terms define what the policy sees. Each term should specify:

```python
observation_terms=config_dict.create(
    policy=config_dict.create(  # For policy network
        term_name=config_dict.create(
            func=observation_function,  # Extraction function
            noise_config={"std": 0.01, "type": "gaussian"},
            clip_range=(-10.0, 10.0),
            scale=1.0,
            history_len=1,
            size=3,  # Dimension of observation
        ),
    ),
    privileged=config_dict.create(  # For value network (optional)
        # Same structure as policy
    ),
)
```

**Common Observation Functions** (from `observation_manager.py` and `quadruped_terms.py`):
- `get_joint_positions`: Joint angles
- `get_joint_velocities`: Joint velocities
- `get_base_linear_velocity`: Base linear velocity
- `get_base_angular_velocity`: Base angular velocity
- `get_base_orientation`: Base orientation quaternion
- `get_projected_gravity`: Gravity vector in robot frame
- `get_last_action`: Previous action
- `get_current_command`: Current command signal
- `get_feet_positions`: Feet positions
- `get_feet_contact`: Feet contact states

## Reward Terms

Reward terms define the objectives. Each term should specify:

```python
reward_terms=config_dict.create(
    term_name=config_dict.create(
        func=reward_function,  # Reward computation function
        weight=1.0,  # Reward weight
        params={},  # Additional parameters
    ),
)
```

**Common Reward Functions** (from `reward_manager.py` and `quadruped_terms.py`):

**Tracking**:
- `reward_velocity_tracking`: Track velocity commands
- `reward_tracking_lin_vel_xy`: Track XY linear velocity
- `reward_tracking_ang_vel_z`: Track yaw rate

**Base Control**:
- `reward_orientation_penalty`: Penalize tilting
- `reward_upright_orientation`: Reward staying upright
- `reward_base_height`: Maintain target height
- `reward_lin_vel_z_penalty`: Penalize vertical velocity
- `reward_ang_vel_xy_penalty`: Penalize roll/pitch rates

**Energy & Smoothness**:
- `reward_alive`: Staying alive bonus
- `reward_energy_penalty`: Penalize energy consumption
- `reward_torque_penalty`: Penalize high torques
- `reward_smoothness`: Penalize jerky motions
- `reward_joint_velocity_penalty`: Penalize high joint velocities

**Constraints**:
- `reward_joint_limits`: Penalize approaching joint limits
- `reward_default_pose`: Stay close to default pose
- `reward_stand_still`: Penalty for moving when command is zero

**Feet (Quadruped)**:
- `reward_feet_clearance`: Appropriate foot height during swing
- `reward_feet_air_time`: Maintain proper air time
- `reward_feet_slip`: Penalize foot slipping

## Termination Terms

Termination terms define failure conditions:

```python
termination_terms=config_dict.create(
    term_name=config_dict.create(
        func=termination_function,  # Check function
        time_out=False,  # If True, counts as truncation not termination
        params={},  # Additional parameters
    ),
)
```

**Common Termination Functions** (from `termination_manager.py`):
- `termination_bad_orientation`: Robot tipped over
- `termination_height_limit`: Height out of bounds
- `termination_joint_limit`: Joint limit exceeded
- `termination_velocity_limit`: Velocity too high
- `termination_nan_check`: NaN/Inf detection

## Command Generation

Commands define the task objectives:

```python
command_func=command_velocity_tracking,  # Generation function
command_dim=3,  # Command dimension
resample_interval=500,  # Steps between resampling
command_ranges=config_dict.create(
    lin_vel_x=(-1.0, 1.0),
    lin_vel_y=(-0.5, 0.5),
    ang_vel_z=(-1.0, 1.0),
),
```

**Common Command Functions** (from `command_manager.py`):
- `command_velocity_tracking`: Random velocity targets
- `command_position_tracking`: Random position targets
- `command_heading_tracking`: Random heading angles
- `command_zero`: Zero commands (for testing)

## Events

Events modify the environment during training:

```python
event_terms=config_dict.create(
    term_name=config_dict.create(
        func=event_function,  # Event handler
        mode="interval",  # "interval", "reset", or "condition"
        interval=1000,  # Steps between triggers
        params={},  # Additional parameters
    ),
)
```

**Common Event Functions** (from `event_manager.py`):
- `event_push_robot`: Apply random push forces
- `event_randomize_friction`: Randomize friction
- `event_randomize_mass`: Randomize link masses
- `event_reset_to_random_pose`: Reset to random configuration

## Curriculum

Curriculum terms adapt difficulty over time:

```python
curriculum_terms=config_dict.create(
    term_name=CurriculumTerm(
        param_name="command_scale",  # Config parameter to modify
        start_value=0.2,  # Easy
        end_value=1.0,  # Hard
        schedule="linear",  # "linear" or "exponential"
    ),
)
```

## Creating a New Configuration

To create a new task configuration:

1. **Create configuration file**: `configs/my_robot_my_task.py`

2. **Import necessary components**:
```python
from ml_collections import config_dict
from mujoco_playground.manager.observation_manager import ObservationTerm, ...
from mujoco_playground.manager.reward_manager import RewardTerm, ...
from mujoco_playground.manager.termination_manager import TerminationTerm, ...
```

3. **Define configuration function**:
```python
def get_my_robot_my_task_config() -> config_dict.ConfigDict:
    return config_dict.create(
        # ... all configuration fields
    )
```

4. **Add helper functions**:
```python
def get_xml_path(variant: str) -> str:
    ...

def create_my_robot_my_task_env(**kwargs):
    from mujoco_playground.manager import ManagerBasedEnv
    config = get_my_robot_my_task_config()
    # ... setup
    return ManagerBasedEnv(config)
```

5. **Create training script**: `learning/train_my_robot_my_task.py`

## Best Practices

1. **Start Simple**: Begin with basic tracking rewards and terminations
2. **Add Incrementally**: Add more sophisticated rewards as needed
3. **Tune Weights**: Reward weights are critical - start conservative
4. **Use Curriculum**: Gradually increase difficulty for better learning
5. **Test Thoroughly**: Test environment with random actions before training
6. **Document Well**: Add docstrings explaining configuration choices

## Examples

See existing configurations for reference:
- `go2_velocity.py`: Complete quadruped locomotion setup
- More configurations coming soon...

## Support

For questions or issues:
- Check the main README: `mujoco_playground/manager/README.md`
- See design document: `MANAGER_ARCHITECTURE_DESIGN.md`
- Open an issue on GitHub
