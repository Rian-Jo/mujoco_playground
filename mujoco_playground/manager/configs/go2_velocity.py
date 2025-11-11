"""Go2 Velocity Tracking task configuration for manager-based architecture.

This module provides a complete configuration for the Unitree Go2 quadruped robot
performing velocity tracking tasks. It's based on:
- mujoco_playground's Go1 implementation
- mjlab's velocity tracking task patterns
- Manager-based modular architecture

The configuration includes observation terms, reward terms, termination conditions,
command generation, and curriculum settings optimized for quadruped locomotion.
"""

from ml_collections import config_dict

from mujoco_playground.manager.observation_manager import ObservationTerm
from mujoco_playground.manager.reward_manager import RewardTerm
from mujoco_playground.manager.termination_manager import TerminationTerm
from mujoco_playground.manager.command_manager import command_velocity_tracking
from mujoco_playground.manager.event_manager import EventTerm, event_push_robot
from mujoco_playground.manager.curriculum_manager import CurriculumTerm

# Import observation functions
from mujoco_playground.manager.observation_manager import (
    get_joint_positions,
    get_joint_velocities,
    get_base_orientation,
    get_base_linear_velocity,
    get_base_angular_velocity,
    get_last_action,
    get_current_command,
)

# Import reward functions
from mujoco_playground.manager.reward_manager import (
    reward_alive,
    reward_energy_penalty,
    reward_velocity_tracking,
    reward_orientation_penalty,
    reward_smoothness,
    reward_joint_velocity_penalty,
)

# Import termination functions
from mujoco_playground.manager.termination_manager import (
    termination_bad_orientation,
    termination_height_limit,
    termination_nan_check,
)


def get_go2_velocity_flat_config() -> config_dict.ConfigDict:
    """Get manager-based configuration for Go2 velocity tracking on flat terrain.

    This configuration is optimized for:
    - Velocity command tracking (linear and angular)
    - Stable quadruped locomotion
    - Energy efficiency
    - Smooth, natural gaits

    Returns:
        ConfigDict with complete manager-based configuration
    """
    config = config_dict.create(
        # ==================== Environment Settings ====================
        # XML path (using Go1 as Go2 is not in mujoco_menagerie yet)
        # Note: Go2 is very similar to Go1, so we use Go1 model as base
        xml_path=None,  # Will be set dynamically
        num_envs=4096,
        backend="mjx",

        # ==================== Physics Settings ====================
        ctrl_dt=0.02,  # 50 Hz control frequency
        sim_dt=0.004,  # 250 Hz simulation frequency
        n_substeps=5,  # ctrl_dt / sim_dt

        # ==================== Action Settings ====================
        action_dim=12,  # 12 joints for quadruped (3 per leg)
        action_space="position",  # Position control
        action_scale=0.5,  # Scale actions to reasonable range
        action_clip=(-1.0, 1.0),  # Clip to [-1, 1]
        default_pose=None,  # Will use "home" keyframe from XML
        history_len=1,  # No action history

        # Joint limits
        soft_joint_pos_limit_factor=0.95,

        # PD controller gains
        Kp=35.0,
        Kd=0.5,

        # ==================== Observation Settings ====================
        skip_freejoint_dofs=7,  # Skip freejoint position (7 DOF)

        observation_terms=config_dict.create(
            # Policy observations (what the policy sees)
            policy=config_dict.create(
                # Base velocities
                base_lin_vel=config_dict.create(
                    func=get_base_linear_velocity,
                    noise_config={"std": 0.1, "type": "gaussian"},
                    clip_range=(-10.0, 10.0),
                    scale=1.0,
                    history_len=1,
                    size=3,  # (vx, vy, vz)
                ),
                base_ang_vel=config_dict.create(
                    func=get_base_angular_velocity,
                    noise_config={"std": 0.2, "type": "gaussian"},
                    clip_range=(-10.0, 10.0),
                    scale=1.0,
                    history_len=1,
                    size=3,  # (wx, wy, wz)
                ),
                # Projected gravity (orientation)
                projected_gravity=config_dict.create(
                    func=get_base_orientation,  # Will need custom function for gravity
                    noise_config={"std": 0.05, "type": "gaussian"},
                    scale=1.0,
                    history_len=1,
                    size=3,  # Projected gravity vector
                ),
                # Joint states
                joint_pos=config_dict.create(
                    func=get_joint_positions,
                    noise_config={"std": 0.03, "type": "gaussian"},
                    clip_range=(-10.0, 10.0),
                    scale=1.0,
                    history_len=1,
                    size=12,  # 12 joints
                ),
                joint_vel=config_dict.create(
                    func=get_joint_velocities,
                    noise_config={"std": 1.5, "type": "gaussian"},
                    clip_range=(-20.0, 20.0),
                    scale=0.1,  # Scale down velocities
                    history_len=1,
                    size=12,  # 12 joints
                ),
                # Previous action
                last_action=config_dict.create(
                    func=get_last_action,
                    scale=1.0,
                    history_len=1,
                    size=12,  # 12 joints
                ),
                # Command
                command=config_dict.create(
                    func=get_current_command,
                    scale=1.0,
                    history_len=1,
                    size=3,  # (vx, vy, wz)
                ),
            ),
            # Privileged observations (for critic/value function)
            privileged=config_dict.create(
                # All policy observations
                base_lin_vel=config_dict.create(
                    func=get_base_linear_velocity,
                    scale=1.0,
                    size=3,
                ),
                base_ang_vel=config_dict.create(
                    func=get_base_angular_velocity,
                    scale=1.0,
                    size=3,
                ),
                joint_pos=config_dict.create(
                    func=get_joint_positions,
                    scale=1.0,
                    size=12,
                ),
                joint_vel=config_dict.create(
                    func=get_joint_velocities,
                    scale=0.1,
                    size=12,
                ),
                # Plus privileged information (no noise)
                # TODO: Add feet contact states, actuator forces, etc.
            ),
        ),
        concatenate_groups=True,
        add_noise=True,

        # ==================== Reward Settings ====================
        reward_terms=config_dict.create(
            # Tracking rewards (primary objectives)
            tracking_lin_vel=config_dict.create(
                func=reward_velocity_tracking,
                weight=1.0,
                params={"velocity_indices": (0, 1)},  # Track vx, vy
            ),
            tracking_ang_vel=config_dict.create(
                func=reward_velocity_tracking,
                weight=0.5,
                params={"velocity_indices": (5,)},  # Track wz
            ),

            # Base orientation rewards
            orientation=config_dict.create(
                func=reward_orientation_penalty,
                weight=-5.0,  # Penalize tilting
            ),

            # Energy and efficiency
            alive=config_dict.create(
                func=reward_alive,
                weight=0.1,
            ),
            energy=config_dict.create(
                func=reward_energy_penalty,
                weight=-0.001,
            ),
            torques=config_dict.create(
                func=reward_energy_penalty,  # Using energy as torque proxy
                weight=-0.0002,
            ),

            # Smoothness and regularization
            action_rate=config_dict.create(
                func=reward_smoothness,
                weight=-0.01,
            ),
            joint_velocity=config_dict.create(
                func=reward_joint_velocity_penalty,
                weight=-0.001,
                params={"skip_dofs": 6},
            ),

            # TODO: Add feet-specific rewards (clearance, air time, slip, etc.)
        ),
        normalize_rewards=False,
        clip_rewards=None,
        tracking_sigma=0.25,  # For velocity tracking reward

        # ==================== Termination Settings ====================
        termination_terms=config_dict.create(
            bad_orientation=config_dict.create(
                func=termination_bad_orientation,
                time_out=False,  # Counts as failure
                params={"threshold": 0.0},  # Up vector z-component must be > 0
            ),
            height_limit=config_dict.create(
                func=termination_height_limit,
                time_out=False,
                params={"min_height": 0.15, "max_height": 1.0},
            ),
            nan_check=config_dict.create(
                func=termination_nan_check,
                time_out=False,
            ),
        ),
        episode_length=1000,  # 20 seconds at 50Hz

        # ==================== Command Settings ====================
        command_func=command_velocity_tracking,
        command_dim=3,  # (vx, vy, wz)
        resample_interval=500,  # Resample every 10 seconds
        resample_on_reset=True,
        command_ranges=config_dict.create(
            lin_vel_x=(-1.5, 1.5),  # Forward/backward velocity
            lin_vel_y=(-0.8, 0.8),  # Lateral velocity
            ang_vel_z=(-1.2, 1.2),  # Yaw rate
        ),
        # Command probability (probability of non-zero command)
        command_prob=config_dict.create(
            lin_vel_x=0.9,
            lin_vel_y=0.25,
            ang_vel_z=0.5,
        ),

        # ==================== Event Settings ====================
        event_terms=config_dict.create(
            # External push disturbances
            push_robot=config_dict.create(
                func=event_push_robot,
                mode="interval",
                interval=1000,  # Every 20 seconds
                params={"push_force_range": (-30.0, 30.0)},
            ),
        ),

        # ==================== Curriculum Settings ====================
        curriculum_terms=config_dict.create(
            command_scale=CurriculumTerm(
                param_name="command_scale",
                start_value=0.3,  # Start with 30% of full command range
                end_value=1.0,  # Progress to full range
                schedule="linear",
            ),
        ),
        curriculum_mode="threshold",
        curriculum_threshold=0.75,  # Increase difficulty when 75% success
        curriculum_length=10_000_000,
        success_buffer_size=100,
        update_interval=1000,

        # ==================== Noise Configuration ====================
        noise_config=config_dict.create(
            level=1.0,  # Global noise level (set to 0.0 to disable)
            scales=config_dict.create(
                joint_pos=0.03,
                joint_vel=1.5,
                gyro=0.2,
                gravity=0.05,
                linvel=0.1,
            ),
        ),

        # ==================== MJX Settings ====================
        impl="jax",
        nconmax=4 * 8192,  # Maximum number of contacts
        njmax=40,  # Maximum number of scalar constraints
    )

    return config


def get_go2_velocity_rough_config() -> config_dict.ConfigDict:
    """Get manager-based configuration for Go2 velocity tracking on rough terrain.

    Similar to flat terrain but with adjustments for rough terrain navigation.

    Returns:
        ConfigDict with complete manager-based configuration for rough terrain
    """
    # Start with flat terrain config
    config = get_go2_velocity_flat_config()

    # Modify for rough terrain
    config.nconmax = 100 * 8192  # More contacts for rough terrain
    config.njmax = 12 + 100 * 4  # More constraints

    # Adjust command ranges for rough terrain (slower)
    config.command_ranges.lin_vel_x = (-1.0, 1.0)
    config.command_ranges.lin_vel_y = (-0.5, 0.5)
    config.command_ranges.ang_vel_z = (-0.8, 0.8)

    # Adjust reward weights for rough terrain
    config.reward_terms.orientation.weight = -3.0  # Less strict orientation
    config.reward_terms.energy.weight = -0.002  # More energy expected

    # Adjust termination (be more forgiving)
    config.termination_terms.bad_orientation.params.threshold = -0.2

    return config


# ==================== Helper Functions ====================

def get_xml_path(task: str = "flat") -> str:
    """Get XML path for Go2 task.

    Args:
        task: Task type ("flat" or "rough")

    Returns:
        Path to XML file
    """
    from mujoco_playground._src import mjx_env
    from mujoco_playground._src.locomotion.go1 import go1_constants

    # Use Go1 XML as Go2 is not in mujoco_menagerie yet
    # Go2 is mechanically very similar to Go1
    return go1_constants.task_to_xml(
        "flat_terrain" if task == "flat" else "rough_terrain"
    ).as_posix()


def create_go2_velocity_env(
    task: str = "flat",
    num_envs: int = 4096,
    backend: str = "mjx",
    **config_overrides,
):
    """Create a Go2 velocity tracking environment with manager architecture.

    Args:
        task: Task type ("flat" or "rough")
        num_envs: Number of parallel environments
        backend: Physics backend ("mjx" or "warp")
        **config_overrides: Additional config overrides

    Returns:
        ManagerBasedEnv instance configured for Go2 velocity tracking

    Example:
        >>> env = create_go2_velocity_env(task="flat", num_envs=4096)
        >>> rng = jax.random.PRNGKey(0)
        >>> state = env.reset(rng)
        >>> action = jnp.zeros((4096, 12))
        >>> state = env.step(state, action)
    """
    from mujoco_playground.manager import ManagerBasedEnv

    # Get config
    if task == "flat":
        config = get_go2_velocity_flat_config()
    elif task == "rough":
        config = get_go2_velocity_rough_config()
    else:
        raise ValueError(f"Unknown task: {task}. Must be 'flat' or 'rough'")

    # Set XML path
    config.xml_path = get_xml_path(task)
    config.num_envs = num_envs
    config.backend = backend

    # Apply overrides
    config.update(config_overrides)

    # Separate manager configs
    config.action_config = config.copy()
    config.observation_config = config.copy()
    config.reward_config = config.copy()
    config.termination_config = config.copy()
    config.command_config = config.copy()
    config.event_config = config.copy()
    config.curriculum_config = config.copy()

    # Create environment
    env = ManagerBasedEnv(config)

    return env
