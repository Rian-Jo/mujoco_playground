"""Quadruped-specific observation and reward terms for manager-based architecture.

This module provides specialized terms for quadruped robot locomotion based on
the existing Go1 implementation in mujoco_playground. These terms can be used
with the manager-based architecture for any quadruped robot (Go1, Go2, etc.).
"""

from typing import Optional

import jax
import jax.numpy as jnp
from ml_collections import config_dict
from mujoco import mjx


# ==================== Observation Functions ====================

def get_projected_gravity(
    env_state: mjx.Data,
    config: config_dict.ConfigDict,
    **kwargs,
) -> jax.Array:
    """Get gravity vector projected into robot's local frame.

    This is equivalent to the robot's orientation relative to the world frame.
    Useful for understanding if the robot is tilted.

    Args:
        env_state: MJX state
        config: Configuration dict
        **kwargs: Additional arguments

    Returns:
        Projected gravity vector (num_envs, 3)
    """
    # Get quaternion (qpos[3:7] for freejoint)
    quat = env_state.qpos[:, 3:7]

    # World gravity is [0, 0, -1]
    # Transform to local frame using quaternion inverse
    # For quaternion q = [w, x, y, z]:
    # Inverse is q* = [w, -x, -y, -z] (for unit quaternions)

    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]

    # Rotate [0, 0, -1] by quaternion
    # This gives us the up vector in local frame
    gx = 2 * (x * z - w * y)
    gy = 2 * (y * z + w * x)
    gz = 1 - 2 * (x * x + y * y)

    # Stack into projected gravity
    projected_gravity = jnp.stack([gx, gy, -gz], axis=-1)

    return projected_gravity


def get_feet_positions(
    env_state: mjx.Data,
    config: config_dict.ConfigDict,
    feet_site_ids: Optional[jax.Array] = None,
    **kwargs,
) -> jax.Array:
    """Get positions of feet sites.

    Args:
        env_state: MJX state
        config: Configuration dict
        feet_site_ids: Site IDs for feet (if None, uses config)
        **kwargs: Additional arguments

    Returns:
        Feet positions (num_envs, num_feet * 3)
    """
    if feet_site_ids is None:
        # Default to 4 feet for quadruped
        # TODO: Get from config or model
        feet_site_ids = jnp.array([0, 1, 2, 3])  # Placeholder

    feet_pos = env_state.site_xpos[:, feet_site_ids]
    return feet_pos.reshape(feet_pos.shape[0], -1)


def get_feet_contact(
    env_state: mjx.Data,
    config: config_dict.ConfigDict,
    feet_sensor_ids: Optional[jax.Array] = None,
    **kwargs,
) -> jax.Array:
    """Get binary contact state for each foot.

    Args:
        env_state: MJX state
        config: Configuration dict
        feet_sensor_ids: Sensor IDs for feet contact
        **kwargs: Additional arguments

    Returns:
        Contact state (num_envs, num_feet)
    """
    # TODO: Implement proper contact detection
    # For now, return placeholder
    num_envs = env_state.qpos.shape[0]
    return jnp.zeros((num_envs, 4))


# ==================== Reward Functions ====================

def reward_tracking_lin_vel_xy(
    env_state: mjx.Data,
    action: jax.Array,
    config: config_dict.ConfigDict,
    command_state=None,
    tracking_sigma: float = 0.25,
    **kwargs,
) -> jax.Array:
    """Reward for tracking linear velocity commands in XY plane.

    Uses exponential kernel to reward tracking accuracy.

    Args:
        env_state: MJX state
        action: Current action
        config: Configuration dict
        command_state: Command manager state
        tracking_sigma: Sigma for exponential kernel
        **kwargs: Additional arguments

    Returns:
        Tracking reward (num_envs,)
    """
    if command_state is None:
        num_envs = env_state.qpos.shape[0]
        return jnp.zeros(num_envs)

    # Get current linear velocity (base frame)
    # TODO: Implement proper local velocity extraction
    # For now, use global velocity as approximation
    current_vel = env_state.qvel[:, :2]  # vx, vy in base

    # Get command velocities (first 2 components)
    command_vel = command_state.current_command[:, :2]

    # Compute tracking error
    lin_vel_error = jnp.sum(jnp.square(command_vel - current_vel), axis=-1)

    # Exponential reward
    reward = jnp.exp(-lin_vel_error / tracking_sigma)

    return reward


def reward_tracking_ang_vel_z(
    env_state: mjx.Data,
    action: jax.Array,
    config: config_dict.ConfigDict,
    command_state=None,
    tracking_sigma: float = 0.25,
    **kwargs,
) -> jax.Array:
    """Reward for tracking angular velocity command (yaw rate).

    Args:
        env_state: MJX state
        action: Current action
        config: Configuration dict
        command_state: Command manager state
        tracking_sigma: Sigma for exponential kernel
        **kwargs: Additional arguments

    Returns:
        Tracking reward (num_envs,)
    """
    if command_state is None:
        num_envs = env_state.qpos.shape[0]
        return jnp.zeros(num_envs)

    # Get current angular velocity (z component)
    current_angvel = env_state.qvel[:, 5]  # wz

    # Get command angular velocity (third component)
    command_angvel = command_state.current_command[:, 2]

    # Compute tracking error
    ang_vel_error = jnp.square(command_angvel - current_angvel)

    # Exponential reward
    reward = jnp.exp(-ang_vel_error / tracking_sigma)

    return reward


def reward_base_height(
    env_state: mjx.Data,
    action: jax.Array,
    config: config_dict.ConfigDict,
    target_height: float = 0.3,
    **kwargs,
) -> jax.Array:
    """Reward for maintaining target base height.

    Args:
        env_state: MJX state
        action: Current action
        config: Configuration dict
        target_height: Target height in meters
        **kwargs: Additional arguments

    Returns:
        Height reward (num_envs,)
    """
    # Get base height (z position)
    height = env_state.qpos[:, 2]

    # Penalize deviation from target
    height_error = jnp.square(height - target_height)
    reward = jnp.exp(-height_error / 0.1)  # Sharp penalty for deviation

    return reward


def reward_upright_orientation(
    env_state: mjx.Data,
    action: jax.Array,
    config: config_dict.ConfigDict,
    **kwargs,
) -> jax.Array:
    """Reward for maintaining upright orientation.

    Penalizes roll and pitch deviations.

    Args:
        env_state: MJX state
        action: Current action
        config: Configuration dict
        **kwargs: Additional arguments

    Returns:
        Orientation reward (num_envs,)
    """
    # Get quaternion
    quat = env_state.qpos[:, 3:7]

    # Compute up vector z-component
    # For upright orientation, this should be close to 1
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    up_z = 1 - 2 * (x * x + y * y)

    # Reward for being upright
    reward = jnp.exp(5.0 * (up_z - 1.0))  # Sharp penalty for tilting

    return reward


def reward_lin_vel_z_penalty(
    env_state: mjx.Data,
    action: jax.Array,
    config: config_dict.ConfigDict,
    **kwargs,
) -> jax.Array:
    """Penalty for excessive vertical velocity.

    Args:
        env_state: MJX state
        action: Current action
        config: Configuration dict
        **kwargs: Additional arguments

    Returns:
        Vertical velocity penalty (num_envs,) - negative values
    """
    # Get vertical velocity
    vel_z = env_state.qvel[:, 2]

    # L2 penalty
    penalty = -jnp.square(vel_z)

    return penalty


def reward_ang_vel_xy_penalty(
    env_state: mjx.Data,
    action: jax.Array,
    config: config_dict.ConfigDict,
    **kwargs,
) -> jax.Array:
    """Penalty for excessive roll/pitch rates.

    Args:
        env_state: MJX state
        action: Current action
        config: Configuration dict
        **kwargs: Additional arguments

    Returns:
        Angular velocity penalty (num_envs,) - negative values
    """
    # Get roll/pitch rates
    ang_vel_xy = env_state.qvel[:, 3:5]

    # L2 penalty
    penalty = -jnp.sum(jnp.square(ang_vel_xy), axis=-1)

    return penalty


def reward_feet_clearance(
    env_state: mjx.Data,
    action: jax.Array,
    config: config_dict.ConfigDict,
    max_foot_height: float = 0.1,
    **kwargs,
) -> jax.Array:
    """Reward for appropriate foot clearance during swing phase.

    Penalizes feet that are too high or too low during swing.

    Args:
        env_state: MJX state
        action: Current action
        config: Configuration dict
        max_foot_height: Maximum desired foot height
        **kwargs: Additional arguments

    Returns:
        Clearance penalty (num_envs,) - negative values
    """
    # TODO: Implement proper feet clearance calculation
    # Requires feet position tracking and swing phase detection
    num_envs = env_state.qpos.shape[0]
    return jnp.zeros(num_envs)


def reward_feet_air_time(
    env_state: mjx.Data,
    action: jax.Array,
    config: config_dict.ConfigDict,
    command_state=None,
    min_air_time: float = 0.1,
    **kwargs,
) -> jax.Array:
    """Reward for maintaining appropriate air time during gait.

    Encourages dynamic gaits with proper swing phases.

    Args:
        env_state: MJX state
        action: Current action
        config: Configuration dict
        command_state: Command state
        min_air_time: Minimum desired air time
        **kwargs: Additional arguments

    Returns:
        Air time reward (num_envs,)
    """
    # TODO: Implement air time tracking
    # Requires contact state tracking over time
    num_envs = env_state.qpos.shape[0]
    return jnp.zeros(num_envs)


def reward_feet_slip(
    env_state: mjx.Data,
    action: jax.Array,
    config: config_dict.ConfigDict,
    command_state=None,
    **kwargs,
) -> jax.Array:
    """Penalty for foot slipping during stance phase.

    Penalizes feet that slide when they should be in contact.

    Args:
        env_state: MJX state
        action: Current action
        config: Configuration dict
        command_state: Command state
        **kwargs: Additional arguments

    Returns:
        Slip penalty (num_envs,) - negative values
    """
    # TODO: Implement slip detection
    # Requires feet velocity and contact state
    num_envs = env_state.qpos.shape[0]
    return jnp.zeros(num_envs)


def reward_joint_limits(
    env_state: mjx.Data,
    action: jax.Array,
    config: config_dict.ConfigDict,
    soft_limit_factor: float = 0.95,
    skip_dofs: int = 7,
    **kwargs,
) -> jax.Array:
    """Penalty for approaching joint limits.

    Uses soft limits to prevent joints from hitting hard limits.

    Args:
        env_state: MJX state
        action: Current action
        config: Configuration dict
        soft_limit_factor: Fraction of range to use as soft limit
        skip_dofs: Number of DOFs to skip (freejoint)
        **kwargs: Additional arguments

    Returns:
        Joint limit penalty (num_envs,) - negative values
    """
    # Get joint positions
    joint_pos = env_state.qpos[:, skip_dofs:]

    # TODO: Get joint limits from model
    # For now, assume symmetric limits [-2, 2] for all joints
    lower_limit = -2.0 * soft_limit_factor
    upper_limit = 2.0 * soft_limit_factor

    # Compute violations
    out_of_limits = -jnp.clip(joint_pos - lower_limit, None, 0.0)
    out_of_limits += jnp.clip(joint_pos - upper_limit, 0.0, None)

    # Sum penalties
    penalty = -jnp.sum(out_of_limits, axis=-1)

    return penalty


def reward_default_pose(
    env_state: mjx.Data,
    action: jax.Array,
    config: config_dict.ConfigDict,
    default_pose: Optional[jax.Array] = None,
    skip_dofs: int = 7,
    **kwargs,
) -> jax.Array:
    """Reward for staying close to default pose.

    Encourages natural, stable joint configurations.

    Args:
        env_state: MJX state
        action: Current action
        config: Configuration dict
        default_pose: Default joint positions
        skip_dofs: Number of DOFs to skip (freejoint)
        **kwargs: Additional arguments

    Returns:
        Pose reward (num_envs,)
    """
    # Get joint positions
    joint_pos = env_state.qpos[:, skip_dofs:]

    if default_pose is None:
        # Use zero as default
        default_pose = jnp.zeros_like(joint_pos[0])

    # Compute weighted distance
    # Weight hip joints more than knee/ankle
    weight = jnp.array([1.0, 1.0, 0.1] * 4)  # Assuming 3 joints per leg

    pose_error = jnp.sum(jnp.square(joint_pos - default_pose) * weight, axis=-1)
    reward = jnp.exp(-pose_error)

    return reward


def reward_stand_still(
    env_state: mjx.Data,
    action: jax.Array,
    config: config_dict.ConfigDict,
    command_state=None,
    default_pose: Optional[jax.Array] = None,
    skip_dofs: int = 7,
    **kwargs,
) -> jax.Array:
    """Penalty for moving when command is zero (stand still).

    Args:
        env_state: MJX state
        action: Current action
        config: Configuration dict
        command_state: Command state
        default_pose: Default joint positions
        skip_dofs: Number of DOFs to skip
        **kwargs: Additional arguments

    Returns:
        Stand still penalty (num_envs,) - negative when command is zero
    """
    if command_state is None:
        num_envs = env_state.qpos.shape[0]
        return jnp.zeros(num_envs)

    # Check if command is near zero
    command_norm = jnp.linalg.norm(command_state.current_command, axis=-1)
    is_zero_command = command_norm < 0.01

    # Get joint positions
    joint_pos = env_state.qpos[:, skip_dofs:]

    if default_pose is None:
        default_pose = jnp.zeros_like(joint_pos[0])

    # Penalty for deviation from default when command is zero
    deviation = jnp.sum(jnp.abs(joint_pos - default_pose), axis=-1)
    penalty = -deviation * is_zero_command

    return penalty


def reward_torque_penalty(
    env_state: mjx.Data,
    action: jax.Array,
    config: config_dict.ConfigDict,
    **kwargs,
) -> jax.Array:
    """Penalty for high torques.

    Encourages energy-efficient motions.

    Args:
        env_state: MJX state
        action: Current action
        config: Configuration dict
        **kwargs: Additional arguments

    Returns:
        Torque penalty (num_envs,) - negative values
    """
    # Get actuator forces (torques)
    torques = env_state.actuator_force

    # Combined L1 and L2 penalty
    l2_penalty = jnp.sqrt(jnp.sum(jnp.square(torques), axis=-1))
    l1_penalty = jnp.sum(jnp.abs(torques), axis=-1)
    penalty = -(l2_penalty + l1_penalty)

    return penalty
