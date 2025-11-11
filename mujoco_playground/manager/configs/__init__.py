"""Manager-based task configurations.

This module provides complete configurations for various robot tasks using the
manager-based architecture. Each configuration includes observation terms,
reward terms, termination conditions, commands, events, and curriculum settings.

Available Configurations:
- go2_velocity: Unitree Go2 velocity tracking (flat and rough terrain)

Example:
    >>> from mujoco_playground.manager.configs.go2_velocity import create_go2_velocity_env
    >>> env = create_go2_velocity_env(task="flat", num_envs=4096)
"""

from mujoco_playground.manager.configs.go2_velocity import (
    get_go2_velocity_flat_config,
    get_go2_velocity_rough_config,
    create_go2_velocity_env,
    get_xml_path as get_go2_xml_path,
)

__all__ = [
    "get_go2_velocity_flat_config",
    "get_go2_velocity_rough_config",
    "create_go2_velocity_env",
    "get_go2_xml_path",
]
