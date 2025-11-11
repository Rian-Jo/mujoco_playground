# MuJoCo Playground Manager Architecture Design

## Executive Summary

This document outlines the design for a manager-based training architecture for mujoco_playground, inspired by mjlab's manager pattern but fully adapted for JAX/Flax/Brax/MJX/MJWarp compatibility with functional programming paradigms.

## Architecture Comparison

### Current mujoco_playground Architecture
- **Registry-based** environment discovery
- **Functional JAX-first** design with immutable states
- **Wrapper composition** for training transformations
- **Three-layer configuration**: env → RL → runtime overrides
- **No centralized orchestration** - training scripts handle everything inline

### mjlab Manager Architecture
- **7 specialized managers**: Action, Observation, Reward, Termination, Command, Event, Curriculum
- **Base manager class** with lifecycle methods (reset, compute, etc.)
- **Term-based configuration** for modular composition
- **ManagerBasedRlEnv** orchestrates all managers
- **Flexible composition** through configuration dictionaries

### Proposed Hybrid Architecture

Combine the best of both worlds:
- **JAX-native managers** using pure functions and immutable state
- **Modular term system** inspired by mjlab
- **Registry integration** with existing mujoco_playground patterns
- **Backward compatible** with existing environments
- **JIT-compilable** manager operations for performance

---

## Core Design Principles

### 1. **Functional JAX-First**
```python
# All manager operations are pure functions
state, obs = observation_manager.compute(state, config)
state, reward = reward_manager.compute(state, action, config)
state, done = termination_manager.compute(state, config)
```

### 2. **Immutable State Management**
```python
@flax.struct.dataclass
class ManagerState:
    """Immutable state container for all managers."""
    obs_state: ObservationState
    reward_state: RewardState
    termination_state: TerminationState
    command_state: CommandState
    event_state: EventState
    curriculum_state: CurriculumState
    action_state: ActionState
```

### 3. **JIT-Compatible Operations**
```python
@jax.jit
def manager_step(manager_state, env_state, action, config):
    """Fully JIT-compiled manager step."""
    # All operations are functional and JIT-compilable
    ...
```

### 4. **VMAP-Ready for Parallelization**
```python
# All managers support automatic vectorization
vmapped_compute = jax.vmap(manager.compute, in_axes=(0, 0, None))
states, outputs = vmapped_compute(states, inputs, config)
```

---

## Manager Architecture

### Base Manager Interface

```python
class ManagerBase(abc.ABC):
    """Base class for all managers.

    Key differences from mjlab:
    - Stateless/functional design (no self._env)
    - Pure functions that take state as input
    - Return updated state + outputs
    - JIT and VMAP compatible
    """

    @abc.abstractmethod
    def init_state(self, rng: jax.Array, config: ConfigDict) -> Any:
        """Initialize manager state."""
        pass

    @abc.abstractmethod
    def reset(
        self,
        state: Any,
        env_ids: jax.Array,
        rng: jax.Array
    ) -> Any:
        """Reset manager state for specified environments."""
        pass

    @abc.abstractmethod
    def compute(
        self,
        state: Any,
        env_state: mjx.Data,
        config: ConfigDict,
        **kwargs
    ) -> Tuple[Any, Any]:
        """Compute manager output. Returns (new_state, output)."""
        pass
```

### 1. ObservationManager

**Responsibilities:**
- Collect observations from MJX state
- Apply noise, clipping, scaling
- Manage observation history buffers
- Support privileged observations
- Group observations (policy vs value)

**JAX Implementation:**
```python
@flax.struct.dataclass
class ObservationState:
    history_buffer: jax.Array  # (num_envs, history_len, obs_dim)
    step_counter: jax.Array    # (num_envs,)

@flax.struct.dataclass
class ObservationTerm:
    """Configuration for a single observation term."""
    func: Callable  # Observation extraction function
    noise_config: Optional[Dict[str, float]]
    clip_range: Optional[Tuple[float, float]]
    scale: float = 1.0
    history_len: int = 1
    group: str = "policy"  # "policy", "value", or custom

class ObservationManager(ManagerBase):
    """Manages observation collection and processing.

    Features:
    - Modular observation terms
    - Noise injection (JAX-native random)
    - Observation history via circular buffers
    - Grouped observations (policy/value)
    - JIT-compiled computation
    """

    def compute(
        self,
        state: ObservationState,
        env_state: mjx.Data,
        config: ConfigDict,
        rng: jax.Array
    ) -> Tuple[ObservationState, Dict[str, jax.Array]]:
        """Compute all observation groups."""
        observations = {}

        for group_name, terms in config.observation_terms.items():
            group_obs = []

            for term_name, term_config in terms.items():
                # Extract raw observation
                raw_obs = term_config.func(env_state, config)

                # Apply noise
                if term_config.noise_config:
                    rng, noise_key = jax.random.split(rng)
                    noise = jax.random.normal(noise_key, raw_obs.shape)
                    noise *= term_config.noise_config["std"]
                    raw_obs += noise

                # Clip and scale
                if term_config.clip_range:
                    raw_obs = jnp.clip(raw_obs, *term_config.clip_range)
                raw_obs *= term_config.scale

                group_obs.append(raw_obs)

            # Concatenate group observations
            observations[group_name] = jnp.concatenate(group_obs, axis=-1)

        return state, observations
```

### 2. RewardManager

**Responsibilities:**
- Compute modular reward terms
- Apply weights and combine
- Track per-term rewards for logging
- Support shaped rewards
- Reset episodic counters

**JAX Implementation:**
```python
@flax.struct.dataclass
class RewardState:
    episode_rewards: jax.Array  # (num_envs, num_terms)
    step_rewards: jax.Array     # (num_envs, num_terms)

@flax.struct.dataclass
class RewardTerm:
    """Configuration for a single reward term."""
    func: Callable  # Reward computation function
    weight: float = 1.0
    params: Dict[str, Any] = None

class RewardManager(ManagerBase):
    """Manages modular reward computation.

    Features:
    - Composable reward terms
    - Per-term weights
    - Episodic tracking
    - JIT-compiled computation
    """

    def compute(
        self,
        state: RewardState,
        env_state: mjx.Data,
        action: jax.Array,
        config: ConfigDict
    ) -> Tuple[RewardState, Tuple[jax.Array, Dict[str, jax.Array]]]:
        """Compute weighted sum of all reward terms."""
        total_reward = jnp.zeros(config.num_envs)
        term_rewards = {}

        for term_name, term_config in config.reward_terms.items():
            # Compute term reward
            params = term_config.params or {}
            term_reward = term_config.func(
                env_state, action, config, **params
            )

            # Apply weight and dt
            weighted_reward = term_reward * term_config.weight * config.dt

            total_reward += weighted_reward
            term_rewards[term_name] = term_reward

        # Update state
        new_state = state.replace(
            step_rewards=jnp.stack([term_rewards[k] for k in config.reward_terms]),
            episode_rewards=state.episode_rewards + jnp.stack([
                term_rewards[k] for k in config.reward_terms
            ])
        )

        return new_state, (total_reward, term_rewards)
```

### 3. TerminationManager

**Responsibilities:**
- Check termination conditions
- Track time limits
- Handle multiple termination terms
- Support both done and truncated flags

**JAX Implementation:**
```python
@flax.struct.dataclass
class TerminationState:
    episode_length: jax.Array  # (num_envs,)
    terminated: jax.Array      # (num_envs,) bool
    truncated: jax.Array       # (num_envs,) bool

@flax.struct.dataclass
class TerminationTerm:
    """Configuration for a termination condition."""
    func: Callable  # Termination check function
    time_out: bool = False  # If True, counts as truncation not termination

class TerminationManager(ManagerBase):
    """Manages episode termination conditions.

    Features:
    - Multiple termination conditions
    - Time limit handling
    - Distinction between done/truncated
    - JIT-compiled checks
    """

    def compute(
        self,
        state: TerminationState,
        env_state: mjx.Data,
        config: ConfigDict
    ) -> Tuple[TerminationState, Dict[str, jax.Array]]:
        """Check all termination conditions."""
        terminated = jnp.zeros(config.num_envs, dtype=bool)
        truncated = jnp.zeros(config.num_envs, dtype=bool)

        # Check time limit
        new_length = state.episode_length + 1
        time_limit_exceeded = new_length >= config.episode_length
        truncated = jnp.logical_or(truncated, time_limit_exceeded)

        # Check termination terms
        for term_name, term_config in config.termination_terms.items():
            term_done = term_config.func(env_state, config)

            if term_config.time_out:
                truncated = jnp.logical_or(truncated, term_done)
            else:
                terminated = jnp.logical_or(terminated, term_done)

        done = jnp.logical_or(terminated, truncated)

        new_state = state.replace(
            episode_length=new_length,
            terminated=terminated,
            truncated=truncated
        )

        return new_state, {
            "done": done,
            "terminated": terminated,
            "truncated": truncated
        }
```

### 4. ActionManager

**Responsibilities:**
- Process raw actions from policy
- Apply action scaling/clipping
- Convert to actuator commands
- Support different action spaces (joint positions, velocities, torques)

**JAX Implementation:**
```python
@flax.struct.dataclass
class ActionState:
    last_action: jax.Array  # (num_envs, action_dim)
    action_history: jax.Array  # (num_envs, history_len, action_dim)

class ActionManager(ManagerBase):
    """Manages action processing and application.

    Features:
    - Action scaling/clipping
    - Action history tracking
    - PD controller integration
    - Multiple action spaces
    """

    def compute(
        self,
        state: ActionState,
        action: jax.Array,
        env_state: mjx.Data,
        config: ConfigDict
    ) -> Tuple[ActionState, jax.Array]:
        """Process raw actions into actuator commands."""
        # Clip actions
        action = jnp.clip(action, -1.0, 1.0)

        # Scale actions
        scaled_action = action * config.action_scale

        # Convert to actuator command based on action_space
        if config.action_space == "position":
            # Position control
            ctrl = config.default_pose + scaled_action
        elif config.action_space == "velocity":
            # Velocity control
            ctrl = scaled_action
        elif config.action_space == "torque":
            # Direct torque control
            ctrl = scaled_action
        else:
            raise ValueError(f"Unknown action space: {config.action_space}")

        new_state = state.replace(last_action=action)

        return new_state, ctrl
```

### 5. CommandManager

**Responsibilities:**
- Generate command signals (velocity commands, target poses, etc.)
- Resample commands on episode reset or at intervals
- Track command achievement

**JAX Implementation:**
```python
@flax.struct.dataclass
class CommandState:
    current_command: jax.Array  # (num_envs, command_dim)
    command_counter: jax.Array  # (num_envs,)
    resample_time: jax.Array    # (num_envs,)

@flax.struct.dataclass
class CommandTerm:
    """Configuration for a command generator."""
    func: Callable  # Command generation function
    resample_interval: int = 500  # Steps between resampling

class CommandManager(ManagerBase):
    """Manages command generation and tracking.

    Features:
    - Multiple command types
    - Automatic resampling
    - Command tracking for rewards
    - Curriculum support
    """

    def compute(
        self,
        state: CommandState,
        env_state: mjx.Data,
        config: ConfigDict,
        rng: jax.Array
    ) -> Tuple[CommandState, jax.Array]:
        """Update commands, resampling if needed."""
        # Check if resample needed
        should_resample = state.command_counter >= config.command_resample_interval

        # Generate new commands
        rng, cmd_key = jax.random.split(rng)
        new_commands = config.command_func(cmd_key, config)

        # Conditionally update
        current_command = jnp.where(
            should_resample[:, None],
            new_commands,
            state.current_command
        )

        command_counter = jnp.where(
            should_resample,
            0,
            state.command_counter + 1
        )

        new_state = state.replace(
            current_command=current_command,
            command_counter=command_counter
        )

        return new_state, current_command
```

### 6. EventManager

**Responsibilities:**
- Trigger events at specified intervals or conditions
- Handle randomization events
- Modify environment state (domain randomization, object spawning, etc.)

**JAX Implementation:**
```python
@flax.struct.dataclass
class EventState:
    event_counters: jax.Array  # (num_events, num_envs)
    last_trigger_step: jax.Array  # (num_events, num_envs)

@flax.struct.dataclass
class EventTerm:
    """Configuration for an event."""
    func: Callable  # Event handler function
    mode: str = "interval"  # "interval", "reset", "condition"
    interval: int = 1000  # For interval mode
    condition_func: Optional[Callable] = None  # For condition mode

class EventManager(ManagerBase):
    """Manages environment events and modifications.

    Features:
    - Interval-based events
    - Condition-based triggers
    - Domain randomization
    - JIT-compiled event handling
    """

    def compute(
        self,
        state: EventState,
        env_state: mjx.Data,
        config: ConfigDict,
        rng: jax.Array
    ) -> Tuple[EventState, mjx.Data]:
        """Execute events and modify environment state."""
        modified_env_state = env_state

        for event_idx, (event_name, event_config) in enumerate(
            config.event_terms.items()
        ):
            # Check if event should trigger
            should_trigger = False

            if event_config.mode == "interval":
                elapsed = state.event_counters[event_idx]
                should_trigger = elapsed >= event_config.interval
            elif event_config.mode == "condition":
                should_trigger = event_config.condition_func(env_state, config)

            # Execute event if triggered
            if should_trigger:
                rng, event_key = jax.random.split(rng)
                modified_env_state = event_config.func(
                    modified_env_state, event_key, config
                )

        return state, modified_env_state
```

### 7. CurriculumManager

**Responsibilities:**
- Adjust task difficulty over time
- Modify config parameters during training
- Track learning progress metrics

**JAX Implementation:**
```python
@flax.struct.dataclass
class CurriculumState:
    difficulty_level: jax.Array  # (num_envs,)
    success_rate: jax.Array      # (num_envs,)
    curriculum_step: int

@flax.struct.dataclass
class CurriculumTerm:
    """Configuration for curriculum progression."""
    param_name: str  # Config parameter to modify
    start_value: float
    end_value: float
    schedule: str = "linear"  # "linear", "exponential", "threshold"

class CurriculumManager(ManagerBase):
    """Manages curriculum learning progression.

    Features:
    - Parameter scheduling
    - Performance-based adaptation
    - Multiple curriculum strategies
    - JIT-compiled updates
    """

    def compute(
        self,
        state: CurriculumState,
        metrics: Dict[str, jax.Array],
        config: ConfigDict
    ) -> Tuple[CurriculumState, ConfigDict]:
        """Update curriculum and return modified config."""
        # Update success rate
        new_success_rate = jnp.mean(metrics.get("success", 0.0))

        # Compute new difficulty
        if config.curriculum_mode == "threshold":
            # Increase difficulty if success rate above threshold
            should_increase = new_success_rate > config.curriculum_threshold
            new_difficulty = state.difficulty_level + should_increase.astype(float)
            new_difficulty = jnp.clip(new_difficulty, 0.0, 1.0)
        else:
            # Linear progression
            progress = state.curriculum_step / config.curriculum_length
            new_difficulty = jnp.clip(progress, 0.0, 1.0)

        # Modify config based on curriculum
        updated_config = config.copy()
        for term_name, term_config in config.curriculum_terms.items():
            interpolated_value = (
                term_config.start_value +
                new_difficulty * (term_config.end_value - term_config.start_value)
            )
            updated_config[term_config.param_name] = interpolated_value

        new_state = state.replace(
            difficulty_level=new_difficulty,
            success_rate=new_success_rate,
            curriculum_step=state.curriculum_step + 1
        )

        return new_state, updated_config
```

---

## Manager-Based Environment

### Core Environment Class

```python
@flax.struct.dataclass
class ManagerBasedState:
    """Complete state for manager-based environment."""
    env_state: mjx.Data  # MJX physics state
    obs_state: ObservationState
    reward_state: RewardState
    termination_state: TerminationState
    action_state: ActionState
    command_state: CommandState
    event_state: EventState
    curriculum_state: CurriculumState
    info: Dict[str, Any]  # Logging info

class ManagerBasedEnv:
    """Manager-based RL environment for MuJoCo Playground.

    Integrates all managers into a unified training interface.
    Compatible with JAX/Flax/Brax/MJX/MJWarp.
    """

    def __init__(
        self,
        xml_path: str,
        config: ConfigDict,
        backend: str = "mjx"  # "mjx" or "warp"
    ):
        self.xml_path = xml_path
        self.config = config
        self.backend = backend

        # Load MuJoCo model
        self.mj_model = mujoco.MjModel.from_xml_path(xml_path)

        # Create backend-specific model
        if backend == "mjx":
            self.model = mjx.put_model(self.mj_model)
        elif backend == "warp":
            # MJWarp integration
            import warp as wp
            wp.init()
            self.model = mjwarp.put_model(self.mj_model)

        # Initialize managers
        self.obs_manager = ObservationManager()
        self.reward_manager = RewardManager()
        self.termination_manager = TerminationManager()
        self.action_manager = ActionManager()
        self.command_manager = CommandManager()
        self.event_manager = EventManager()
        self.curriculum_manager = CurriculumManager()

    @property
    def observation_space(self) -> gym.Space:
        """Compute observation space from manager config."""
        obs_dims = {}
        for group, terms in self.config.observation_terms.items():
            total_dim = sum(term.dim for term in terms.values())
            obs_dims[group] = gym.spaces.Box(-np.inf, np.inf, (total_dim,))
        return gym.spaces.Dict(obs_dims)

    @property
    def action_space(self) -> gym.Space:
        """Action space from config."""
        return gym.spaces.Box(
            -1.0, 1.0, (self.config.action_dim,), dtype=np.float32
        )

    def reset(self, rng: jax.Array) -> ManagerBasedState:
        """Reset environment and all managers."""
        # Split RNG for each manager
        rngs = jax.random.split(rng, 8)

        # Reset physics
        env_state = self._reset_physics(rngs[0])

        # Initialize all manager states
        obs_state = self.obs_manager.init_state(rngs[1], self.config)
        reward_state = self.reward_manager.init_state(rngs[2], self.config)
        termination_state = self.termination_manager.init_state(
            rngs[3], self.config
        )
        action_state = self.action_manager.init_state(rngs[4], self.config)
        command_state = self.command_manager.init_state(rngs[5], self.config)
        event_state = self.event_manager.init_state(rngs[6], self.config)
        curriculum_state = self.curriculum_manager.init_state(
            rngs[7], self.config
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
            info={}
        )

    def step(
        self,
        state: ManagerBasedState,
        action: jax.Array,
        rng: jax.Array
    ) -> ManagerBasedState:
        """Execute one environment step with all managers."""
        rngs = jax.random.split(rng, 5)

        # 1. Process action
        action_state, ctrl = self.action_manager.compute(
            state.action_state, action, state.env_state, self.config
        )

        # 2. Update commands
        command_state, command = self.command_manager.compute(
            state.command_state, state.env_state, self.config, rngs[0]
        )

        # 3. Step physics
        env_state = self._step_physics(state.env_state, ctrl)

        # 4. Handle events
        event_state, env_state = self.event_manager.compute(
            state.event_state, env_state, self.config, rngs[1]
        )

        # 5. Compute observations
        obs_state, observations = self.obs_manager.compute(
            state.obs_state, env_state, self.config, rngs[2]
        )

        # 6. Compute rewards
        reward_state, (reward, reward_info) = self.reward_manager.compute(
            state.reward_state, env_state, action, self.config
        )

        # 7. Check terminations
        termination_state, termination_info = self.termination_manager.compute(
            state.termination_state, env_state, self.config
        )

        # 8. Update curriculum
        curriculum_state, updated_config = self.curriculum_manager.compute(
            state.curriculum_state,
            {"success": reward_info.get("success", 0.0)},
            self.config
        )

        # 9. Auto-reset terminated environments
        should_reset = termination_info["done"]
        env_state, obs_state = self._auto_reset(
            env_state, obs_state, should_reset, rngs[3]
        )

        # 10. Collect info
        info = {
            "reward": reward,
            "observations": observations,
            **reward_info,
            **termination_info,
            "command": command
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
            info=info
        )

    def _step_physics(self, env_state: mjx.Data, ctrl: jax.Array) -> mjx.Data:
        """Step physics simulation."""
        env_state = env_state.replace(ctrl=ctrl)

        # Sub-stepping for simulation stability
        for _ in range(self.config.n_substeps):
            if self.backend == "mjx":
                env_state = mjx.step(self.model, env_state)
            elif self.backend == "warp":
                env_state = mjwarp.step(self.model, env_state)

        return env_state
```

---

## Training Pipeline

### Manager-Based Training Script

```python
# learning/train_manager_ppo.py

def create_manager_based_env(
    env_name: str,
    config: ConfigDict,
    backend: str = "mjx"
) -> ManagerBasedEnv:
    """Create manager-based environment from registry."""
    # Get base environment config
    base_config = registry.get_default_config(env_name)

    # Override with training config
    base_config.update(config)

    # Get XML path
    xml_path = registry.get_xml_path(env_name)

    # Create manager-based env
    env = ManagerBasedEnv(xml_path, base_config, backend)

    return env

def train_manager_ppo(
    env_name: str,
    num_envs: int = 4096,
    num_timesteps: int = 100_000_000,
    backend: str = "mjx",
    **kwargs
):
    """Train using manager-based environment."""
    # Load config
    config = load_manager_config(env_name)
    config.update(kwargs)

    # Create environment
    env = create_manager_based_env(env_name, config, backend)

    # Wrap for Brax training
    env = wrap_for_brax_training(
        env,
        episode_length=config.episode_length,
        num_envs=num_envs
    )

    # Create network
    network_factory = functools.partial(
        ppo_networks.make_ppo_networks,
        policy_hidden_layer_sizes=config.policy_hidden_layers,
        value_hidden_layer_sizes=config.value_hidden_layers
    )

    # Train
    make_inference_fn, params, _ = train_ppo.train(
        environment=env,
        num_timesteps=num_timesteps,
        num_evals=config.num_evals,
        episode_length=config.episode_length,
        network_factory=network_factory,
        **config.ppo_config
    )

    return make_inference_fn, params
```

---

## Configuration System

### Manager Configuration Structure

```python
def get_manager_config(env_name: str) -> ConfigDict:
    """Get manager-based configuration for environment."""
    return config_dict.create(
        # Base environment config
        ctrl_dt=0.02,
        sim_dt=0.004,
        episode_length=1000,
        num_envs=4096,
        action_dim=12,

        # Action manager config
        action_space="position",
        action_scale=0.5,
        action_clip=(-1.0, 1.0),

        # Observation manager config
        observation_terms=config_dict.create(
            policy=config_dict.create(
                joint_pos=ObservationTerm(
                    func=get_joint_positions,
                    noise_config={"std": 0.01},
                    clip_range=(-10.0, 10.0),
                    scale=1.0,
                    history_len=1
                ),
                joint_vel=ObservationTerm(
                    func=get_joint_velocities,
                    noise_config={"std": 0.1},
                    clip_range=(-20.0, 20.0),
                    scale=0.1,
                    history_len=1
                ),
                imu=ObservationTerm(
                    func=get_imu_data,
                    noise_config={"std": 0.05},
                    scale=1.0
                ),
                command=ObservationTerm(
                    func=get_command,
                    scale=1.0
                )
            ),
            privileged=config_dict.create(
                joint_pos=ObservationTerm(
                    func=get_joint_positions,
                    scale=1.0
                ),
                joint_vel=ObservationTerm(
                    func=get_joint_velocities,
                    scale=0.1
                ),
                contact_forces=ObservationTerm(
                    func=get_contact_forces,
                    scale=0.01
                )
            )
        ),

        # Reward manager config
        reward_terms=config_dict.create(
            tracking=RewardTerm(
                func=reward_tracking,
                weight=1.0
            ),
            energy=RewardTerm(
                func=reward_energy,
                weight=-0.01
            ),
            alive=RewardTerm(
                func=reward_alive,
                weight=0.1
            )
        ),

        # Termination manager config
        termination_terms=config_dict.create(
            bad_orientation=TerminationTerm(
                func=check_orientation,
                time_out=False
            ),
            height_limit=TerminationTerm(
                func=check_height,
                time_out=False
            )
        ),

        # Command manager config
        command_func=sample_velocity_command,
        command_resample_interval=500,
        command_ranges=config_dict.create(
            lin_vel_x=(-1.0, 1.0),
            lin_vel_y=(-0.5, 0.5),
            ang_vel_z=(-1.0, 1.0)
        ),

        # Event manager config
        event_terms=config_dict.create(
            domain_randomization=EventTerm(
                func=randomize_physics,
                mode="interval",
                interval=1000
            )
        ),

        # Curriculum manager config
        curriculum_mode="threshold",
        curriculum_threshold=0.8,
        curriculum_terms=config_dict.create(
            command_scale=CurriculumTerm(
                param_name="command_scale",
                start_value=0.2,
                end_value=1.0,
                schedule="linear"
            )
        )
    )
```

---

## Integration with Existing System

### 1. Registry Integration

```python
# Add manager-based environments to registry
def register_manager_env(name: str, config_fn: Callable):
    """Register a manager-based environment."""
    _manager_envs[name] = config_fn

def load_manager_env(name: str, config: ConfigDict) -> ManagerBasedEnv:
    """Load a manager-based environment."""
    config_fn = _manager_envs[name]
    full_config = config_fn()
    full_config.update(config)
    return create_manager_based_env(name, full_config)
```

### 2. Wrapper Compatibility

```python
class ManagerBasedWrapper(VmapWrapper):
    """Wrapper to make ManagerBasedEnv compatible with Brax training."""

    def reset(self, rng: jax.Array) -> State:
        manager_state = self.env.reset(rng)
        return self._manager_to_brax_state(manager_state)

    def step(self, state: State, action: jax.Array) -> State:
        manager_state = self._brax_to_manager_state(state)
        new_manager_state = self.env.step(manager_state, action, state.rng)
        return self._manager_to_brax_state(new_manager_state)
```

### 3. Backward Compatibility

```python
# Existing environments continue to work
env = registry.load("Go1-joystick", config)

# New manager-based environments use new API
env = registry.load_manager("Go1-joystick-manager", config)
```

---

## Implementation Roadmap

### Phase 1: Core Infrastructure
1. ✅ Design document
2. ⬜ Implement `ManagerBase` abstract class
3. ⬜ Implement manager state dataclasses
4. ⬜ Implement term configuration dataclasses

### Phase 2: Individual Managers
5. ⬜ Implement `ObservationManager`
6. ⬜ Implement `RewardManager`
7. ⬜ Implement `TerminationManager`
8. ⬜ Implement `ActionManager`
9. ⬜ Implement `CommandManager`
10. ⬜ Implement `EventManager`
11. ⬜ Implement `CurriculumManager`

### Phase 3: Environment Integration
12. ⬜ Implement `ManagerBasedEnv` class
13. ⬜ Implement `ManagerBasedState` dataclass
14. ⬜ Implement auto-reset logic
15. ⬜ Add MJWarp backend support

### Phase 4: Training Pipeline
16. ⬜ Create `train_manager_ppo.py` script
17. ⬜ Implement manager config loading
18. ⬜ Create Brax wrapper for manager envs
19. ⬜ Add logging and visualization

### Phase 5: Example Tasks
20. ⬜ Convert Go1-joystick to manager-based
21. ⬜ Add manager config for locomotion
22. ⬜ Create example reward/observation terms
23. ⬜ Add curriculum example

### Phase 6: Testing & Documentation
24. ⬜ Write unit tests for each manager
25. ⬜ Integration tests with Brax training
26. ⬜ Performance benchmarks (JIT/VMAP)
27. ⬜ Usage documentation and tutorials

---

## Key Advantages

### 1. **Modularity**
- Separate concerns (obs, reward, termination, etc.)
- Easy to add/remove/modify individual terms
- Reusable components across tasks

### 2. **Configuration-Driven**
- No code changes for task variations
- Easy hyperparameter tuning
- Supports curriculum and adaptation

### 3. **JAX-Native Performance**
- Full JIT compilation
- VMAP for parallelization
- GPU acceleration

### 4. **Compatibility**
- Works with existing registry system
- Compatible with Brax training
- Supports both MJX and MJWarp backends

### 5. **Extensibility**
- Easy to add custom managers
- Plugin-based term system
- Follows JAX ecosystem patterns

---

## Example Usage

```python
# Define a manager-based task
def go1_velocity_manager_config():
    config = get_manager_config("Go1")

    # Customize observation terms
    config.observation_terms.policy.update({
        "velocity_command": ObservationTerm(
            func=lambda state, cfg: cfg.command_state.current_command,
            scale=1.0
        )
    })

    # Customize reward terms
    config.reward_terms.update({
        "velocity_tracking": RewardTerm(
            func=compute_velocity_tracking_reward,
            weight=2.0
        ),
        "smooth_motion": RewardTerm(
            func=compute_smoothness_reward,
            weight=0.5
        )
    })

    return config

# Train the task
train_manager_ppo(
    "Go1-velocity-manager",
    num_envs=4096,
    num_timesteps=50_000_000,
    backend="mjx"
)
```

---

## Conclusion

This manager-based architecture brings the proven patterns from mjlab to mujoco_playground while maintaining full JAX/Flax/Brax/MJX/MJWarp compatibility. The functional, immutable design ensures high performance through JIT compilation and VMAP parallelization, while the modular structure makes it easy to compose complex training tasks.

The architecture is backward compatible with existing environments and provides a clear migration path for converting existing tasks to the manager-based pattern.
