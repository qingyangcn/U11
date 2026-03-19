"""
Single UAV Training Wrapper for Event-Driven Shared Policy

This wrapper enables training a shared policy that can be used by all drones
in a decentralized manner. It provides:

1. Discrete(5) action space (independent of number of drones N)
2. Rule-discriminant compact observation for the current decision drone (20-dim Box)
3. Drone sampling strategy for training data generation
4. Compatibility with SB3 PPO and other single-agent RL algorithms

Key Features:
- Action space: Discrete(5) - one rule_id per step
- Observation: Rule-based compact state vector (20 features, Box)
  Replaces the former high-dimensional Dict(drone_state, candidates, global_context).
  Features cover drone own-state, candidate task structure, rule-discriminant
  metrics and global context – see env._get_rule_based_state_for_drone() for details.
- Drone selection: Randomly samples from drones at decision points
- Episode handling: Advances until episode end or max steps reached

Two operating modes (controlled by ``round_mode``):

  ``round_synchronous`` (default, recommended):
      Aligns with DecentralizedEventDrivenExecutor semantics.  In every decision
      round ALL decision-drones have their actions collected *before* env.step() is
      called once.  SB3 compatibility is maintained via a per-round index: the
      first N-1 agent-steps within a round return reward=0; the Nth (final) step
      applies all pending actions, advances the environment once, and returns the
      true reward.  info['wrapper'] always contains round metadata.

  ``legacy_stepwise``:
      Original behaviour: one drone decides then env.step() is called immediately.
      Retained for regression comparison only.

Usage:
    env = ThreeObjectiveDroneDeliveryEnv(...)
    env = SingleUAVTrainingWrapper(env)                          # round_synchronous
    env = SingleUAVTrainingWrapper(env, round_mode='legacy_stepwise')  # old behaviour

    # Compatible with SB3 PPO:
    model = PPO("MlpPolicy", env, ...)
"""

from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import gymnasium as gym
from gymnasium import spaces


class SingleUAVTrainingWrapper(gym.Wrapper):
    """
    Training wrapper that converts multi-drone environment to single-drone interface.

    This wrapper:
    1. Converts action space from MultiDiscrete([R]*N) to Discrete(R)
    2. Provides rule-discriminant compact observation (20-dim Box) for the current drone
    3. Samples drones at decision points to generate training data
    4. Advances environment to next decision event when needed

    Args:
        env: The base UAV environment
        max_skip_steps: Maximum steps to skip when waiting for decisions
        drone_sampling: Strategy for selecting drones ('random', 'round_robin').
            Only relevant in ``legacy_stepwise`` mode; in ``round_synchronous`` mode
            the full decision-round drone list is always used in deterministic order.
        local_obs_only: Retained for API compatibility; observation is always the
                        compact rule-based state vector regardless of this flag.
        round_mode: Training semantics.  ``'round_synchronous'`` (default) collects
                    all drones' actions within a decision round before calling
                    env.step() once, matching the decentralized executor.
                    ``'legacy_stepwise'`` restores the original one-drone-then-step
                    behaviour for regression purposes.
    """

    def __init__(
            self,
            env: gym.Env,
            max_skip_steps: int = 10,
            drone_sampling: str = 'random',
            local_obs_only: bool = False,
            round_mode: str = 'round_synchronous',
    ):
        super().__init__(env)

        if round_mode not in ('round_synchronous', 'legacy_stepwise'):
            raise ValueError(
                f"round_mode must be 'round_synchronous' or 'legacy_stepwise', "
                f"got {round_mode!r}"
            )

        self.max_skip_steps = max_skip_steps
        self.drone_sampling = drone_sampling
        self.local_obs_only = local_obs_only
        self.round_mode = round_mode

        # Get number of rules from environment
        if hasattr(env.unwrapped, 'rule_count'):
            self.rule_count = env.unwrapped.rule_count
        else:
            self.rule_count = 5  # Default

        # Override action space to single discrete action
        self.action_space = spaces.Discrete(self.rule_count)

        # Compact rule-based state: flat Box of RULE_BASED_STATE_DIM features.
        # All values are normalised to [0, 1].
        state_dim = getattr(env.unwrapped, 'RULE_BASED_STATE_DIM', 20)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(state_dim,), dtype=np.float32
        )

        # ── Legacy (stepwise) state ────────────────────────────────────────────
        self.current_drone_id: Optional[int] = None
        self.decision_queue: List[int] = []
        self.round_robin_index: int = 0

        # ── Round-synchronous state ────────────────────────────────────────────
        # _round_drones: ordered list of drone IDs for the active decision round
        # _round_index:  index of the drone waiting for its action right now
        # _pending_actions: drone_id -> action collected so far this round
        self._round_drones: List[int] = []
        self._round_index: int = 0
        self._pending_actions: Dict[int, int] = {}
        self._last_round_reward: float = 0.0

        # Track last observation from environment
        self.last_full_obs: Optional[Dict] = None
        self.last_info: Optional[Dict] = None

        # ── Statistics ─────────────────────────────────────────────────────────
        # decision_rounds:     number of rounds where env.step() was actually called
        # individual_decisions: per-drone policy calls (one per step() invocation)
        # actions_applied:     total drone-actions submitted to apply_rule_to_drone
        # round_flush_count:   alias for decision_rounds (rounds that were flushed)
        # total_skip_steps:    env steps taken while waiting for decision events
        self.decision_rounds: int = 0
        self.individual_decisions: int = 0
        self.actions_applied: int = 0
        self.round_flush_count: int = 0
        self.total_skip_steps: int = 0
        # Keep total_decisions as a backward-compatible alias
        self.total_decisions: int = 0
        self.episode_steps: int = 0

    # ──────────────────────────────────────────────────────────────────────────
    # Public interface
    # ──────────────────────────────────────────────────────────────────────────

    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict]:
        """
        Reset environment and get first compact local observation.

        Returns:
            observation: Compact rule-based state vector for first decision drone
            info: Info dictionary
        """
        # Reset underlying environment
        obs, info = self.env.reset(**kwargs)

        # Store full observation
        self.last_full_obs = obs
        self.last_info = info

        # Reset all state
        self.current_drone_id = None
        self.decision_queue = []
        self.round_robin_index = 0
        self._round_drones = []
        self._round_index = 0
        self._pending_actions = {}
        self._last_round_reward = 0.0
        self.decision_rounds = 0
        self.individual_decisions = 0
        self.actions_applied = 0
        self.round_flush_count = 0
        self.total_skip_steps = 0
        self.total_decisions = 0
        self.episode_steps = 0

        if self.round_mode == 'round_synchronous':
            local_obs, info = self._reset_round_synchronous(info)
        else:
            local_obs, info = self._reset_legacy(info)

        return local_obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute action for the current drone.

        In ``round_synchronous`` mode the action is buffered until all drones in
        the current decision round have provided their actions; only then is
        ``env.step()`` called once.  Intermediate (intra-round) calls return
        ``reward=0``.

        In ``legacy_stepwise`` mode ``env.step()`` is called immediately after
        each drone action (original behaviour).

        Args:
            action: Rule ID (0..rule_count-1) to apply to current drone

        Returns:
            observation: Next compact local observation
            reward: Accumulated reward (0 for intra-round steps in round_synchronous)
            terminated: Whether episode terminated
            truncated: Whether episode truncated
            info: Info dictionary with round metadata in info['wrapper']
        """
        if self.round_mode == 'round_synchronous':
            return self._step_round_synchronous(action)
        else:
            return self._step_legacy(action)

    # ──────────────────────────────────────────────────────────────────────────
    # Round-synchronous implementation
    # ──────────────────────────────────────────────────────────────────────────

    def _reset_round_synchronous(self, info: Dict) -> Tuple[np.ndarray, Dict]:
        """Initialise round state after env.reset() and return first obs."""
        # Find the first decision round, skipping if necessary
        round_drones = self.env.unwrapped.get_decision_drones()
        if not round_drones:
            obs, skip_reward, terminated, truncated, info = self._skip_to_next_decision()
            self.last_full_obs = obs
            self.last_info = info
            if not (terminated or truncated):
                round_drones = self.env.unwrapped.get_decision_drones()

        self._round_drones = list(round_drones)
        self._round_index = 0
        self._pending_actions = {}

        # current drone is the first in the round
        self.current_drone_id = (
            self._round_drones[0] if self._round_drones else None
        )

        local_obs = self._extract_local_observation(
            self.last_full_obs, self.current_drone_id
        )
        info['wrapper'] = self._build_wrapper_info(
            round_completed=False,
            last_round_reward=0.0,
        )
        return local_obs, info

    def _step_round_synchronous(
            self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Round-synchronous step: buffer action; flush round when all drones decided.
        """
        if not self._round_drones or self.current_drone_id is None:
            raise RuntimeError(
                "No active decision round. This should not happen – likely a reset() issue."
            )

        drone_id = self.current_drone_id
        self._pending_actions[drone_id] = action
        self.individual_decisions += 1
        self.total_decisions += 1

        round_completed = (self._round_index >= len(self._round_drones) - 1)

        if not round_completed:
            # ── intra-round: advance index, return zero reward ─────────────────
            self._round_index += 1
            self.current_drone_id = self._round_drones[self._round_index]
            local_obs = self._extract_local_observation(
                self.last_full_obs, self.current_drone_id
            )
            info = dict(self.last_info) if self.last_info else {}
            info['wrapper'] = self._build_wrapper_info(
                round_completed=False,
                last_round_reward=self._last_round_reward,
            )
            return local_obs, 0.0, False, False, info

        # ── round flush: apply all pending actions, then step env once ─────────
        for did, act in self._pending_actions.items():
            if hasattr(self.env.unwrapped, 'apply_rule_to_drone_with_info'):
                self.env.unwrapped.apply_rule_to_drone_with_info(did, act)
            else:
                self.env.unwrapped.apply_rule_to_drone(did, act)
            self.actions_applied += 1

        dummy_action = np.zeros(self.env.unwrapped.num_drones, dtype=np.int32)
        obs, reward, terminated, truncated, env_info = self.env.step(dummy_action)

        self.last_full_obs = obs
        self.last_info = env_info
        self.decision_rounds += 1
        self.round_flush_count += 1
        self.episode_steps += 1
        self._last_round_reward = float(reward)

        # Reset round state
        self._pending_actions = {}

        # ── find next decision round (skipping if needed) ──────────────────────
        if not (terminated or truncated):
            next_round_drones = self.env.unwrapped.get_decision_drones()
            if not next_round_drones:
                skip_obs, skip_reward, terminated, truncated, skip_info = \
                    self._skip_to_next_decision()
                reward += skip_reward
                self._last_round_reward += skip_reward
                obs = skip_obs
                env_info = skip_info
                self.last_full_obs = obs
                self.last_info = env_info
                if not (terminated or truncated):
                    next_round_drones = self.env.unwrapped.get_decision_drones()
            self._round_drones = list(next_round_drones) if not (terminated or truncated) else []
        else:
            self._round_drones = []

        self._round_index = 0
        self.current_drone_id = (
            self._round_drones[0] if self._round_drones else None
        )

        local_obs = self._extract_local_observation(obs, self.current_drone_id)

        env_info['wrapper'] = self._build_wrapper_info(
            round_completed=True,
            last_round_reward=self._last_round_reward,
        )
        return local_obs, reward, terminated, truncated, env_info

    def _build_wrapper_info(
            self,
            round_completed: bool,
            last_round_reward: float,
    ) -> Dict[str, Any]:
        """Build the standardised info['wrapper'] dict."""
        return {
            # Round metadata
            'round_completed': round_completed,
            'round_size': len(self._round_drones),
            'round_index': self._round_index,
            'pending_actions': len(self._pending_actions),
            'last_round_reward': last_round_reward,
            # Current drone
            'current_drone_id': self.current_drone_id,
            # Cumulative statistics
            'decision_rounds': self.decision_rounds,
            'individual_decisions': self.individual_decisions,
            'actions_applied': self.actions_applied,
            'round_flush_count': self.round_flush_count,
            'total_skip_steps': self.total_skip_steps,
            'episode_steps': self.episode_steps,
        }

    # ──────────────────────────────────────────────────────────────────────────
    # Legacy (stepwise) implementation  – unchanged original logic
    # ──────────────────────────────────────────────────────────────────────────

    def _reset_legacy(self, info: Dict) -> Tuple[np.ndarray, Dict]:
        """Legacy reset: find first decision drone via queue."""
        self._populate_decision_queue()

        if self.current_drone_id is None:
            obs, _, _, _, info = self._skip_to_next_decision()
            self.last_full_obs = obs
            self.last_info = info
            self._populate_decision_queue()

        local_obs = self._extract_local_observation(
            self.last_full_obs, self.current_drone_id
        )
        info['wrapper'] = {
            'current_drone_id': self.current_drone_id,
            'queue_length': len(self.decision_queue),
            'total_decisions': self.total_decisions,
            'total_skip_steps': self.total_skip_steps,
            'decision_rounds': self.decision_rounds,
            'individual_decisions': self.individual_decisions,
            'actions_applied': self.actions_applied,
            'round_flush_count': self.round_flush_count,
        }
        return local_obs, info

    def _step_legacy(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Legacy step: apply rule for one drone then immediately advance env."""
        if self.current_drone_id is None:
            raise RuntimeError(
                "No current drone. This should not happen - likely a reset() issue."
            )

        drone_id = self.current_drone_id

        # Apply rule to drone through arbitrator
        success = self.env.unwrapped.apply_rule_to_drone(drone_id, action)
        self.actions_applied += 1
        self.individual_decisions += 1

        # Advance environment one step
        dummy_action = np.zeros(self.env.unwrapped.num_drones, dtype=np.int32)
        obs, reward, terminated, truncated, info = self.env.step(dummy_action)
        self.decision_rounds += 1
        self.round_flush_count += 1

        # Store observation
        self.last_full_obs = obs
        self.last_info = info

        # Update statistics
        self.total_decisions += 1
        self.episode_steps += 1

        # Move to next drone
        self.current_drone_id = None
        self._populate_decision_queue()

        # If no more drones and episode not done, skip forward
        if self.current_drone_id is None and not (terminated or truncated):
            skip_obs, skip_reward, terminated, truncated, skip_info = \
                self._skip_to_next_decision()
            obs = skip_obs
            reward += skip_reward
            info.update(skip_info)
            self.last_full_obs = obs
            self.last_info = info
            self._populate_decision_queue()

        # Get next compact local observation
        local_obs = self._extract_local_observation(
            self.last_full_obs, self.current_drone_id
        )

        # Add metadata to info
        info['wrapper'] = {
            'last_drone_id': drone_id,
            'last_rule_id': action,
            'last_decision_success': success,
            'current_drone_id': self.current_drone_id,
            'queue_length': len(self.decision_queue),
            'total_decisions': self.total_decisions,
            'total_skip_steps': self.total_skip_steps,
            'episode_steps': self.episode_steps,
            'decision_rounds': self.decision_rounds,
            'individual_decisions': self.individual_decisions,
            'actions_applied': self.actions_applied,
            'round_flush_count': self.round_flush_count,
        }

        return local_obs, reward, terminated, truncated, info

    # ──────────────────────────────────────────────────────────────────────────
    # Shared helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _populate_decision_queue(self):
        """
        Populate decision queue with drones at decision points (legacy mode only).

        Uses the configured drone_sampling strategy to select next drone.
        """
        if not self.decision_queue:
            # Get drones at decision points from environment
            decision_drones = self.env.unwrapped.get_decision_drones()

            if not decision_drones:
                self.current_drone_id = None
                return

            # Select drone based on sampling strategy
            if self.drone_sampling == 'random':
                # Randomly sample one drone
                selected_drone = np.random.choice(decision_drones)
                self.current_drone_id = selected_drone
                # Add remaining drones to queue for potential future use
                self.decision_queue = [d for d in decision_drones if d != selected_drone]

            elif self.drone_sampling == 'round_robin':
                # Sort for consistent ordering
                decision_drones = sorted(decision_drones)
                # Use round-robin index to select
                selected_idx = self.round_robin_index % len(decision_drones)
                selected_drone = decision_drones[selected_idx]
                self.current_drone_id = selected_drone
                self.round_robin_index += 1
                # Add remaining drones to queue
                self.decision_queue = [
                    d for d in decision_drones if d != selected_drone
                ]

            else:
                raise ValueError(f"Unknown drone_sampling strategy: {self.drone_sampling}")

        else:
            # Pop next drone from queue
            self.current_drone_id = self.decision_queue.pop(0)

    def _skip_to_next_decision(self) -> Tuple[Any, float, bool, bool, Dict]:
        """
        Skip forward until a decision point appears or episode ends.

        Returns:
            observation, accumulated_reward, terminated, truncated, info
        """
        total_reward = 0.0
        terminated = False
        truncated = False
        obs = self.last_full_obs
        info = {}

        for skip_step in range(self.max_skip_steps):
            # Advance environment with no-op
            dummy_action = np.zeros(self.env.unwrapped.num_drones, dtype=np.int32)
            obs, reward, terminated, truncated, info = self.env.step(dummy_action)

            total_reward += reward
            self.total_skip_steps += 1
            self.episode_steps += 1

            # Check if episode ended
            if terminated or truncated:
                break

            # Check for decision points
            decision_drones = self.env.unwrapped.get_decision_drones()
            if decision_drones:
                # Found decision points
                break

        info['skip_info'] = {
            'steps_skipped': skip_step + 1,
            'reason': 'episode_end' if (terminated or truncated) else 'decision_found'
        }

        return obs, total_reward, terminated, truncated, info

    def _extract_local_observation(
            self,
            full_obs: Optional[Any],
            drone_id: Optional[int]
    ) -> np.ndarray:
        """
        Return the compact rule-based state vector for a specific drone.

        Delegates to env.unwrapped._get_rule_based_state_for_drone() which builds
        the 20-dimensional rule-discriminant feature vector directly from environment
        state (independent of the full observation dict).

        Args:
            full_obs: Full observation from environment (unused; kept for API compat)
            drone_id: Drone ID to extract observation for (None = return zeros)

        Returns:
            np.ndarray of shape (RULE_BASED_STATE_DIM,), dtype=np.float32
        """
        state_dim = self.observation_space.shape[0]

        if drone_id is None:
            return np.zeros(state_dim, dtype=np.float32)
        return self.env.unwrapped._get_rule_based_state_for_drone(drone_id)