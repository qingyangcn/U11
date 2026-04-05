from __future__ import annotations

from typing import Dict

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from config.default import AppConfig, build_default_config
from env.simulator import EventDrivenSimulator
from rl.action_mask import build_action_alias_details, build_action_mask
from rl.reward import compute_step_reward
from rl.rule_actions import INDEX_TO_ACTION, RULE_ACTIONS
from rl.rule_decoder import build_rule_prototypes, select_rule_projection
from rl.state_encoder import StateEncoder


class DeliveryEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, cfg: AppConfig | None = None, seed: int | None = None) -> None:
        super().__init__()
        self.cfg = cfg or build_default_config()
        self.simulator = EventDrivenSimulator(self.cfg, seed=seed)
        self.state_encoder = StateEncoder(self.cfg, self.simulator.x_bounds, self.simulator.y_bounds)
        self.action_space = spaces.Discrete(len(RULE_ACTIONS))
        self.observation_space = spaces.Box(
            low=np.zeros(self.cfg.rl.observation_dim, dtype=np.float32),
            high=np.full(self.cfg.rl.observation_dim, 4.0, dtype=np.float32),
            dtype=np.float32,
        )
        self._last_obs = np.zeros(self.cfg.rl.observation_dim, dtype=np.float32)

    def reset(self, *, seed: int | None = None, options: Dict | None = None):
        del options
        super().reset(seed=seed)
        self.simulator.reset(seed=seed)
        self.state_encoder = StateEncoder(self.cfg, self.simulator.x_bounds, self.simulator.y_bounds)
        self._last_obs = self._build_observation()
        return self._last_obs, self._build_info()

    def step(self, action: int):
        action_name = INDEX_TO_ACTION[int(action)]
        metrics = self.simulator.step(action_name, select_rule_projection)
        reward = compute_step_reward(metrics, self.cfg)
        terminated = bool(self.simulator.done)
        truncated = False
        if terminated:
            obs = np.zeros(self.cfg.rl.observation_dim, dtype=np.float32)
        else:
            obs = self._build_observation()
        self._last_obs = obs
        info = self._build_info()
        info["step_metrics"] = metrics.as_dict()
        info["action_name"] = action_name
        return obs, reward, terminated, truncated, info

    def action_masks(self) -> np.ndarray:
        if self.simulator.done or self.simulator.current_decision is None:
            return np.zeros(len(RULE_ACTIONS), dtype=bool)
        rule_prototypes, force_charge = self._current_rule_context()
        return build_action_mask(rule_prototypes=rule_prototypes, force_charge=force_charge)

    def action_alias_info(self) -> Dict[str, Dict[str, object]]:
        if self.simulator.done or self.simulator.current_decision is None:
            return {
                rule_name: {
                    "available": False,
                    "kept": False,
                    "alias_of": None,
                    "order_id": None,
                    "task_type": None,
                }
                for rule_name in RULE_ACTIONS
            }
        rule_prototypes, force_charge = self._current_rule_context()
        return build_action_alias_details(rule_prototypes=rule_prototypes, force_charge=force_charge)

    def _build_observation(self) -> np.ndarray:
        if self.simulator.done or self.simulator.current_decision is None:
            return np.zeros(self.cfg.rl.observation_dim, dtype=np.float32)
        drone_id = self.simulator.current_decision.drone_id
        drone = self.simulator.drones[drone_id]
        projections = self.simulator.get_feasible_projections(drone_id)
        rule_prototypes = build_rule_prototypes(projections, self.simulator.orders)
        cancel_lookup = self.simulator.get_cancel_risk_lookup(projections.keys())
        battery_lower = self.simulator.get_battery_lower_bound(drone_id)
        return self.state_encoder.encode(
            current_time=self.simulator.current_time,
            drone=drone,
            orders=self.simulator.orders,
            weather=self.simulator.current_weather,
            rule_prototypes=rule_prototypes,
            cancel_risk_lookup=cancel_lookup,
            battery_lower_bound=battery_lower,
        )

    def _build_info(self) -> Dict:
        info = {
            "time_min": self.simulator.current_time,
            "done": self.simulator.done,
            "episode_stats": dict(self.simulator.episode_stats),
        }
        if self.simulator.current_decision is not None:
            info["decision_drone_id"] = self.simulator.current_decision.drone_id
        return info

    def _current_rule_context(self) -> tuple[Dict[str, object], bool]:
        drone_id = self.simulator.current_decision.drone_id
        projections = self.simulator.get_feasible_projections(drone_id)
        rule_prototypes = build_rule_prototypes(projections, self.simulator.orders)
        force_charge = self.simulator.get_force_charge_flag(drone_id)
        return rule_prototypes, force_charge
