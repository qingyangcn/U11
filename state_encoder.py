from __future__ import annotations

from typing import Dict

import numpy as np

from config.default import AppConfig
from core.entities import Drone, Order, WeatherSnapshot
from env.constraints import TaskProjection
from rl.observation_builder import build_observation


class StateEncoder:
    def __init__(self, cfg: AppConfig, x_bounds: tuple[float, float], y_bounds: tuple[float, float]) -> None:
        self.cfg = cfg
        self.x_bounds = x_bounds
        self.y_bounds = y_bounds

    def encode(
        self,
        current_time: float,
        drone: Drone,
        orders: Dict[str, Order],
        weather: WeatherSnapshot,
        rule_prototypes: Dict[str, TaskProjection | None],
        cancel_risk_lookup: Dict[str, float],
        battery_lower_bound: float,
    ) -> np.ndarray:
        return build_observation(
            current_time=current_time,
            drone=drone,
            orders=orders,
            weather=weather,
            rule_prototypes=rule_prototypes,
            cancel_risk_lookup=cancel_risk_lookup,
            battery_lower_bound=battery_lower_bound,
            x_bounds=self.x_bounds,
            y_bounds=self.y_bounds,
            cfg=self.cfg,
        )
