from __future__ import annotations

from typing import Dict

import numpy as np

from config.default import AppConfig
from core.entities import Drone, Order, WeatherSnapshot
from core.enums import OrderStatus, TaskType
from core.utils import clip, minmax_normalize
from env.constraints import TaskProjection
from rl.rule_actions import NON_WAIT_RULES


def build_observation(
    current_time: float,
    drone: Drone,
    orders: Dict[str, Order],
    weather: WeatherSnapshot,
    rule_prototypes: Dict[str, TaskProjection | None],
    cancel_risk_lookup: Dict[str, float],
    battery_lower_bound: float,
    x_bounds: tuple[float, float],
    y_bounds: tuple[float, float],
    cfg: AppConfig,
) -> np.ndarray:
    payload = drone.payload_sum(orders)
    assigned_ready_count = sum(
        1
        for oid in drone.assigned_order_ids
        if oid in orders
        and orders[oid].status == OrderStatus.ASSIGNED_READY
        and orders[oid].ready_time <= current_time
    )
    battery_margin_norm = clip(
        (drone.battery_current - battery_lower_bound)
        / max(drone.battery_max - battery_lower_bound, 1e-6),
        0.0,
        1.0,
    )
    obs = [
        minmax_normalize(drone.position.x, x_bounds[0], x_bounds[1], 0.5),
        minmax_normalize(drone.position.y, y_bounds[0], y_bounds[1], 0.5),
        battery_margin_norm,
        clip(1.0 - payload / max(drone.max_capacity_kg, 1e-6), 0.0, 1.0),
        clip(drone.onboard_count / max(cfg.rl.onboard_norm_ref, 1e-6), 0.0, 1.0),
        clip(assigned_ready_count / max(cfg.rl.assigned_norm_ref, 1e-6), 0.0, 1.0),
        float(weather.speed_factor),
        float(weather.energy_factor),
    ]
    for rule_name in NON_WAIT_RULES:
        proto = rule_prototypes.get(rule_name)
        if proto is None:
            obs.extend([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            continue
        order = orders[proto.order_id]
        complete_delta = max(0.0, proto.order_finish_time - current_time)
        obs.extend(
            [
                1.0,
                1.0 if proto.task_type == TaskType.DROPOFF else 0.0,
                clip(complete_delta / cfg.rl.time_ref_min, 0.0, cfg.rl.complete_norm_cap),
                clip(proto.lateness_min / cfg.rl.time_ref_min, 0.0, cfg.rl.lateness_norm_cap),
                0.0 if proto.task_type == TaskType.DROPOFF else cancel_risk_lookup.get(proto.order_id, 0.0),
                clip(order.quantity_kg / max(cfg.rl.quantity_ref_kg, 1e-6), 0.0, 2.0),
            ]
        )
    return np.asarray(obs, dtype=np.float32)
