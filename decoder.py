from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np

from allocator.objectives import IncrementalEstimate, estimate_assignment_increment
from config.default import AppConfig
from core.entities import Drone, Order, Station, WeatherSnapshot


@dataclass(slots=True)
class DecodedAllocation:
    assignments: Dict[str, list[str]]
    estimates: Dict[tuple[str, str], IncrementalEstimate]


def decode_particle(
    particle: np.ndarray,
    current_time: float,
    drones: list[Drone],
    candidate_orders: list[Order],
    order_lookup: Dict[str, Order],
    stations: list[Station],
    weather: WeatherSnapshot,
    cfg: AppConfig,
    available_time_by_drone: Dict[str, float] | None = None,
    anchor_by_drone: Dict[str, object] | None = None,
    payload_by_drone: Dict[str, float] | None = None,
    battery_by_drone: Dict[str, float] | None = None,
) -> DecodedAllocation:
    assignments = {drone.drone_id: [] for drone in drones}
    estimates: Dict[tuple[str, str], IncrementalEstimate] = {}
    available_time_by_drone = dict(available_time_by_drone or {
        drone.drone_id: max(current_time, drone.next_available_time) for drone in drones
    })
    anchor_by_drone = dict(anchor_by_drone or {
        drone.drone_id: (drone.planned_destination or drone.position) for drone in drones
    })
    payload_by_drone = dict(payload_by_drone or {
        drone.drone_id: drone.payload_sum(order_lookup) for drone in drones
    })
    battery_by_drone = dict(battery_by_drone or {
        drone.drone_id: float(drone.battery_current) for drone in drones
    })

    order_indices = list(np.argsort(-particle[:, 0]))
    for order_idx in order_indices:
        order = candidate_orders[int(order_idx)]
        pref = int(np.floor(np.clip(particle[order_idx, 1], 0.0, 0.999999) * len(drones)))
        preferred_drone = drones[pref]
        candidates = [preferred_drone] + [drone for drone in drones if drone.drone_id != preferred_drone.drone_id]

        best_choice = None
        best_estimate = None
        best_score = float("inf")
        for drone in candidates:
            est = estimate_assignment_increment(
                current_time=current_time,
                drone=drone,
                order=order,
                order_lookup=order_lookup,
                stations=stations,
                weather=weather,
                cfg=cfg,
                available_time=available_time_by_drone[drone.drone_id],
                anchor_position=anchor_by_drone[drone.drone_id],
                payload_before_access=payload_by_drone[drone.drone_id],
                battery_available=battery_by_drone[drone.drone_id],
            )
            if not est.feasible:
                continue
            score = est.deliver_duration_min + 2.0 * est.cancel_proxy + 0.5 * est.lateness_min
            if score < best_score:
                best_score = score
                best_choice = drone
                best_estimate = est

        if best_choice is None or best_estimate is None:
            continue
        assignments[best_choice.drone_id].append(order.order_id)
        available_time_by_drone[best_choice.drone_id] = best_estimate.predicted_deliver_time
        anchor_by_drone[best_choice.drone_id] = order.customer_loc
        battery_by_drone[best_choice.drone_id] = max(0.0, battery_by_drone[best_choice.drone_id] - best_estimate.energy_used)
        estimates[(best_choice.drone_id, order.order_id)] = best_estimate

    return DecodedAllocation(assignments=assignments, estimates=estimates)
