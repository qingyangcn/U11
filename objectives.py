from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np

from config.default import AppConfig
from core.entities import Drone, Order, Station, WeatherSnapshot
from core.enums import DroneMode
from env.battery import energy_to_station
from env.travel import flight_energy, flight_time_min, hover_energy


@dataclass(slots=True)
class IncrementalEstimate:
    deliver_duration_min: float
    predicted_deliver_time: float
    lateness_min: float
    cancel_proxy: float
    workload_after: float
    energy_used: float
    feasible: bool


def current_workload_minutes(current_time: float, drone: Drone) -> float:
    return max(0.0, drone.next_available_time - current_time)


def projected_position(drone: Drone):
    return drone.planned_destination or drone.position


def estimate_assignment_increment(
    current_time: float,
    drone: Drone,
    order: Order,
    order_lookup: Dict[str, Order],
    stations: Iterable[Station],
    weather: WeatherSnapshot,
    cfg: AppConfig,
    available_time: float | None = None,
    anchor_position=None,
    payload_before_access: float | None = None,
    battery_available: float | None = None,
) -> IncrementalEstimate:
    if order.quantity_kg > drone.max_capacity_kg + 1e-6:
        return IncrementalEstimate(0.0, current_time, 0.0, 1.0, 0.0, 0.0, False)
    start_time = max(current_time, drone.next_available_time if available_time is None else available_time)
    anchor = anchor_position or projected_position(drone)
    payload = drone.payload_sum(order_lookup) if payload_before_access is None else float(payload_before_access)
    access = flight_time_min(
        anchor,
        order.merchant_loc,
        drone,
        weather,
        payload,
    )
    pickup_hover = hover_energy(
        drone=drone,
        weather=weather,
        payload_kg=payload,
        duration_min=drone.pickup_service_mean_min,
        mode=DroneMode.PICKUP_SERVICE,
    )
    linehaul = flight_time_min(
        order.merchant_loc,
        order.customer_loc,
        drone,
        weather,
        payload + order.quantity_kg,
    )
    dropoff_hover = hover_energy(
        drone=drone,
        weather=weather,
        payload_kg=payload + order.quantity_kg,
        duration_min=drone.dropoff_service_mean_min,
        mode=DroneMode.DROPOFF_SERVICE,
    )
    duration = access + drone.pickup_service_mean_min + linehaul + drone.dropoff_service_mean_min
    predicted_deliver = start_time + duration
    lateness = max(0.0, predicted_deliver - order.deadline)
    slack_scale = max(order.deadline - order.created_time, 1.0)
    cancel_proxy = min(1.0, order.base_cancel_prob + lateness / slack_scale)
    access_energy = flight_energy(anchor, order.merchant_loc, drone, weather, payload)
    linehaul_energy = flight_energy(order.merchant_loc, order.customer_loc, drone, weather, payload + order.quantity_kg)
    return_energy = energy_to_station(order.customer_loc, drone, stations, weather, payload)
    battery = drone.battery_current if battery_available is None else float(battery_available)
    feasible = battery >= access_energy + pickup_hover + linehaul_energy + dropoff_hover + return_energy + drone.battery_safety_margin
    return IncrementalEstimate(
        deliver_duration_min=duration,
        predicted_deliver_time=predicted_deliver,
        lateness_min=lateness,
        cancel_proxy=cancel_proxy,
        workload_after=max(0.0, predicted_deliver - current_time),
        energy_used=access_energy + pickup_hover + linehaul_energy + dropoff_hover,
        feasible=feasible,
    )


def evaluate_solution(
    current_time: float,
    assignments: Dict[str, list[str]],
    estimates: Dict[tuple[str, str], IncrementalEstimate],
    drones: Iterable[Drone],
) -> tuple[float, float, float]:
    f1 = 0.0
    f2 = 0.0
    workloads: list[float] = []
    for drone in drones:
        wl = current_workload_minutes(current_time, drone)
        for order_id in assignments.get(drone.drone_id, []):
            est = estimates[(drone.drone_id, order_id)]
            f1 += est.deliver_duration_min
            f2 += est.lateness_min + 2.0 * est.cancel_proxy
            wl = est.workload_after
        workloads.append(wl)
    f3 = float(np.var(workloads)) if workloads else 0.0
    return f1, f2, f3
