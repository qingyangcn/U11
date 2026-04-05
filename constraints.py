from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List

from config.default import AppConfig
from core.entities import Drone, Order, Point, Station, Task, WeatherSnapshot
from core.enums import OrderStatus, TaskType
from env.battery import energy_to_station, lower_bound_energy
from env.travel import flight_energy, flight_time_min


@dataclass(slots=True)
class TaskProjection:
    task_type: TaskType
    order_id: str
    feasible: bool
    travel_time_min: float
    service_time_min: float
    task_finish_time: float
    order_finish_time: float
    lateness_min: float
    task_energy: float
    return_energy: float
    end_position: Point
    payload_after_task: float


def _pickup_service_time(drone: Drone) -> float:
    return drone.pickup_service_mean_min


def _dropoff_service_time(drone: Drone) -> float:
    return drone.dropoff_service_mean_min


def project_task(
    now: float,
    drone: Drone,
    order: Order,
    task_type: TaskType,
    orders: Dict[str, Order],
    stations: Iterable[Station],
    weather: WeatherSnapshot,
    cfg: AppConfig,
) -> TaskProjection:
    current_payload = drone.payload_sum(orders)
    if task_type == TaskType.PICKUP:
        travel_time = flight_time_min(
            drone.position,
            order.merchant_loc,
            drone,
            weather,
            current_payload,
        )
        service_time = _pickup_service_time(drone)
        task_energy = flight_energy(drone.position, order.merchant_loc, drone, weather, current_payload)
        payload_after_task = current_payload + order.quantity_kg
        end_position = order.merchant_loc
        post_task_return = energy_to_station(end_position, drone, stations, weather, payload_after_task)
        # Rule features for pickup use predicted full-order completion time.
        dropoff_time = flight_time_min(
            order.merchant_loc,
            order.customer_loc,
            drone,
            weather,
            payload_after_task,
        )
        order_finish_time = now + travel_time + service_time + dropoff_time + _dropoff_service_time(drone)
    else:
        travel_time = flight_time_min(
            drone.position,
            order.customer_loc,
            drone,
            weather,
            current_payload,
        )
        service_time = _dropoff_service_time(drone)
        task_energy = flight_energy(drone.position, order.customer_loc, drone, weather, current_payload)
        payload_after_task = max(0.0, current_payload - order.quantity_kg)
        end_position = order.customer_loc
        post_task_return = energy_to_station(end_position, drone, stations, weather, payload_after_task)
        order_finish_time = now + travel_time + service_time

    task_finish_time = now + travel_time + service_time
    lateness = max(0.0, order_finish_time - order.deadline)
    feasible = (
        drone.battery_current >= task_energy + post_task_return + drone.battery_safety_margin
        and payload_after_task <= drone.max_capacity_kg + 1e-6
    )
    return TaskProjection(
        task_type=task_type,
        order_id=order.order_id,
        feasible=feasible,
        travel_time_min=travel_time,
        service_time_min=service_time,
        task_finish_time=task_finish_time,
        order_finish_time=order_finish_time,
        lateness_min=lateness,
        task_energy=task_energy,
        return_energy=post_task_return,
        end_position=end_position,
        payload_after_task=payload_after_task,
    )


def collect_feasible_task_projections(
    now: float,
    drone: Drone,
    orders: Dict[str, Order],
    stations: Iterable[Station],
    weather: WeatherSnapshot,
    cfg: AppConfig,
) -> List[TaskProjection]:
    projections: List[TaskProjection] = []
    for order_id in drone.picked_order_ids:
        order = orders[order_id]
        if order.status != OrderStatus.PICKED:
            continue
        proj = project_task(now, drone, order, TaskType.DROPOFF, orders, stations, weather, cfg)
        if proj.feasible:
            projections.append(proj)
    for order_id in drone.assigned_order_ids:
        if order_id in drone.picked_order_ids:
            continue
        order = orders[order_id]
        if order.status != OrderStatus.ASSIGNED_READY or order.ready_time > now:
            continue
        proj = project_task(now, drone, order, TaskType.PICKUP, orders, stations, weather, cfg)
        if proj.feasible:
            projections.append(proj)
    return projections


def must_force_charge(
    now: float,
    drone: Drone,
    orders: Dict[str, Order],
    stations: Iterable[Station],
    weather: WeatherSnapshot,
    cfg: AppConfig,
) -> bool:
    payload = drone.payload_sum(orders)
    lower = lower_bound_energy(drone, stations, weather, payload, cfg)
    if drone.battery_current <= lower:
        return True
    projections = collect_feasible_task_projections(now, drone, orders, stations, weather, cfg)
    if projections:
        return False
    has_pending_ready = any(
        orders[oid].status == OrderStatus.ASSIGNED_READY and orders[oid].ready_time <= now
        for oid in drone.assigned_order_ids
        if oid in orders
    )
    return has_pending_ready
