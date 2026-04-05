from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List

import numpy as np

from config.default import AppConfig
from core.entities import Drone, Order, Point, Station, WeatherSnapshot
from core.enums import DroneMode, OrderStatus, TaskType
from env.battery import energy_to_station, nearest_station
from env.constraints import TaskProjection
from env.travel import flight_energy, flight_time_min, hover_energy


@dataclass(slots=True)
class FutureDroneState:
    drone_id: str
    available_time: float
    position: Point
    battery_current: float
    carried_order_ids: list[str]
    assigned_ready_order_ids: list[str]


@dataclass(slots=True)
class SwarmTask:
    order_id: str
    task_type: TaskType
    priority: float


@dataclass(slots=True)
class PlanEvaluation:
    objective_cost: float
    expected_deliveries: float
    expected_cancellations: float
    total_lateness: float
    total_energy: float
    makespan_delta: float
    unproductive_wait: float


@dataclass(slots=True)
class SwarmPlan:
    assignments: Dict[str, list[str]]
    task_sequences: Dict[str, list[SwarmTask]]
    evaluation: PlanEvaluation


@dataclass(slots=True)
class SwarmProblem:
    current_time: float
    horizon_time: float
    drones: list[Drone]
    drone_by_id: Dict[str, Drone]
    orders: Dict[str, Order]
    stations: list[Station]
    weather: WeatherSnapshot
    future_states: Dict[str, FutureDroneState]
    free_order_ids: list[str]
    optim_order_ids: list[str]


@dataclass(slots=True)
class _DronePlanState:
    time: float
    position: Point
    battery_current: float
    carried_order_ids: list[str]
    pending_pickups: Dict[str, float]
    pending_dropoffs: Dict[str, float]
    energy_total: float = 0.0

    @property
    def payload_kg(self) -> float:
        return 0.0


def _payload_kg(order_ids: Iterable[str], orders: Dict[str, Order]) -> float:
    return float(sum(orders[order_id].quantity_kg for order_id in order_ids if order_id in orders))


def _future_assigned_ready_ids(drone: Drone, orders: Dict[str, Order]) -> list[str]:
    return [
        order_id
        for order_id in drone.assigned_order_ids
        if order_id in orders
        and order_id not in drone.picked_order_ids
        and orders[order_id].status == OrderStatus.ASSIGNED_READY
    ]


def _future_picked_ids(drone: Drone, orders: Dict[str, Order]) -> list[str]:
    return [order_id for order_id in drone.picked_order_ids if order_id in orders and orders[order_id].status == OrderStatus.PICKED]


def _project_idle_state(
    current_time: float,
    drone: Drone,
    orders: Dict[str, Order],
    weather: WeatherSnapshot,
    cfg: AppConfig,
) -> FutureDroneState:
    available_time = max(current_time, drone.wait_until_time)
    current_payload = _payload_kg(drone.picked_order_ids, orders)
    battery = drone.battery_current
    if available_time > current_time:
        battery -= hover_energy(
            drone,
            weather,
            current_payload,
            available_time - current_time,
            DroneMode.IDLE,
        )
        battery = max(0.0, battery)
    return FutureDroneState(
        drone_id=drone.drone_id,
        available_time=available_time,
        position=drone.position,
        battery_current=battery,
        carried_order_ids=_future_picked_ids(drone, orders),
        assigned_ready_order_ids=_future_assigned_ready_ids(drone, orders),
    )


def _project_flight_to_pickup_state(
    current_time: float,
    drone: Drone,
    order: Order | None,
    orders: Dict[str, Order],
    weather: WeatherSnapshot,
    cfg: AppConfig,
) -> FutureDroneState:
    current_payload = _payload_kg(drone.picked_order_ids, orders)
    battery = drone.battery_current
    available_time = drone.next_available_time
    carried = _future_picked_ids(drone, orders)
    assigned = _future_assigned_ready_ids(drone, orders)
    position = drone.planned_destination or drone.position
    if order is not None and order.status == OrderStatus.ASSIGNED_READY:
        battery -= hover_energy(
            drone,
            weather,
            current_payload,
            drone.pickup_service_mean_min,
            DroneMode.PICKUP_SERVICE,
        )
        battery = max(0.0, battery)
        available_time = drone.next_available_time + drone.pickup_service_mean_min
        position = order.merchant_loc
        if order.order_id not in carried:
            carried.append(order.order_id)
        assigned = [order_id for order_id in assigned if order_id != order.order_id]
    return FutureDroneState(
        drone_id=drone.drone_id,
        available_time=available_time,
        position=position,
        battery_current=battery,
        carried_order_ids=carried,
        assigned_ready_order_ids=assigned,
    )


def _project_pickup_service_state(
    current_time: float,
    drone: Drone,
    order: Order | None,
    orders: Dict[str, Order],
    weather: WeatherSnapshot,
    cfg: AppConfig,
) -> FutureDroneState:
    current_payload = _payload_kg(drone.picked_order_ids, orders)
    remaining_service = max(0.0, drone.next_available_time - current_time)
    battery = drone.battery_current - hover_energy(
        drone,
        weather,
        current_payload,
        remaining_service,
        DroneMode.PICKUP_SERVICE,
    )
    battery = max(0.0, battery)
    carried = _future_picked_ids(drone, orders)
    assigned = _future_assigned_ready_ids(drone, orders)
    if order is not None and order.status == OrderStatus.ASSIGNED_READY:
        if order.order_id not in carried:
            carried.append(order.order_id)
        assigned = [order_id for order_id in assigned if order_id != order.order_id]
    return FutureDroneState(
        drone_id=drone.drone_id,
        available_time=drone.next_available_time,
        position=drone.position,
        battery_current=battery,
        carried_order_ids=carried,
        assigned_ready_order_ids=assigned,
    )


def _project_flight_to_dropoff_state(
    current_time: float,
    drone: Drone,
    order: Order | None,
    orders: Dict[str, Order],
    weather: WeatherSnapshot,
    cfg: AppConfig,
) -> FutureDroneState:
    current_payload = _payload_kg(drone.picked_order_ids, orders)
    battery = drone.battery_current
    carried = _future_picked_ids(drone, orders)
    assigned = _future_assigned_ready_ids(drone, orders)
    position = drone.planned_destination or drone.position
    if order is not None and order.status == OrderStatus.PICKED:
        battery -= hover_energy(
            drone,
            weather,
            current_payload,
            drone.dropoff_service_mean_min,
            DroneMode.DROPOFF_SERVICE,
        )
        battery = max(0.0, battery)
        position = order.customer_loc
        carried = [order_id for order_id in carried if order_id != order.order_id]
        assigned = [order_id for order_id in assigned if order_id != order.order_id]
        available_time = drone.next_available_time + drone.dropoff_service_mean_min
    else:
        available_time = drone.next_available_time
    return FutureDroneState(
        drone_id=drone.drone_id,
        available_time=available_time,
        position=position,
        battery_current=battery,
        carried_order_ids=carried,
        assigned_ready_order_ids=assigned,
    )


def _project_dropoff_service_state(
    current_time: float,
    drone: Drone,
    order: Order | None,
    orders: Dict[str, Order],
    weather: WeatherSnapshot,
    cfg: AppConfig,
) -> FutureDroneState:
    current_payload = _payload_kg(drone.picked_order_ids, orders)
    remaining_service = max(0.0, drone.next_available_time - current_time)
    battery = drone.battery_current - hover_energy(
        drone,
        weather,
        current_payload,
        remaining_service,
        DroneMode.DROPOFF_SERVICE,
    )
    battery = max(0.0, battery)
    carried = _future_picked_ids(drone, orders)
    assigned = _future_assigned_ready_ids(drone, orders)
    if order is not None and order.status == OrderStatus.PICKED:
        carried = [order_id for order_id in carried if order_id != order.order_id]
        assigned = [order_id for order_id in assigned if order_id != order.order_id]
    return FutureDroneState(
        drone_id=drone.drone_id,
        available_time=drone.next_available_time,
        position=drone.position,
        battery_current=battery,
        carried_order_ids=carried,
        assigned_ready_order_ids=assigned,
    )


def _project_charge_state(drone: Drone, stations: list[Station]) -> FutureDroneState:
    if drone.mode == DroneMode.FLYING_TO_STATION:
        station_location = drone.planned_destination or nearest_station(drone.position, stations).location
        available_time = drone.next_available_time + drone.swap_time_min
    else:
        station_location = drone.position
        available_time = drone.next_available_time
    return FutureDroneState(
        drone_id=drone.drone_id,
        available_time=available_time,
        position=station_location,
        battery_current=drone.battery_max,
        carried_order_ids=list(drone.picked_order_ids),
        assigned_ready_order_ids=[],
    )


def project_drone_future_state(
    current_time: float,
    drone: Drone,
    orders: Dict[str, Order],
    stations: list[Station],
    weather: WeatherSnapshot,
    cfg: AppConfig,
) -> FutureDroneState:
    active_order = orders.get(drone.active_order_id) if drone.active_order_id else None
    if drone.mode == DroneMode.IDLE:
        return _project_idle_state(current_time, drone, orders, weather, cfg)
    if drone.mode == DroneMode.FLYING_TO_PICKUP:
        return _project_flight_to_pickup_state(current_time, drone, active_order, orders, weather, cfg)
    if drone.mode == DroneMode.PICKUP_SERVICE:
        return _project_pickup_service_state(current_time, drone, active_order, orders, weather, cfg)
    if drone.mode == DroneMode.FLYING_TO_DROPOFF:
        return _project_flight_to_dropoff_state(current_time, drone, active_order, orders, weather, cfg)
    if drone.mode == DroneMode.DROPOFF_SERVICE:
        return _project_dropoff_service_state(current_time, drone, active_order, orders, weather, cfg)
    if drone.mode in {DroneMode.FLYING_TO_STATION, DroneMode.CHARGING}:
        return _project_charge_state(drone, stations)
    return _project_idle_state(current_time, drone, orders, weather, cfg)


def build_problem(
    current_time: float,
    drones: Dict[str, Drone],
    orders: Dict[str, Order],
    stations: list[Station],
    weather: WeatherSnapshot,
    cfg: AppConfig,
    allow_new_assignments: bool,
) -> SwarmProblem:
    drone_list = list(drones.values())
    future_states = {
        drone.drone_id: project_drone_future_state(current_time, drone, orders, stations, weather, cfg)
        for drone in drone_list
    }
    locked_order_ids = set()
    for state in future_states.values():
        locked_order_ids.update(state.carried_order_ids)
        locked_order_ids.update(state.assigned_ready_order_ids)
    free_order_ids = (
        sorted(order.order_id for order in orders.values() if order.status == OrderStatus.READY_UNASSIGNED)
        if allow_new_assignments
        else []
    )
    optim_order_ids = sorted(locked_order_ids.union(free_order_ids))
    return SwarmProblem(
        current_time=current_time,
        horizon_time=float(cfg.env.simulation_horizon_min),
        drones=drone_list,
        drone_by_id={drone.drone_id: drone for drone in drone_list},
        orders=orders,
        stations=stations,
        weather=weather,
        future_states=future_states,
        free_order_ids=free_order_ids,
        optim_order_ids=optim_order_ids,
    )


def _estimate_cancel_probability(
    current_time: float,
    horizon_time: float,
    order: Order,
    predicted_pickup_finish: float | None,
    predicted_delivery_finish: float | None,
    will_be_assigned: bool,
    cfg: AppConfig,
) -> float:
    if order.status == OrderStatus.PICKED:
        return 0.0
    protected_time = predicted_pickup_finish if predicted_pickup_finish is not None else horizon_time
    delta = max(0.0, min(protected_time, horizon_time) - current_time)
    if delta <= 0:
        return 0.0
    pickup_finish = protected_time
    delivery_finish = predicted_delivery_finish if predicted_delivery_finish is not None else horizon_time
    wait_norm = max(0.0, pickup_finish - order.ready_time) / max(cfg.order.cancel_wait_norm_min, 1e-6)
    eta_norm = max(0.0, delivery_finish - current_time) / max(cfg.order.cancel_eta_norm_min, 1e-6)
    late_norm = max(0.0, delivery_finish - order.deadline) / max(cfg.order.cancel_late_norm_min, 1e-6)
    unassigned = 0.0 if will_be_assigned else 1.0
    base = min(max(order.base_cancel_prob, 1e-4), 1.0 - 1e-4)
    logit = math.log(base / (1.0 - base))
    z = (
        logit
        + cfg.order.cancel_beta_wait * wait_norm
        + cfg.order.cancel_beta_eta * eta_norm
        + cfg.order.cancel_beta_late * late_norm
        + cfg.order.cancel_beta_unassigned * unassigned
        + cfg.order.cancel_beta_release * order.release_count
    )
    hazard = cfg.order.cancel_lambda_max / (1.0 + math.exp(-z))
    return float(1.0 - math.exp(-hazard * delta))


def _ensure_energy(
    state: _DronePlanState,
    drone: Drone,
    orders: Dict[str, Order],
    stations: list[Station],
    weather: WeatherSnapshot,
    cfg: AppConfig,
    required_task_energy: float,
    payload_after_task: float,
) -> bool:
    required = required_task_energy + energy_to_station(state.position, drone, stations, weather, payload_after_task) + drone.battery_safety_margin
    if state.battery_current + 1e-9 >= required:
        return False
    current_payload = _payload_kg(state.carried_order_ids, orders)
    station = nearest_station(state.position, stations)
    travel_time = flight_time_min(
        state.position,
        station.location,
        drone,
        weather,
        current_payload,
    )
    travel_energy = flight_energy(state.position, station.location, drone, weather, current_payload)
    state.time += travel_time + drone.swap_time_min
    state.energy_total += travel_energy
    state.battery_current = drone.battery_max
    state.position = station.location
    state.pending_pickups.clear()
    return True


def _simulate_drone_sequence(
    drone: Drone,
    state: FutureDroneState,
    pickup_keys: Dict[str, float],
    dropoff_keys: Dict[str, float],
    orders: Dict[str, Order],
    stations: list[Station],
    weather: WeatherSnapshot,
    cfg: AppConfig,
    horizon_time: float,
) -> tuple[list[SwarmTask], float, Dict[str, float], Dict[str, float], float, float, set[str]]:
    working = _DronePlanState(
        time=state.available_time,
        position=state.position,
        battery_current=state.battery_current,
        carried_order_ids=list(state.carried_order_ids),
        pending_pickups=dict(pickup_keys),
        pending_dropoffs={order_id: dropoff_keys[order_id] for order_id in state.carried_order_ids if order_id in dropoff_keys},
    )
    for order_id in state.assigned_ready_order_ids:
        if order_id not in working.pending_pickups and order_id in pickup_keys:
            working.pending_pickups[order_id] = pickup_keys[order_id]
    pickup_finish: Dict[str, float] = {}
    delivery_finish: Dict[str, float] = {}
    task_sequence: list[SwarmTask] = []
    unproductive_wait = 0.0
    while working.pending_pickups or working.pending_dropoffs:
        available: list[tuple[float, str, TaskType]] = []
        for order_id, key in working.pending_pickups.items():
            available.append((key, order_id, TaskType.PICKUP))
        for order_id, key in working.pending_dropoffs.items():
            if order_id in working.carried_order_ids:
                available.append((key, order_id, TaskType.DROPOFF))
        if not available:
            break
        available.sort(key=lambda item: (item[0], item[1], item[2].value))
        chosen = None
        charge_triggered = False
        for priority, order_id, task_type in available:
            order = orders[order_id]
            payload_before = _payload_kg(working.carried_order_ids, orders)
            if task_type == TaskType.PICKUP and payload_before + order.quantity_kg > drone.max_capacity_kg + 1e-6:
                continue
            target = order.merchant_loc if task_type == TaskType.PICKUP else order.customer_loc
            payload_after = payload_before + order.quantity_kg if task_type == TaskType.PICKUP else max(0.0, payload_before - order.quantity_kg)
            task_energy = flight_energy(working.position, target, drone, weather, payload_before)
            charged = _ensure_energy(working, drone, orders, stations, weather, cfg, task_energy, payload_after)
            if charged:
                charge_triggered = True
                break
            chosen = (priority, order_id, task_type)
            break
        if charge_triggered:
            continue
        if chosen is None:
            if working.pending_pickups or working.pending_dropoffs:
                # If the decoded sequence leaves active tasks blocked, penalize the stalled tail.
                unproductive_wait += max(0.0, horizon_time - working.time)
            break
        priority, order_id, task_type = chosen
        if task_type == TaskType.PICKUP and order_id not in working.pending_pickups:
            continue
        order = orders[order_id]
        payload_before = _payload_kg(working.carried_order_ids, orders)
        target = order.merchant_loc if task_type == TaskType.PICKUP else order.customer_loc
        travel_time = flight_time_min(
            working.position,
            target,
            drone,
            weather,
            payload_before,
        )
        flight_used = flight_energy(working.position, target, drone, weather, payload_before)
        service_time = drone.pickup_service_mean_min if task_type == TaskType.PICKUP else drone.dropoff_service_mean_min
        hover_used = hover_energy(
            drone,
            weather,
            payload_before,
            service_time,
            DroneMode.PICKUP_SERVICE if task_type == TaskType.PICKUP else DroneMode.DROPOFF_SERVICE,
        )
        finish_time = working.time + travel_time + service_time
        if finish_time > horizon_time + 1e-9:
            unproductive_wait += max(0.0, horizon_time - working.time)
            break
        working.time = finish_time
        working.position = target
        working.battery_current = max(0.0, working.battery_current - flight_used - hover_used)
        working.energy_total += flight_used + hover_used
        task_sequence.append(SwarmTask(order_id=order_id, task_type=task_type, priority=priority))
        if task_type == TaskType.PICKUP:
            pickup_finish[order_id] = finish_time
            if order_id not in working.carried_order_ids:
                working.carried_order_ids.append(order_id)
            working.pending_pickups.pop(order_id, None)
            working.pending_dropoffs[order_id] = dropoff_keys[order_id]
        else:
            delivery_finish[order_id] = finish_time
            working.pending_dropoffs.pop(order_id, None)
            working.carried_order_ids = [item for item in working.carried_order_ids if item != order_id]
    return task_sequence, working.time, pickup_finish, delivery_finish, working.energy_total, unproductive_wait, set(working.pending_pickups.keys())


def _evaluate_priority_plan(
    problem: SwarmProblem,
    cfg: AppConfig,
    assignments: Dict[str, list[str]],
    pickup_keys_by_drone: Dict[str, Dict[str, float]],
    dropoff_keys_by_drone: Dict[str, Dict[str, float]],
) -> SwarmPlan:
    drone_ids = [drone.drone_id for drone in problem.drones]
    task_sequences: Dict[str, list[SwarmTask]] = {drone_id: [] for drone_id in drone_ids}
    pickup_finish_times: Dict[str, float] = {}
    delivery_finish_times: Dict[str, float] = {}
    total_energy = 0.0
    total_unproductive_wait = 0.0
    finish_times: list[float] = [problem.current_time]
    retained_assigned_order_ids: set[str] = set()
    for drone_id in drone_ids:
        state = problem.future_states[drone_id]
        sequence, finish_time, pickup_times, delivery_times, energy, unproductive_wait, retained_pickups = _simulate_drone_sequence(
            drone=problem.drone_by_id[drone_id],
            state=state,
            pickup_keys=pickup_keys_by_drone[drone_id],
            dropoff_keys=dropoff_keys_by_drone[drone_id],
            orders=problem.orders,
            stations=problem.stations,
            weather=problem.weather,
            cfg=cfg,
            horizon_time=problem.horizon_time,
        )
        task_sequences[drone_id] = sequence
        pickup_finish_times.update(pickup_times)
        delivery_finish_times.update(delivery_times)
        total_energy += energy
        total_unproductive_wait += unproductive_wait
        finish_times.append(finish_time)
        retained_assigned_order_ids.update(retained_pickups)

    expected_deliveries = 0.0
    expected_cancellations = 0.0
    total_lateness = 0.0
    planned_pickup_order_ids = {
        task.order_id
        for sequence in task_sequences.values()
        for task in sequence
        if task.task_type == TaskType.PICKUP
    }
    for order_id in problem.optim_order_ids:
        order = problem.orders[order_id]
        if order.status == OrderStatus.PICKED:
            if order_id in delivery_finish_times:
                expected_deliveries += 1.0
                total_lateness += max(0.0, delivery_finish_times[order_id] - order.deadline)
            continue
        assigned_now = order_id in retained_assigned_order_ids or order_id in planned_pickup_order_ids
        p_cancel = _estimate_cancel_probability(
            current_time=problem.current_time,
            horizon_time=problem.horizon_time,
            order=order,
            predicted_pickup_finish=pickup_finish_times.get(order_id),
            predicted_delivery_finish=delivery_finish_times.get(order_id),
            will_be_assigned=assigned_now,
            cfg=cfg,
        )
        expected_cancellations += p_cancel
        if order_id in delivery_finish_times:
            survival = 1.0 - p_cancel
            expected_deliveries += survival
            total_lateness += survival * max(0.0, delivery_finish_times[order_id] - order.deadline)

    makespan_delta = max(0.0, max(finish_times) - problem.current_time)
    objective_cost = (
        -cfg.reward.delivered * expected_deliveries
        + cfg.reward.canceled * expected_cancellations
        + cfg.reward.time_cost * (makespan_delta / max(cfg.rl.time_ref_min, 1e-6))
        + cfg.reward.energy_cost * (total_energy / max(cfg.drone.battery_max, 1e-6))
        + cfg.reward.lateness_cost * (total_lateness / max(cfg.rl.time_ref_min, 1e-6))
    )
    evaluation = PlanEvaluation(
        objective_cost=float(objective_cost),
        expected_deliveries=float(expected_deliveries),
        expected_cancellations=float(expected_cancellations),
        total_lateness=float(total_lateness),
        total_energy=float(total_energy),
        makespan_delta=float(makespan_delta),
        unproductive_wait=float(total_unproductive_wait),
    )
    return SwarmPlan(assignments=assignments, task_sequences=task_sequences, evaluation=evaluation)


def build_plan_from_sequences(
    task_sequences: Dict[str, list[SwarmTask]],
    assignments: Dict[str, list[str]],
    problem: SwarmProblem,
    cfg: AppConfig,
) -> SwarmPlan:
    drone_ids = [drone.drone_id for drone in problem.drones]
    normalized_assignments = {drone_id: list(assignments.get(drone_id, [])) for drone_id in drone_ids}
    pickup_keys_by_drone = {drone_id: {} for drone_id in drone_ids}
    dropoff_keys_by_drone = {drone_id: {} for drone_id in drone_ids}
    for drone_id in drone_ids:
        sequence = task_sequences.get(drone_id, [])
        for priority_index, task in enumerate(sequence):
            priority = float(priority_index)
            if task.task_type == TaskType.PICKUP:
                pickup_keys_by_drone[drone_id][task.order_id] = priority
            elif task.task_type == TaskType.DROPOFF:
                dropoff_keys_by_drone[drone_id][task.order_id] = priority
    return _evaluate_priority_plan(problem, cfg, normalized_assignments, pickup_keys_by_drone, dropoff_keys_by_drone)


def decode_plan(position: np.ndarray, problem: SwarmProblem, cfg: AppConfig) -> SwarmPlan:
    if not problem.optim_order_ids:
        empty_sequences = {drone.drone_id: [] for drone in problem.drones}
        return SwarmPlan(
            assignments={drone.drone_id: [] for drone in problem.drones},
            task_sequences=empty_sequences,
            evaluation=PlanEvaluation(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        )

    matrix = np.asarray(position, dtype=float).reshape(len(problem.optim_order_ids), 3)
    row_by_order = {order_id: idx for idx, order_id in enumerate(problem.optim_order_ids)}
    drone_ids = [drone.drone_id for drone in problem.drones]

    assignments = {drone_id: [] for drone_id in drone_ids}
    pickup_keys_by_drone = {drone_id: {} for drone_id in drone_ids}
    dropoff_keys_by_drone = {drone_id: {} for drone_id in drone_ids}

    for drone_id, state in problem.future_states.items():
        for order_id in state.assigned_ready_order_ids:
            row = matrix[row_by_order[order_id]]
            pickup_keys_by_drone[drone_id][order_id] = float(row[1])
            dropoff_keys_by_drone[drone_id][order_id] = float(row[2])
        for order_id in state.carried_order_ids:
            row = matrix[row_by_order[order_id]]
            dropoff_keys_by_drone[drone_id][order_id] = float(row[2])

    for order_id in problem.free_order_ids:
        row = matrix[row_by_order[order_id]]
        pref = int(np.floor(np.clip(row[0], 0.0, 0.999999) * len(drone_ids)))
        drone_id = drone_ids[pref]
        assignments[drone_id].append(order_id)
        pickup_keys_by_drone[drone_id][order_id] = float(row[1])
        dropoff_keys_by_drone[drone_id][order_id] = float(row[2])

    return _evaluate_priority_plan(problem, cfg, assignments, pickup_keys_by_drone, dropoff_keys_by_drone)


def plan_matches_projection(task: SwarmTask, projection: TaskProjection) -> bool:
    return task.order_id == projection.order_id and task.task_type == projection.task_type
