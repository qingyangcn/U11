from __future__ import annotations

from dataclasses import dataclass

from config.default import AppConfig
from core.entities import Drone, Order, Station, WeatherSnapshot
from core.enums import DroneMode, TaskType
from env.battery import energy_to_station, nearest_station
from env.constraints import TaskProjection
from env.travel import flight_energy, flight_time_min, hover_energy
from swarm.base import BaseSwarmScheduler
from swarm.common import FutureDroneState, SwarmPlan, SwarmTask, build_plan_from_sequences, build_problem, plan_matches_projection


FULL_RULE_BASELINE_POLICIES = (
    "nearest_full",
    "earliest_deadline_full",
    "minimum_slack_full",
)


@dataclass(slots=True)
class _RuleCandidate:
    drone_id: str
    order_id: str
    task_type: TaskType
    score: tuple[float, ...]
    next_state: FutureDroneState


def _payload_kg(state: FutureDroneState, orders: dict[str, Order]) -> float:
    return float(sum(orders[order_id].quantity_kg for order_id in state.carried_order_ids if order_id in orders))


def _project_order_finish_time(
    position,
    drone: Drone,
    order: Order,
    weather: WeatherSnapshot,
    payload_after: float,
    cfg: AppConfig,
    pickup_finish_time: float,
) -> float:
    linehaul = flight_time_min(
        position,
        order.customer_loc,
        drone,
        weather,
        payload_after,
    )
    return pickup_finish_time + linehaul + drone.dropoff_service_mean_min


def _build_candidate(
    policy_name: str,
    drone: Drone,
    state: FutureDroneState,
    order: Order,
    task_type: TaskType,
    stations: list[Station],
    weather: WeatherSnapshot,
    cfg: AppConfig,
    horizon_time: float,
    orders: dict[str, Order],
) -> _RuleCandidate | None:
    payload_before = _payload_kg(state, orders)
    if task_type == TaskType.PICKUP and payload_before + order.quantity_kg > drone.max_capacity_kg + 1e-6:
        return None
    if task_type == TaskType.DROPOFF and order.order_id not in state.carried_order_ids:
        return None

    current_time = state.available_time
    current_position = state.position
    battery_current = state.battery_current
    target = order.merchant_loc if task_type == TaskType.PICKUP else order.customer_loc
    payload_after = payload_before + order.quantity_kg if task_type == TaskType.PICKUP else max(0.0, payload_before - order.quantity_kg)
    service_mode = DroneMode.PICKUP_SERVICE if task_type == TaskType.PICKUP else DroneMode.DROPOFF_SERVICE

    pre_task_delay = 0.0
    task_flight_energy = flight_energy(current_position, target, drone, weather, payload_before)
    service_time = drone.pickup_service_mean_min if task_type == TaskType.PICKUP else drone.dropoff_service_mean_min
    service_hover_energy = hover_energy(
        drone,
        weather,
        payload_before,
        service_time,
        service_mode,
    )
    return_energy = energy_to_station(target, drone, stations, weather, payload_after)
    required_energy = task_flight_energy + service_hover_energy + return_energy + drone.battery_safety_margin

    if battery_current + 1e-9 < required_energy:
        if task_type == TaskType.PICKUP:
            return None
        station = nearest_station(current_position, stations)
        charge_travel_time = flight_time_min(
            current_position,
            station.location,
            drone,
            weather,
            payload_before,
        )
        pre_task_delay += charge_travel_time + drone.swap_time_min
        current_time += pre_task_delay
        current_position = station.location
        battery_current = drone.battery_max
        state = FutureDroneState(
            drone_id=state.drone_id,
            available_time=current_time,
            position=current_position,
            battery_current=battery_current,
            carried_order_ids=list(state.carried_order_ids),
            assigned_ready_order_ids=[],
        )
        task_flight_energy = flight_energy(current_position, target, drone, weather, payload_before)
        service_hover_energy = hover_energy(
            drone,
            weather,
            payload_before,
            service_time,
            service_mode,
        )
        return_energy = energy_to_station(target, drone, stations, weather, payload_after)
        required_energy = task_flight_energy + service_hover_energy + return_energy + drone.battery_safety_margin
        if battery_current + 1e-9 < required_energy:
            return None

    travel_time = flight_time_min(
        current_position,
        target,
        drone,
        weather,
        payload_before,
    )
    finish_time = current_time + travel_time + service_time
    if finish_time > horizon_time + 1e-9:
        return None

    carried_after = list(state.carried_order_ids)
    assigned_after = [order_id for order_id in state.assigned_ready_order_ids if order_id != order.order_id]
    if task_type == TaskType.PICKUP:
        if order.order_id not in carried_after:
            carried_after.append(order.order_id)
        order_finish_time = _project_order_finish_time(target, drone, order, weather, payload_after, cfg, finish_time)
    else:
        carried_after = [order_id for order_id in carried_after if order_id != order.order_id]
        order_finish_time = finish_time

    if policy_name == "nearest_full":
        score = (pre_task_delay + travel_time, finish_time, order.deadline, drone.drone_id, order.order_id)
    elif policy_name == "earliest_deadline_full":
        score = (order.deadline, finish_time, pre_task_delay + travel_time, drone.drone_id, order.order_id)
    else:
        slack = order.deadline - order_finish_time
        score = (slack, order.deadline, finish_time, drone.drone_id, order.order_id)

    next_state = FutureDroneState(
        drone_id=drone.drone_id,
        available_time=finish_time,
        position=target,
        battery_current=max(0.0, battery_current - task_flight_energy - service_hover_energy),
        carried_order_ids=carried_after,
        assigned_ready_order_ids=assigned_after,
    )
    return _RuleCandidate(
        drone_id=drone.drone_id,
        order_id=order.order_id,
        task_type=task_type,
        score=score,
        next_state=next_state,
    )


class GlobalGreedyRuleScheduler(BaseSwarmScheduler):
    def __init__(self, cfg: AppConfig, policy_name: str) -> None:
        if policy_name not in FULL_RULE_BASELINE_POLICIES:
            raise KeyError(f"Unknown full rule baseline policy: {policy_name}")
        self.cfg = cfg
        self.policy_name = policy_name
        self.current_plan: SwarmPlan | None = None

    def reset(self, seed: int | None = None) -> None:
        del seed
        self.current_plan = None

    def replan(
        self,
        current_time: float,
        drones: dict[str, Drone],
        orders: dict[str, Order],
        stations: list[Station],
        weather: WeatherSnapshot,
        allow_new_assignments: bool,
    ) -> SwarmPlan:
        problem = build_problem(
            current_time=current_time,
            drones=drones,
            orders=orders,
            stations=stations,
            weather=weather,
            cfg=self.cfg,
            allow_new_assignments=allow_new_assignments,
        )
        drone_ids = [drone.drone_id for drone in problem.drones]
        task_sequences = {drone_id: [] for drone_id in drone_ids}
        assignments = {drone_id: [] for drone_id in drone_ids}
        future_states = {
            drone_id: FutureDroneState(
                drone_id=state.drone_id,
                available_time=state.available_time,
                position=state.position,
                battery_current=state.battery_current,
                carried_order_ids=list(state.carried_order_ids),
                assigned_ready_order_ids=list(state.assigned_ready_order_ids),
            )
            for drone_id, state in problem.future_states.items()
        }
        free_order_ids = set(problem.free_order_ids)

        while True:
            candidates: list[_RuleCandidate] = []
            for drone in problem.drones:
                state = future_states[drone.drone_id]
                for order_id in sorted(state.carried_order_ids):
                    candidate = _build_candidate(
                        self.policy_name,
                        drone,
                        state,
                        problem.orders[order_id],
                        TaskType.DROPOFF,
                        problem.stations,
                        problem.weather,
                        self.cfg,
                        problem.horizon_time,
                        problem.orders,
                    )
                    if candidate is not None:
                        candidates.append(candidate)
                for order_id in sorted(state.assigned_ready_order_ids):
                    candidate = _build_candidate(
                        self.policy_name,
                        drone,
                        state,
                        problem.orders[order_id],
                        TaskType.PICKUP,
                        problem.stations,
                        problem.weather,
                        self.cfg,
                        problem.horizon_time,
                        problem.orders,
                    )
                    if candidate is not None:
                        candidates.append(candidate)
                for order_id in sorted(free_order_ids):
                    candidate = _build_candidate(
                        self.policy_name,
                        drone,
                        state,
                        problem.orders[order_id],
                        TaskType.PICKUP,
                        problem.stations,
                        problem.weather,
                        self.cfg,
                        problem.horizon_time,
                        problem.orders,
                    )
                    if candidate is not None:
                        candidates.append(candidate)

            if not candidates:
                break

            chosen = min(candidates, key=lambda candidate: candidate.score)
            sequence = task_sequences[chosen.drone_id]
            sequence.append(
                SwarmTask(
                    order_id=chosen.order_id,
                    task_type=chosen.task_type,
                    priority=float(len(sequence)),
                )
            )
            future_states[chosen.drone_id] = chosen.next_state
            if chosen.task_type == TaskType.PICKUP and chosen.order_id in free_order_ids:
                free_order_ids.remove(chosen.order_id)
                assignments[chosen.drone_id].append(chosen.order_id)

        self.current_plan = build_plan_from_sequences(task_sequences, assignments, problem, self.cfg)
        return self.current_plan

    def next_projection(
        self,
        drone_id: str,
        feasible_projections: dict[str, TaskProjection],
        orders: dict[str, Order],
    ) -> TaskProjection | None:
        del orders
        if self.current_plan is None:
            return None
        for task in self.current_plan.task_sequences.get(drone_id, []):
            projection = feasible_projections.get(task.order_id)
            if projection is not None and plan_matches_projection(task, projection):
                return projection
        return None


class NearestFullScheduler(GlobalGreedyRuleScheduler):
    def __init__(self, cfg: AppConfig) -> None:
        super().__init__(cfg, policy_name="nearest_full")


class EarliestDeadlineFullScheduler(GlobalGreedyRuleScheduler):
    def __init__(self, cfg: AppConfig) -> None:
        super().__init__(cfg, policy_name="earliest_deadline_full")


class MinimumSlackFullScheduler(GlobalGreedyRuleScheduler):
    def __init__(self, cfg: AppConfig) -> None:
        super().__init__(cfg, policy_name="minimum_slack_full")
