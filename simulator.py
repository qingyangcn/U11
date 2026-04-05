from __future__ import annotations

import hashlib
import math
from collections import deque
from typing import Callable, Deque, Dict, Iterable, Optional

from allocator.base import BaseAllocator
from allocator.mopso_solver import MOPSOSolver
from config.default import AppConfig, build_default_config
from core.entities import DecisionContext, Drone, Event, Order, Point, Station, StepMetrics, WeatherSnapshot
from core.enums import DroneMode, EventType, OrderStatus, TaskType
from core.utils import clip, safe_div, sample_gamma_by_mean
from env.battery import lower_bound_energy, nearest_station
from env.constraints import TaskProjection, collect_feasible_task_projections, must_force_charge
from env.event import EventQueue
from env.generator import ScenarioGenerator
from env.travel import flight_energy, flight_time_min, hover_energy
from env.weather import WeatherProcess
from swarm.base import BaseSwarmScheduler


RuleSelector = Callable[[str, Dict[str, TaskProjection], Dict[str, Order]], TaskProjection | None]


class EventDrivenSimulator:
    def __init__(
        self,
        cfg: AppConfig | None = None,
        allocator: BaseAllocator | None = None,
        swarm_scheduler: BaseSwarmScheduler | None = None,
        seed: int | None = None,
    ) -> None:
        self.cfg = cfg or build_default_config()
        self.generator = ScenarioGenerator(self.cfg, seed=seed)
        self.allocator = allocator or MOPSOSolver(self.cfg, seed=seed)
        self.swarm_scheduler = swarm_scheduler
        self.weather_process = WeatherProcess(self.cfg)

        self.orders: Dict[str, Order] = {}
        self.drones: Dict[str, Drone] = {}
        self.stations: list[Station] = []
        self.current_time = 0.0
        self.current_weather: WeatherSnapshot | None = None
        self.event_queue = EventQueue()
        self.pending_decisions: Deque[str] = deque()
        self.current_decision: DecisionContext | None = None
        self.done = False
        self.episode_stats: Dict[str, float] = {}
        self._active_metrics: StepMetrics | None = None
        self.x_bounds = (0.0, 1.0)
        self.y_bounds = (0.0, 1.0)
        self._decision_order_seed = int(seed if seed is not None else self.cfg.training.seed)

    @property
    def business_duration_min(self) -> float:
        return float((self.cfg.order.business_end_hour - self.cfg.order.business_start_hour) * 60.0)

    @property
    def terminal_time_min(self) -> float:
        return float(self.cfg.env.simulation_horizon_min)

    def reset(self, seed: int | None = None) -> None:
        base_seed = seed if seed is not None else self.cfg.training.seed
        self.generator.reseed(seed)
        self._decision_order_seed = int(base_seed)
        if self.swarm_scheduler is not None:
            self.swarm_scheduler.reset(seed=seed)
        self.orders = self.generator.build_orders()
        self.drones = self.generator.build_drones()
        self.stations = list(self.generator.stations)
        spatial = self.generator.artifacts.spatial_model
        self.x_bounds = (float(spatial["x_bounds"][0]), float(spatial["x_bounds"][1]))
        self.y_bounds = (float(spatial["y_bounds"][0]), float(spatial["y_bounds"][1]))

        self.current_time = 0.0
        self.event_queue.clear()
        self.pending_decisions.clear()
        self.current_decision = None
        self.done = False
        self.episode_stats = {
            "total_orders": float(len(self.orders)),
            "delivered_orders": 0.0,
            "canceled_orders": 0.0,
            "total_lateness_min": 0.0,
            "total_energy": 0.0,
            "total_wait_min": 0.0,
            "total_idle_hover_time_min": 0.0,
            "charge_count": 0.0,
            "forced_charge_count": 0.0,
            "released_orders_count": 0.0,
            "total_charge_time_min": 0.0,
            "total_flight_time_min": 0.0,
            "total_pickup_service_time_min": 0.0,
            "total_dropoff_service_time_min": 0.0,
            "total_hover_energy": 0.0,
            "total_flight_energy": 0.0,
        }

        self.weather_process.reset(self.generator.rng, self.terminal_time_min)
        self.current_weather = self.weather_process.snapshot_at(0.0)
        self._schedule_initial_events()
        self._advance_until_decision_or_done(collect_metrics=False)

    def _schedule_initial_events(self) -> None:
        for order in self.orders.values():
            self.event_queue.push(Event(time=order.created_time, event_type=EventType.ORDER_ARRIVAL, payload={"order_id": order.order_id}))
            self.event_queue.push(Event(time=order.ready_time, event_type=EventType.ORDER_READY, payload={"order_id": order.order_id}))
        t = 0.0
        while t <= self.terminal_time_min + 1e-6:
            self.event_queue.push(Event(time=t, event_type=EventType.DISPATCH_EPOCH))
            t += self.cfg.dispatch.interval_min
        for change_time, _ in self.weather_process.schedule[1:]:
            self.event_queue.push(Event(time=change_time, event_type=EventType.WEATHER_CHANGE))

    def _advance_time(self, new_time: float, collect_metrics: bool) -> None:
        delta = max(0.0, new_time - self.current_time)
        if delta <= 0:
            self.current_time = new_time
            return
        if collect_metrics and self._active_metrics is not None:
            self._active_metrics.delta_time += delta
        for drone in self.drones.values():
            if drone.mode in {DroneMode.IDLE, DroneMode.PICKUP_SERVICE, DroneMode.DROPOFF_SERVICE}:
                payload = drone.payload_sum(self.orders)
                if drone.mode == DroneMode.IDLE:
                    self.episode_stats["total_idle_hover_time_min"] += delta
                energy = hover_energy(
                    drone=drone,
                    weather=self.current_weather,
                    payload_kg=payload,
                    duration_min=delta,
                    mode=drone.mode,
                )
                drone.battery_current = max(0.0, drone.battery_current - energy)
                self.episode_stats["total_energy"] += energy
                self.episode_stats["total_hover_energy"] += energy
                if collect_metrics and self._active_metrics is not None:
                    self._active_metrics.delta_energy += energy
        self.current_time = new_time

    def _advance_until_decision_or_done(self, collect_metrics: bool) -> StepMetrics:
        metrics = StepMetrics()
        previous_metrics = self._active_metrics
        self._active_metrics = metrics if collect_metrics else None
        try:
            while True:
                if self._has_current_time_events():
                    self.pending_decisions.clear()
                    self._drain_current_time_events()
                    self._collect_decision_batch()
                    continue
                if self.pending_decisions:
                    drone_id = self.pending_decisions.popleft()
                    drone = self.drones[drone_id]
                    if drone.mode != DroneMode.IDLE:
                        continue
                    if self.current_time + 1e-9 < drone.wait_until_time:
                        continue
                    if self._maybe_force_charge(drone):
                        continue
                    if not self.get_feasible_projections(drone.drone_id):
                        continue
                    self.current_decision = DecisionContext(drone_id=drone_id, time=self.current_time)
                    return metrics
                if self._check_done():
                    self.done = True
                    self.current_decision = None
                    return metrics
                if self.event_queue.empty():
                    self.done = True
                    self.current_decision = None
                    return metrics
                self._advance_time(self.event_queue.peek().time, collect_metrics)
                self._drain_current_time_events()
                self._collect_decision_batch()
        finally:
            self._active_metrics = previous_metrics

    def _has_current_time_events(self) -> bool:
        event = self.event_queue.peek()
        return event is not None and event.time <= self.current_time + 1e-9

    def _drain_current_time_events(self) -> None:
        while self._has_current_time_events():
            self._handle_event(self.event_queue.pop())

    def _collect_decision_batch(self) -> None:
        candidates: list[str] = []
        self.pending_decisions.clear()
        for drone in self.drones.values():
            if drone.mode != DroneMode.IDLE:
                continue
            if self.current_time + 1e-9 < drone.wait_until_time:
                continue
            if self._maybe_force_charge(drone):
                continue
            if self.get_feasible_projections(drone.drone_id):
                candidates.append(drone.drone_id)
        self.pending_decisions.extend(self._order_decision_batch(candidates))

    def _order_decision_batch(self, drone_ids: list[str]) -> list[str]:
        time_token = f"{round(self.current_time, 9):.9f}"

        def key(drone_id: str) -> tuple[str, str]:
            token = f"{self._decision_order_seed}|{time_token}|{drone_id}"
            digest = hashlib.sha256(token.encode("ascii")).hexdigest()
            return digest, drone_id

        return sorted(drone_ids, key=key)

    def _handle_event(self, event: Event) -> None:
        if event.event_type == EventType.ORDER_ARRIVAL:
            order = self.orders[event.payload["order_id"]]
            order.metadata["announced"] = True
            return
        if event.event_type == EventType.ORDER_READY:
            order = self.orders[event.payload["order_id"]]
            if order.status == OrderStatus.NOT_READY and order.metadata.get("announced", False):
                order.status = OrderStatus.READY_UNASSIGNED
                order.last_cancel_eval_time = self.current_time
            return
        if event.event_type == EventType.WEATHER_CHANGE:
            self.current_weather = self.weather_process.snapshot_at(self.current_time)
            self._evaluate_cancellations(trigger="weather_change")
            if self.swarm_scheduler is not None:
                self._run_swarm_replan(assign_new_orders=False)
            return
        if event.event_type == EventType.DISPATCH_EPOCH:
            self._evaluate_cancellations(trigger="dispatch_epoch")
            if self.swarm_scheduler is not None:
                self._run_swarm_replan(assign_new_orders=True)
            else:
                self._run_dispatch()
            return
        if event.event_type == EventType.ARRIVE_MERCHANT:
            self._handle_arrive_merchant(event.payload["drone_id"], event.payload["order_id"])
            return
        if event.event_type == EventType.PICKUP_FINISH:
            self._handle_pickup_finish(event.payload["drone_id"], event.payload["order_id"])
            return
        if event.event_type == EventType.ARRIVE_CUSTOMER:
            self._handle_arrive_customer(event.payload["drone_id"], event.payload["order_id"])
            return
        if event.event_type == EventType.DROPOFF_FINISH:
            self._handle_dropoff_finish(event.payload["drone_id"], event.payload["order_id"])
            return
        if event.event_type == EventType.ARRIVE_STATION:
            self._handle_arrive_station(event.payload["drone_id"])
            return
        if event.event_type == EventType.CHARGE_FINISH:
            self._handle_charge_finish(event.payload["drone_id"])
            return

    def _run_dispatch(self) -> None:
        ready_pool = [order for order in self.orders.values() if order.status == OrderStatus.READY_UNASSIGNED]
        if ready_pool:
            assignments = self.allocator.assign(
                current_time=self.current_time,
                drones=self.drones.values(),
                orders=self.orders.values(),
                stations=self.stations,
                weather=self.current_weather,
            )
            for drone_id, order_ids in assignments.items():
                drone = self.drones[drone_id]
                for order_id in order_ids:
                    order = self.orders[order_id]
                    if order.status != OrderStatus.READY_UNASSIGNED:
                        continue
                    order.status = OrderStatus.ASSIGNED_READY
                    order.assigned_drone_id = drone_id
                    order.last_cancel_eval_time = self.current_time
                    if order_id not in drone.assigned_order_ids:
                        drone.assigned_order_ids.append(order_id)

    def _run_swarm_replan(self, assign_new_orders: bool) -> None:
        if self.swarm_scheduler is None:
            return
        plan = self.swarm_scheduler.replan(
            current_time=self.current_time,
            drones=self.drones,
            orders=self.orders,
            stations=self.stations,
            weather=self.current_weather,
            allow_new_assignments=assign_new_orders,
        )
        if assign_new_orders:
            for drone_id, order_ids in plan.assignments.items():
                drone = self.drones[drone_id]
                for order_id in order_ids:
                    order = self.orders.get(order_id)
                    if order is None or order.status != OrderStatus.READY_UNASSIGNED:
                        continue
                    order.status = OrderStatus.ASSIGNED_READY
                    order.assigned_drone_id = drone_id
                    order.last_cancel_eval_time = self.current_time
                    if order_id not in drone.assigned_order_ids:
                        drone.assigned_order_ids.append(order_id)

    def _handle_arrive_merchant(self, drone_id: str, order_id: str) -> None:
        drone = self.drones[drone_id]
        order = self.orders[order_id]
        drone.position = order.merchant_loc
        drone.planned_destination = None
        if order.status != OrderStatus.ASSIGNED_READY:
            drone.mode = DroneMode.IDLE
            drone.active_order_id = None
            drone.active_task_type = None
            return
        drone.mode = DroneMode.PICKUP_SERVICE
        drone.active_order_id = order_id
        drone.active_task_type = TaskType.PICKUP
        service_time = sample_gamma_by_mean(self.generator.rng, drone.pickup_service_mean_min, drone.pickup_service_shape)
        self.episode_stats["total_pickup_service_time_min"] += service_time
        drone.next_available_time = self.current_time + service_time
        self.event_queue.push(
            Event(
                time=self.current_time + service_time,
                event_type=EventType.PICKUP_FINISH,
                payload={"drone_id": drone_id, "order_id": order_id},
            )
        )

    def _handle_pickup_finish(self, drone_id: str, order_id: str) -> None:
        drone = self.drones[drone_id]
        order = self.orders[order_id]
        if order.status != OrderStatus.ASSIGNED_READY:
            drone.mode = DroneMode.IDLE
            drone.active_order_id = None
            drone.active_task_type = None
            return
        order.status = OrderStatus.PICKED
        order.picked_time = self.current_time
        if order_id not in drone.picked_order_ids:
            drone.picked_order_ids.append(order_id)
        drone.mode = DroneMode.IDLE
        drone.next_available_time = self.current_time
        drone.active_order_id = None
        drone.active_task_type = None

    def _handle_arrive_customer(self, drone_id: str, order_id: str) -> None:
        drone = self.drones[drone_id]
        order = self.orders[order_id]
        drone.position = order.customer_loc
        drone.planned_destination = None
        if order.status != OrderStatus.PICKED:
            drone.mode = DroneMode.IDLE
            drone.active_order_id = None
            drone.active_task_type = None
            return
        drone.mode = DroneMode.DROPOFF_SERVICE
        drone.active_order_id = order_id
        drone.active_task_type = TaskType.DROPOFF
        service_time = sample_gamma_by_mean(self.generator.rng, drone.dropoff_service_mean_min, drone.dropoff_service_shape)
        self.episode_stats["total_dropoff_service_time_min"] += service_time
        drone.next_available_time = self.current_time + service_time
        self.event_queue.push(
            Event(
                time=self.current_time + service_time,
                event_type=EventType.DROPOFF_FINISH,
                payload={"drone_id": drone_id, "order_id": order_id},
            )
        )

    def _handle_dropoff_finish(self, drone_id: str, order_id: str) -> None:
        drone = self.drones[drone_id]
        order = self.orders[order_id]
        if order.status != OrderStatus.PICKED:
            drone.mode = DroneMode.IDLE
            drone.active_order_id = None
            drone.active_task_type = None
            return
        order.status = OrderStatus.DELIVERED
        order.delivered_time = self.current_time
        if order_id in drone.picked_order_ids:
            drone.picked_order_ids.remove(order_id)
        if order_id in drone.assigned_order_ids:
            drone.assigned_order_ids.remove(order_id)
        drone.mode = DroneMode.IDLE
        drone.next_available_time = self.current_time
        drone.active_order_id = None
        drone.active_task_type = None

        lateness = max(0.0, self.current_time - order.deadline)
        self.episode_stats["delivered_orders"] += 1.0
        self.episode_stats["total_lateness_min"] += lateness
        if self._active_metrics is not None:
            self._active_metrics.delivered_count += 1
            self._active_metrics.lateness_total += lateness

    def _handle_arrive_station(self, drone_id: str) -> None:
        drone = self.drones[drone_id]
        station = nearest_station(drone.position, self.stations)
        drone.position = station.location
        drone.planned_destination = None
        drone.active_order_id = None
        drone.active_task_type = None
        drone.mode = DroneMode.CHARGING
        self.episode_stats["charge_count"] += 1.0
        self.episode_stats["total_charge_time_min"] += drone.swap_time_min
        drone.next_available_time = self.current_time + drone.swap_time_min
        self.event_queue.push(
            Event(
                time=self.current_time + drone.swap_time_min,
                event_type=EventType.CHARGE_FINISH,
                payload={"drone_id": drone_id},
            )
        )

    def _handle_charge_finish(self, drone_id: str) -> None:
        drone = self.drones[drone_id]
        drone.battery_current = drone.battery_max
        drone.mode = DroneMode.IDLE
        drone.next_available_time = self.current_time
        drone.active_order_id = None
        drone.active_task_type = None

    def _maybe_force_charge(self, drone: Drone) -> bool:
        if drone.mode != DroneMode.IDLE:
            return False
        if not must_force_charge(self.current_time, drone, self.orders, self.stations, self.current_weather, self.cfg):
            return False
        self._send_to_charge(drone)
        return True

    def _send_to_charge(self, drone: Drone) -> None:
        drone.wait_until_time = 0.0
        released = [oid for oid in drone.assigned_order_ids if oid not in drone.picked_order_ids]
        self.episode_stats["forced_charge_count"] += 1.0
        self.episode_stats["released_orders_count"] += float(len(released))
        for order_id in released:
            order = self.orders[order_id]
            if order.status == OrderStatus.ASSIGNED_READY:
                order.status = OrderStatus.READY_UNASSIGNED
                order.assigned_drone_id = None
                order.release_count += 1
                order.last_cancel_eval_time = self.current_time
        drone.assigned_order_ids = [oid for oid in drone.assigned_order_ids if oid in drone.picked_order_ids]
        drone.active_order_id = None
        drone.active_task_type = None

        station = nearest_station(drone.position, self.stations)
        payload = drone.payload_sum(self.orders)
        travel_time = flight_time_min(
            drone.position,
            station.location,
            drone,
            self.current_weather,
            payload,
        )
        energy = flight_energy(drone.position, station.location, drone, self.current_weather, payload)
        drone.battery_current = max(0.0, drone.battery_current - energy)
        drone.mode = DroneMode.FLYING_TO_STATION
        drone.next_available_time = self.current_time + travel_time
        drone.planned_destination = station.location
        self.episode_stats["total_energy"] += energy
        self.episode_stats["total_flight_energy"] += energy
        self.episode_stats["total_flight_time_min"] += travel_time
        if self._active_metrics is not None:
            self._active_metrics.delta_energy += energy
        self.event_queue.push(
            Event(
                time=self.current_time + travel_time,
                event_type=EventType.ARRIVE_STATION,
                payload={"drone_id": drone.drone_id},
            )
        )

    def get_feasible_projections(self, drone_id: str) -> Dict[str, TaskProjection]:
        drone = self.drones[drone_id]
        projections = collect_feasible_task_projections(
            now=self.current_time,
            drone=drone,
            orders=self.orders,
            stations=self.stations,
            weather=self.current_weather,
            cfg=self.cfg,
        )
        return {projection.order_id: projection for projection in projections}

    def get_battery_lower_bound(self, drone_id: str) -> float:
        drone = self.drones[drone_id]
        payload = drone.payload_sum(self.orders)
        return lower_bound_energy(drone, self.stations, self.current_weather, payload, self.cfg)

    def get_force_charge_flag(self, drone_id: str) -> bool:
        return must_force_charge(
            self.current_time,
            self.drones[drone_id],
            self.orders,
            self.stations,
            self.current_weather,
            self.cfg,
        )

    def get_cancel_risk_lookup(self, order_ids: Iterable[str]) -> Dict[str, float]:
        return {order_id: self._current_cancel_probability(self.orders[order_id], self.cfg.dispatch.interval_min) for order_id in order_ids}

    def step(self, action_name: str, rule_selector: RuleSelector) -> StepMetrics:
        if self.done or self.current_decision is None:
            self.done = True
            return StepMetrics()

        context = self.current_decision
        self.current_decision = None
        drone = self.drones[context.drone_id]
        drone.wait_until_time = 0.0
        projections = self.get_feasible_projections(drone.drone_id)

        if self._maybe_force_charge(drone):
            return self._advance_until_decision_or_done(collect_metrics=True)

        selected = rule_selector(action_name, projections, self.orders)
        if selected is None:
            self._apply_wait(drone)
            return self._advance_until_decision_or_done(collect_metrics=True)

        self._launch_task(drone, selected)
        return self._advance_until_decision_or_done(collect_metrics=True)

    def step_swarm(self) -> StepMetrics:
        if self.swarm_scheduler is None:
            raise RuntimeError("Pure swarm stepping requires a swarm_scheduler.")
        if self.done or self.current_decision is None:
            self.done = True
            return StepMetrics()

        context = self.current_decision
        self.current_decision = None
        drone = self.drones[context.drone_id]
        drone.wait_until_time = 0.0
        projections = self.get_feasible_projections(drone.drone_id)

        if self._maybe_force_charge(drone):
            self._run_swarm_replan(assign_new_orders=False)
            return self._advance_until_decision_or_done(collect_metrics=True)

        selected = self.swarm_scheduler.next_projection(drone.drone_id, projections, self.orders)
        if selected is None:
            self._apply_wait(drone)
            return self._advance_until_decision_or_done(collect_metrics=True)

        self._launch_task(drone, selected)
        return self._advance_until_decision_or_done(collect_metrics=True)

    def _apply_wait(self, drone: Drone) -> None:
        next_dispatch = (math.floor(self.current_time / self.cfg.dispatch.interval_min) + 1) * self.cfg.dispatch.interval_min
        wait_end = min(next_dispatch, self.terminal_time_min)
        drone.mode = DroneMode.IDLE
        drone.next_available_time = self.current_time
        drone.wait_until_time = wait_end
        self.episode_stats["total_wait_min"] += max(0.0, wait_end - self.current_time)

    def _launch_task(self, drone: Drone, projection: TaskProjection) -> None:
        order = self.orders[projection.order_id]
        drone.wait_until_time = 0.0
        drone.active_order_id = projection.order_id
        drone.active_task_type = projection.task_type
        target = order.merchant_loc if projection.task_type == TaskType.PICKUP else order.customer_loc
        payload = drone.payload_sum(self.orders)
        travel_time = flight_time_min(
            drone.position,
            target,
            drone,
            self.current_weather,
            payload,
        )
        energy = flight_energy(drone.position, target, drone, self.current_weather, payload)
        drone.battery_current = max(0.0, drone.battery_current - energy)
        drone.mode = DroneMode.FLYING_TO_PICKUP if projection.task_type == TaskType.PICKUP else DroneMode.FLYING_TO_DROPOFF
        drone.next_available_time = self.current_time + travel_time
        drone.planned_destination = target
        self.episode_stats["total_energy"] += energy
        self.episode_stats["total_flight_energy"] += energy
        self.episode_stats["total_flight_time_min"] += travel_time
        if self._active_metrics is not None:
            self._active_metrics.delta_energy += energy
        self.event_queue.push(
            Event(
                time=self.current_time + travel_time,
                event_type=EventType.ARRIVE_MERCHANT if projection.task_type == TaskType.PICKUP else EventType.ARRIVE_CUSTOMER,
                payload={"drone_id": drone.drone_id, "order_id": projection.order_id},
            )
        )

    def _evaluate_cancellations(self, trigger: str) -> None:
        del trigger
        for order in list(self.orders.values()):
            if not order.cancelable:
                continue
            last_eval_time = self.current_time if order.last_cancel_eval_time is None else order.last_cancel_eval_time
            probability = self._current_cancel_probability(order, max(0.0, self.current_time - last_eval_time))
            order.last_cancel_eval_time = self.current_time
            if self.generator.rng.random() < probability:
                self._cancel_order(order)

    def _current_cancel_probability(self, order: Order, delta_time: float) -> float:
        if not order.cancelable:
            return 0.0
        wait_norm = max(0.0, self.current_time - order.ready_time) / max(self.cfg.order.cancel_wait_norm_min, 1e-6)
        eta = self._estimate_order_eta(order)
        eta_norm = eta / max(self.cfg.order.cancel_eta_norm_min, 1e-6)
        late_norm = max(0.0, self.current_time + eta - order.deadline) / max(self.cfg.order.cancel_late_norm_min, 1e-6)
        unassigned = 1.0 if order.status == OrderStatus.READY_UNASSIGNED else 0.0
        base = clip(order.base_cancel_prob, 1e-4, 1.0 - 1e-4)
        logit = math.log(base / (1.0 - base))
        z = (
            logit
            + self.cfg.order.cancel_beta_wait * wait_norm
            + self.cfg.order.cancel_beta_eta * eta_norm
            + self.cfg.order.cancel_beta_late * late_norm
            + self.cfg.order.cancel_beta_unassigned * unassigned
            + self.cfg.order.cancel_beta_release * order.release_count
        )
        hazard = self.cfg.order.cancel_lambda_max / (1.0 + math.exp(-z))
        delta = max(delta_time, 0.0)
        return float(1.0 - math.exp(-hazard * delta))

    def _estimate_order_eta(self, order: Order) -> float:
        if order.status == OrderStatus.ASSIGNED_READY and order.assigned_drone_id in self.drones:
            drone = self.drones[order.assigned_drone_id]
            start = max(self.current_time, drone.next_available_time)
            anchor = drone.planned_destination or drone.position
            access = flight_time_min(
                anchor,
                order.merchant_loc,
                drone,
                self.current_weather,
                drone.payload_sum(self.orders),
            )
            linehaul = flight_time_min(
                order.merchant_loc,
                order.customer_loc,
                drone,
                self.current_weather,
                drone.payload_sum(self.orders) + order.quantity_kg,
            )
            return max(0.0, start - self.current_time) + access + drone.pickup_service_mean_min + linehaul + drone.dropoff_service_mean_min
        best = float("inf")
        for drone in self.drones.values():
            anchor = drone.planned_destination or drone.position
            access = flight_time_min(
                anchor,
                order.merchant_loc,
                drone,
                self.current_weather,
                drone.payload_sum(self.orders),
            )
            linehaul = flight_time_min(
                order.merchant_loc,
                order.customer_loc,
                drone,
                self.current_weather,
                drone.payload_sum(self.orders) + order.quantity_kg,
            )
            eta = max(0.0, drone.next_available_time - self.current_time) + access + drone.pickup_service_mean_min + linehaul + drone.dropoff_service_mean_min
            best = min(best, eta)
        return 0.0 if best == float("inf") else best

    def _cancel_order(self, order: Order) -> None:
        order.status = OrderStatus.CANCELED
        order.canceled_time = self.current_time
        self.episode_stats["canceled_orders"] += 1.0
        if self._active_metrics is not None:
            self._active_metrics.canceled_count += 1
        if order.assigned_drone_id and order.assigned_drone_id in self.drones:
            drone = self.drones[order.assigned_drone_id]
            if order.order_id in drone.assigned_order_ids:
                drone.assigned_order_ids.remove(order.order_id)
        order.assigned_drone_id = None

    def _check_done(self) -> bool:
        if self.current_time >= self.terminal_time_min:
            return True
        if self.current_time < self.business_duration_min:
            return False
        unresolved = any(
            order.status not in {OrderStatus.DELIVERED, OrderStatus.CANCELED}
            for order in self.orders.values()
        )
        active_drone = any(drone.mode != DroneMode.IDLE for drone in self.drones.values())
        return (not unresolved) and (not active_drone)
