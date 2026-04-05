from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from core.enums import DroneMode, EventType, OrderStatus, TaskType, WeatherState


@dataclass(slots=True)
class Point:
    x: float
    y: float
    lon: float | None = None
    lat: float | None = None


@dataclass(slots=True)
class Station:
    station_id: str
    location: Point


@dataclass(slots=True)
class WeatherSnapshot:
    state: WeatherState
    speed_factor: float
    energy_factor: float
    hover_factor: float
    wind_speed_kmph: float
    humidity: float
    summary: str = ""


@dataclass(slots=True)
class Order:
    order_id: str
    created_time: float
    ready_time: float
    deadline: float
    merchant_loc: Point
    customer_loc: Point
    quantity_kg: float
    food_quantity: int
    dessert_quantity: int
    drink_quantity: int
    base_cancel_prob: float
    status: OrderStatus = OrderStatus.NOT_READY
    assigned_drone_id: str | None = None
    release_count: int = 0
    picked_time: float | None = None
    delivered_time: float | None = None
    canceled_time: float | None = None
    last_cancel_eval_time: float | None = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def cancelable(self) -> bool:
        return self.status in {OrderStatus.READY_UNASSIGNED, OrderStatus.ASSIGNED_READY}


@dataclass(slots=True)
class Drone:
    drone_id: str
    home_station_id: str
    home_station_loc: Point
    position: Point
    max_speed_km_min: float
    max_capacity_kg: float
    pickup_service_mean_min: float
    dropoff_service_mean_min: float
    pickup_service_shape: float
    dropoff_service_shape: float
    swap_time_min: float
    battery_max: float
    battery_current: float
    battery_safety_margin: float
    cruise_power_empty_w: float
    cruise_power_full_w: float
    hover_power_empty_w: float
    hover_power_full_w: float
    min_speed_ratio: float
    speed_load_penalty: float
    mode: DroneMode = DroneMode.IDLE
    assigned_order_ids: List[str] = field(default_factory=list)
    picked_order_ids: List[str] = field(default_factory=list)
    next_available_time: float = 0.0
    wait_until_time: float = 0.0
    planned_destination: Point | None = None
    active_order_id: str | None = None
    active_task_type: TaskType | None = None

    @property
    def onboard_count(self) -> int:
        return len(self.picked_order_ids)

    def payload_sum(self, orders: Dict[str, Order]) -> float:
        return sum(orders[oid].quantity_kg for oid in self.picked_order_ids if oid in orders)


@dataclass(slots=True)
class Task:
    task_type: TaskType
    order_id: str | None
    target: Point | None
    eta_if_choose: float = 0.0
    lateness_if_choose: float = 0.0
    cancel_risk: float = 0.0


@dataclass(slots=True)
class Event:
    time: float
    event_type: EventType
    payload: Dict[str, Any] | None = None
    priority: int = 100


@dataclass(slots=True)
class DecisionContext:
    drone_id: str
    time: float


@dataclass(slots=True)
class StepMetrics:
    delivered_count: int = 0
    canceled_count: int = 0
    delta_time: float = 0.0
    delta_energy: float = 0.0
    lateness_total: float = 0.0

    def as_dict(self) -> Dict[str, float]:
        return {
            "delivered_count": float(self.delivered_count),
            "canceled_count": float(self.canceled_count),
            "delta_time": self.delta_time,
            "delta_energy": self.delta_energy,
            "lateness_total": self.lateness_total,
        }
