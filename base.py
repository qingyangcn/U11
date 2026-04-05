from __future__ import annotations

from abc import ABC, abstractmethod

from core.entities import Drone, Order, Station, WeatherSnapshot
from env.constraints import TaskProjection


class BaseSwarmScheduler(ABC):
    @abstractmethod
    def reset(self, seed: int | None = None) -> None:
        raise NotImplementedError

    @abstractmethod
    def replan(
        self,
        current_time: float,
        drones: dict[str, Drone],
        orders: dict[str, Order],
        stations: list[Station],
        weather: WeatherSnapshot,
        allow_new_assignments: bool,
    ):
        raise NotImplementedError

    @abstractmethod
    def next_projection(
        self,
        drone_id: str,
        feasible_projections: dict[str, TaskProjection],
        orders: dict[str, Order],
    ) -> TaskProjection | None:
        raise NotImplementedError
