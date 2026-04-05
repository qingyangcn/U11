from __future__ import annotations

from typing import Iterable

from config.default import AppConfig
from core.entities import Drone, Order, Point, Station, WeatherSnapshot
from env.travel import flight_energy


def nearest_station(position: Point, stations: Iterable[Station]) -> Station:
    stations = list(stations)
    if not stations:
        raise ValueError("At least one station is required.")
    return min(stations, key=lambda station: (position.x - station.location.x) ** 2 + (position.y - station.location.y) ** 2)


def energy_to_station(
    position: Point,
    drone: Drone,
    stations: Iterable[Station],
    weather: WeatherSnapshot,
    payload_kg: float,
) -> float:
    station = nearest_station(position, stations)
    return flight_energy(position, station.location, drone, weather, payload_kg)


def lower_bound_energy(
    drone: Drone,
    stations: Iterable[Station],
    weather: WeatherSnapshot,
    payload_kg: float,
    cfg: AppConfig,
) -> float:
    del cfg
    return energy_to_station(drone.position, drone, stations, weather, payload_kg) + drone.battery_safety_margin
