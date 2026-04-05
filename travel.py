from __future__ import annotations

from core.entities import Drone, Point, WeatherSnapshot
from core.enums import DroneMode
from core.utils import clip


def distance_km(a: Point, b: Point) -> float:
    dx = a.x - b.x
    dy = a.y - b.y
    return float((dx * dx + dy * dy) ** 0.5)


def load_ratio(payload_kg: float, capacity_kg: float) -> float:
    if capacity_kg <= 1e-9:
        return 0.0
    return clip(payload_kg / capacity_kg, 0.0, 1.0)


def interpolate_power_w(empty_power_w: float, full_power_w: float, payload_kg: float, capacity_kg: float) -> float:
    z = load_ratio(payload_kg, capacity_kg)
    return empty_power_w + (full_power_w - empty_power_w) * z


def effective_speed_km_min(
    drone: Drone,
    weather: WeatherSnapshot,
    payload_kg: float,
) -> float:
    z = load_ratio(payload_kg, drone.max_capacity_kg)
    load_factor = max(drone.min_speed_ratio, 1.0 - drone.speed_load_penalty * z)
    return drone.max_speed_km_min * weather.speed_factor * load_factor


def flight_time_min(
    origin: Point,
    destination: Point,
    drone: Drone,
    weather: WeatherSnapshot,
    payload_kg: float,
) -> float:
    speed = effective_speed_km_min(
        drone=drone,
        weather=weather,
        payload_kg=payload_kg,
    )
    speed = max(speed, 1e-3)
    return distance_km(origin, destination) / speed


def flight_energy(
    origin: Point,
    destination: Point,
    drone: Drone,
    weather: WeatherSnapshot,
    payload_kg: float,
) -> float:
    duration_min = flight_time_min(
        origin=origin,
        destination=destination,
        drone=drone,
        weather=weather,
        payload_kg=payload_kg,
    )
    cruise_power_w = interpolate_power_w(
        empty_power_w=drone.cruise_power_empty_w,
        full_power_w=drone.cruise_power_full_w,
        payload_kg=payload_kg,
        capacity_kg=drone.max_capacity_kg,
    )
    return cruise_power_w * (max(duration_min, 0.0) / 60.0) * weather.energy_factor


def hover_energy(
    drone: Drone,
    weather: WeatherSnapshot,
    payload_kg: float,
    duration_min: float,
    mode: DroneMode,
) -> float:
    mode_factor = {
        DroneMode.IDLE: 1.00,
        DroneMode.PICKUP_SERVICE: 1.05,
        DroneMode.DROPOFF_SERVICE: 1.02,
    }.get(mode, 1.0)
    hover_power_w = interpolate_power_w(
        empty_power_w=drone.hover_power_empty_w,
        full_power_w=drone.hover_power_full_w,
        payload_kg=payload_kg,
        capacity_kg=drone.max_capacity_kg,
    )
    return hover_power_w * weather.hover_factor * mode_factor * (max(duration_min, 0.0) / 60.0)
