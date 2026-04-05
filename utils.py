from __future__ import annotations

import json
import math
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def safe_div(numerator: float, denominator: float, default: float = 0.0) -> float:
    if abs(denominator) < 1e-12:
        return default
    return numerator / denominator


def minmax_normalize(value: float, low: float, high: float, default: float = 0.0) -> float:
    if high <= low:
        return default
    return clip((value - low) / (high - low), 0.0, 1.0)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def to_jsonable(payload: Any) -> Any:
    if is_dataclass(payload):
        payload = asdict(payload)
    if isinstance(payload, Path):
        return str(payload)
    if isinstance(payload, dict):
        return {str(key): to_jsonable(value) for key, value in payload.items()}
    if isinstance(payload, (list, tuple, set)):
        return [to_jsonable(value) for value in payload]
    if isinstance(payload, np.generic):
        return payload.item()
    return payload


def save_json(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(to_jsonable(payload), handle, ensure_ascii=False, indent=2)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sample_gamma_by_mean(rng: np.random.Generator, mean: float, shape: float) -> float:
    shape = max(shape, 1e-3)
    scale = mean / shape
    return float(rng.gamma(shape, scale))


class LocalProjector:
    """Project lon/lat onto a local planar x-y coordinate system in kilometers."""

    def __init__(self, lon0: float, lat0: float) -> None:
        self.lon0 = float(lon0)
        self.lat0 = float(lat0)
        self.cos_lat = math.cos(math.radians(self.lat0))
        self.km_per_deg_lat = 111.32
        self.km_per_deg_lon = 111.32 * self.cos_lat

    def project(self, lon: float, lat: float) -> tuple[float, float]:
        x = (float(lon) - self.lon0) * self.km_per_deg_lon
        y = (float(lat) - self.lat0) * self.km_per_deg_lat
        return x, y

    def unproject(self, x: float, y: float) -> tuple[float, float]:
        lon = self.lon0 + float(x) / self.km_per_deg_lon
        lat = self.lat0 + float(y) / self.km_per_deg_lat
        return lon, lat


def euclidean_distance_xy(a: tuple[float, float], b: tuple[float, float]) -> float:
    return float(math.hypot(a[0] - b[0], a[1] - b[1]))


def weighted_choice(rng: np.random.Generator, items: Iterable[Any], weights: Iterable[float]) -> Any:
    items_list = list(items)
    probs = np.asarray(list(weights), dtype=float)
    probs = np.clip(probs, 0.0, None)
    if probs.sum() <= 0:
        probs = np.ones(len(items_list), dtype=float) / max(len(items_list), 1)
    else:
        probs = probs / probs.sum()
    idx = int(rng.choice(len(items_list), p=probs))
    return items_list[idx]
