from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd

from config.default import AppConfig, build_default_config
from core.utils import LocalProjector, ensure_dir, euclidean_distance_xy, save_json


def _greedy_weighted_k_medoids(
    points: np.ndarray,
    weights: np.ndarray,
    k: int,
) -> list[int]:
    indices = list(range(len(points)))
    chosen: list[int] = []
    for _ in range(min(k, len(points))):
        best_idx = None
        best_cost = float("inf")
        for idx in indices:
            candidate = chosen + [idx]
            cost = 0.0
            for p_i, point in enumerate(points):
                best_d = min(
                    euclidean_distance_xy(tuple(point), tuple(points[c_idx])) for c_idx in candidate
                )
                cost += weights[p_i] * best_d
            if cost < best_cost:
                best_cost = cost
                best_idx = idx
        if best_idx is None:
            break
        chosen.append(best_idx)
        indices.remove(best_idx)
    return chosen


def _sample_candidate_points(
    points: np.ndarray,
    weights: np.ndarray,
    max_candidates: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if len(points) <= max_candidates:
        return np.arange(len(points))
    probs = np.clip(weights.astype(float), 0.0, None)
    if probs.sum() <= 0:
        probs = np.ones(len(points), dtype=float) / len(points)
    else:
        probs = probs / probs.sum()
    sampled = rng.choice(len(points), size=max_candidates, replace=False, p=probs)
    return np.asarray(sampled, dtype=int)


def fit_spatial_model(cfg: AppConfig | None = None) -> Dict[str, Any]:
    cfg = cfg or build_default_config()
    rng = np.random.default_rng(cfg.fit.random_seed)
    merchants = pd.read_csv(cfg.paths.raw_dir / "food_business_zhanggong_locations.csv")
    users = pd.read_csv(cfg.paths.raw_dir / "user_zhanggong_locations.csv")

    lon0 = float(pd.concat([merchants["longitude"], users["longitude"]]).mean())
    lat0 = float(pd.concat([merchants["latitude"], users["latitude"]]).mean())
    projector = LocalProjector(lon0=lon0, lat0=lat0)

    merchant_points: list[dict[str, Any]] = []
    for _, row in merchants.iterrows():
        x, y = projector.project(row["longitude"], row["latitude"])
        rating = pd.to_numeric(pd.Series([row.get("rating")]), errors="coerce").iloc[0]
        merchant_points.append(
            {
                "merchant_id": str(row["id"]),
                "name": str(row["name"]),
                "x": x,
                "y": y,
                "lon": float(row["longitude"]),
                "lat": float(row["latitude"]),
                "weight": float(rating) if not pd.isna(rating) else 1.0,
            }
        )

    user_xy = np.array([projector.project(lon, lat) for lon, lat in zip(users["longitude"], users["latitude"])])
    merchant_xy = np.array([(item["x"], item["y"]) for item in merchant_points])

    # Approximate "relative distance" by each user's nearest merchant distance.
    nearest_distances: list[float] = []
    for ux, uy in user_xy:
        d = np.sqrt(((merchant_xy - np.array([ux, uy])) ** 2).sum(axis=1))
        nearest_distances.append(float(np.min(d)))

    raw_distance_samples = np.asarray(nearest_distances, dtype=float)
    target_cap = float(cfg.geo.customer_distance_cap_km)
    original_max_distance = float(raw_distance_samples.max()) if len(raw_distance_samples) else 0.0
    distance_scale = target_cap / max(original_max_distance, 1e-6) if target_cap > 0 else 1.0
    scaled_distance_samples = np.clip(raw_distance_samples * distance_scale, 0.0, target_cap)

    bandwidth = float(np.median(raw_distance_samples) / 3.0) if nearest_distances else 0.3
    bandwidth = max(bandwidth, 0.08)
    max_radius = target_cap

    demand_points = np.vstack([merchant_xy, user_xy])
    demand_weights = np.concatenate(
        [
            np.array([item["weight"] for item in merchant_points], dtype=float),
            np.ones(len(user_xy), dtype=float),
        ]
    )
    candidate_indices = _sample_candidate_points(
        demand_points,
        demand_weights,
        max_candidates=min(256, len(demand_points)),
        rng=rng,
    )
    candidate_points = demand_points[candidate_indices]
    candidate_weights = demand_weights[candidate_indices]
    chosen_local_indices = _greedy_weighted_k_medoids(candidate_points, candidate_weights, cfg.geo.station_count)
    medoid_indices = [int(candidate_indices[idx]) for idx in chosen_local_indices]
    station_points = []
    for idx, demand_idx in enumerate(medoid_indices):
        x, y = map(float, demand_points[demand_idx])
        lon, lat = projector.unproject(x, y)
        station_points.append(
            {
                "station_id": f"S{idx}",
                "x": x,
                "y": y,
                "lon": lon,
                "lat": lat,
            }
        )

    result = {
        "projector": {"lon0": lon0, "lat0": lat0},
        "merchant_points": merchant_points,
        "bandwidth_km": bandwidth,
        "relative_distance_km_samples": scaled_distance_samples.tolist(),
        "max_service_radius_km": max_radius,
        "distance_scale_factor": distance_scale,
        "distance_scaling_mode": "stretch_to_cap",
        "original_distance_max_km": original_max_distance,
        "scaled_distance_max_km": float(scaled_distance_samples.max()) if len(scaled_distance_samples) else 0.0,
        "x_bounds": [
            float(min(item["x"] for item in merchant_points) - max_radius),
            float(max(item["x"] for item in merchant_points) + max_radius),
        ],
        "y_bounds": [
            float(min(item["y"] for item in merchant_points) - max_radius),
            float(max(item["y"] for item in merchant_points) + max_radius),
        ],
        "station_points": station_points,
    }
    return result


def main() -> None:
    cfg = build_default_config()
    ensure_dir(cfg.paths.fitted_dir)
    result = fit_spatial_model(cfg)
    save_json(cfg.paths.fitted_dir / "spatial_model.json", result)
    print("saved", cfg.paths.fitted_dir / "spatial_model.json")


if __name__ == "__main__":
    main()
