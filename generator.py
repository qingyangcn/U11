from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import sklearn

from config.default import AppConfig, build_default_config
from core.entities import Drone, Order, Point, Station
from core.enums import OrderStatus
from core.utils import LocalProjector, clip, ensure_dir, load_json, minmax_normalize, sample_gamma_by_mean, weighted_choice
from fit.fit_arrival import fit_arrival_model
from fit.fit_cancel_model import FEATURE_COLUMNS, fit_cancel_model
from fit.fit_spatial_model import fit_spatial_model
from fit.fit_weather_mapping import fit_weather_mapping


@dataclass(slots=True)
class FittedArtifacts:
    arrival_model: Dict[str, Any]
    spatial_model: Dict[str, Any]
    weather_mapping: Dict[str, Any]
    cancel_model: Any
    cancel_meta: Dict[str, Any]
    order_frame: pd.DataFrame


def ensure_fitted_artifacts(cfg: AppConfig | None = None) -> None:
    cfg = cfg or build_default_config()
    ensure_dir(cfg.paths.fitted_dir)
    arrival_path = cfg.paths.fitted_dir / "arrival_model.json"
    spatial_path = cfg.paths.fitted_dir / "spatial_model.json"
    weather_path = cfg.paths.fitted_dir / "weather_mapping.json"
    cancel_model_path = cfg.paths.fitted_dir / "cancel_model.pkl"
    cancel_meta_path = cfg.paths.fitted_dir / "cancel_model_meta.json"

    if not arrival_path.exists():
        from core.utils import save_json

        save_json(arrival_path, fit_arrival_model(cfg))
    else:
        arrival_meta = load_json(arrival_path)
        if arrival_meta.get("assumption") != "single_day_dataset":
            from core.utils import save_json

            save_json(arrival_path, fit_arrival_model(cfg))
    if not spatial_path.exists():
        from core.utils import save_json

        save_json(spatial_path, fit_spatial_model(cfg))
    else:
        spatial_meta = load_json(spatial_path)
        if (
            spatial_meta.get("distance_scaling_mode") != "stretch_to_cap"
            or abs(float(spatial_meta.get("max_service_radius_km", -1.0)) - float(cfg.geo.customer_distance_cap_km)) > 1e-9
        ):
            from core.utils import save_json

            save_json(spatial_path, fit_spatial_model(cfg))
    if not weather_path.exists():
        from core.utils import save_json

        save_json(weather_path, fit_weather_mapping(cfg))
    if not (cancel_model_path.exists() and cancel_meta_path.exists()):
        needs_refit = True
    else:
        meta = load_json(cancel_meta_path)
        needs_refit = meta.get("sklearn_version") != sklearn.__version__
    if needs_refit:
        from core.utils import save_json

        result = fit_cancel_model(cfg)
        with cancel_model_path.open("wb") as handle:
            pickle.dump(result["pipeline"], handle)
        save_json(cancel_meta_path, result["stats"])


def load_fitted_artifacts(cfg: AppConfig | None = None) -> FittedArtifacts:
    cfg = cfg or build_default_config()
    ensure_fitted_artifacts(cfg)
    with (cfg.paths.fitted_dir / "cancel_model.pkl").open("rb") as handle:
        cancel_model = pickle.load(handle)
    order_frame = pd.read_excel(cfg.paths.raw_dir / "food_delivery_data.xlsx")
    order_frame = order_frame[order_frame["Order Type"].fillna("").astype(str).str.lower() == "delivery"].copy()
    return FittedArtifacts(
        arrival_model=load_json(cfg.paths.fitted_dir / "arrival_model.json"),
        spatial_model=load_json(cfg.paths.fitted_dir / "spatial_model.json"),
        weather_mapping=load_json(cfg.paths.fitted_dir / "weather_mapping.json"),
        cancel_model=cancel_model,
        cancel_meta=load_json(cfg.paths.fitted_dir / "cancel_model_meta.json"),
        order_frame=order_frame.reset_index(drop=True),
    )


class ScenarioGenerator:
    def __init__(self, cfg: AppConfig | None = None, seed: int | None = None) -> None:
        self.cfg = cfg or build_default_config()
        self.rng = np.random.default_rng(seed if seed is not None else self.cfg.training.seed)
        self.artifacts = load_fitted_artifacts(self.cfg)
        projector_cfg = self.artifacts.spatial_model["projector"]
        self.projector = LocalProjector(projector_cfg["lon0"], projector_cfg["lat0"])
        self.stations = self._build_stations()

    def reseed(self, seed: int | None = None) -> None:
        self.rng = np.random.default_rng(seed if seed is not None else self.cfg.training.seed)

    def _build_stations(self) -> list[Station]:
        stations: list[Station] = []
        for item in self.artifacts.spatial_model["station_points"]:
            stations.append(
                Station(
                    station_id=str(item["station_id"]),
                    location=Point(
                        x=float(item["x"]),
                        y=float(item["y"]),
                        lon=float(item["lon"]),
                        lat=float(item["lat"]),
                    ),
                )
            )
        return stations

    def sample_arrival_times(self) -> List[float]:
        hours = self.artifacts.arrival_model["hours"]
        rates = self.artifacts.arrival_model["rates_per_min"]
        business_start = self.cfg.order.business_start_hour
        arrivals: list[float] = []
        for hour, rate in zip(hours, rates):
            expected = max(rate * 60.0 * self.cfg.env.load_scale, 0.0)
            count = int(self.rng.poisson(expected))
            offset = (hour - business_start) * 60.0
            for _ in range(count):
                arrivals.append(float(offset + self.rng.uniform(0.0, 60.0)))
        arrivals.sort()
        return arrivals

    def _sample_order_row(self) -> pd.Series:
        idx = int(self.rng.integers(0, len(self.artifacts.order_frame)))
        return self.artifacts.order_frame.iloc[idx]

    def _sample_merchant_point(self) -> Point:
        merchants = self.artifacts.spatial_model["merchant_points"]
        merchant = weighted_choice(self.rng, merchants, [float(item["weight"]) for item in merchants])
        bandwidth = float(self.artifacts.spatial_model["bandwidth_km"])
        x = float(merchant["x"]) + float(self.rng.normal(0.0, bandwidth))
        y = float(merchant["y"]) + float(self.rng.normal(0.0, bandwidth))
        x_low, x_high = self.artifacts.spatial_model["x_bounds"]
        y_low, y_high = self.artifacts.spatial_model["y_bounds"]
        x = clip(x, x_low, x_high)
        y = clip(y, y_low, y_high)
        lon, lat = self.projector.unproject(x, y)
        return Point(x=x, y=y, lon=lon, lat=lat)

    def _sample_customer_point(self, merchant: Point) -> Point:
        distances = self.artifacts.spatial_model["relative_distance_km_samples"]
        max_radius = float(self.artifacts.spatial_model["max_service_radius_km"])
        x_low, x_high = self.artifacts.spatial_model["x_bounds"]
        y_low, y_high = self.artifacts.spatial_model["y_bounds"]
        x_pad = max_radius
        y_pad = max_radius
        for _ in range(self.cfg.geo.max_redraws):
            radius = float(distances[int(self.rng.integers(0, len(distances)))])
            theta = float(self.rng.uniform(0.0, 2.0 * np.pi))
            x = merchant.x + radius * np.cos(theta)
            y = merchant.y + radius * np.sin(theta)
            if x_low - x_pad <= x <= x_high + x_pad and y_low - y_pad <= y <= y_high + y_pad:
                lon, lat = self.projector.unproject(x, y)
                return Point(x=float(x), y=float(y), lon=lon, lat=lat)
        radius = min(max_radius, 0.5 * max_radius)
        theta = float(self.rng.uniform(0.0, 2.0 * np.pi))
        x = merchant.x + radius * np.cos(theta)
        y = merchant.y + radius * np.sin(theta)
        lon, lat = self.projector.unproject(x, y)
        return Point(x=float(x), y=float(y), lon=lon, lat=lat)

    def _sample_quantity_kg(self, row: pd.Series) -> float:
        nf = int(row.get("Food Quantity", 1) or 1)
        nd = int(row.get("Dessert Quantity", 0) or 0)
        nr = int(row.get("Drink Quantity", 0) or 0)
        mean = (
            self.cfg.order.food_weight_mean * nf
            + self.cfg.order.dessert_weight_mean * nd
            + self.cfg.order.drink_weight_mean * nr
        )
        return max(self.cfg.order.min_quantity_kg, float(mean + self.rng.normal(0.0, self.cfg.order.quantity_noise_std)))

    def _predict_cancel_prob(self, row: pd.Series) -> float:
        frame = pd.DataFrame([{column: row.get(column, np.nan) for column in FEATURE_COLUMNS}])
        prob = float(self.artifacts.cancel_model.predict_proba(frame)[0, 1])
        return clip(prob, 0.02, 0.95)

    def _prep_time(self, created_time: float, row: pd.Series) -> float:
        nf = int(row.get("Food Quantity", 1) or 1)
        nd = int(row.get("Dessert Quantity", 0) or 0)
        nr = int(row.get("Drink Quantity", 0) or 0)
        score = (
            self.cfg.order.pickup_shape_weight * nf
            + self.cfg.order.dessert_shape_weight * nd
            + self.cfg.order.drink_shape_weight * nr
        )
        absolute_hour = self.cfg.order.business_start_hour + int(created_time // 60.0)
        peak_bonus = self.cfg.order.prep_peak_bonus_min if absolute_hour in self.cfg.order.peak_hours else 0.0
        mean = self.cfg.order.prep_base_min + self.cfg.order.prep_complexity_weight * score + peak_bonus
        return sample_gamma_by_mean(self.rng, mean, self.cfg.order.prep_gamma_shape)

    def _deadline(self, ready_time: float, merchant: Point, customer: Point) -> float:
        ref_station = min(self.stations, key=lambda station: (merchant.x - station.location.x) ** 2 + (merchant.y - station.location.y) ** 2)
        access_dist = ((merchant.x - ref_station.location.x) ** 2 + (merchant.y - ref_station.location.y) ** 2) ** 0.5
        linehaul_dist = ((merchant.x - customer.x) ** 2 + (merchant.y - customer.y) ** 2) ** 0.5
        v_ref = self.cfg.drone.max_speed_kmph / 60.0
        nominal = (
            access_dist / max(v_ref, 1e-3)
            + linehaul_dist / max(v_ref, 1e-3)
            + self.cfg.drone.pickup_service_mean_min
            + self.cfg.drone.dropoff_service_mean_min
        )
        log_sla = self.cfg.order.sla_base_min + self.cfg.order.sla_scale * nominal
        return ready_time + log_sla

    def build_orders(self) -> Dict[str, Order]:
        arrivals = self.sample_arrival_times()
        orders: Dict[str, Order] = {}
        for idx, created_time in enumerate(arrivals):
            row = self._sample_order_row()
            merchant = self._sample_merchant_point()
            customer = self._sample_customer_point(merchant)
            quantity = self._sample_quantity_kg(row)
            ready_time = created_time + self._prep_time(created_time, row)
            deadline = self._deadline(ready_time, merchant, customer)
            order = Order(
                order_id=f"O{idx:05d}",
                created_time=created_time,
                ready_time=ready_time,
                deadline=deadline,
                merchant_loc=merchant,
                customer_loc=customer,
                quantity_kg=quantity,
                food_quantity=int(row.get("Food Quantity", 1) or 1),
                dessert_quantity=int(row.get("Dessert Quantity", 0) or 0),
                drink_quantity=int(row.get("Drink Quantity", 0) or 0),
                base_cancel_prob=self._predict_cancel_prob(row),
                status=OrderStatus.NOT_READY,
                metadata={
                    "order_method": row.get("Order Method"),
                    "customer_loyalty": row.get("Customer Loyalty"),
                    "order_status_raw": row.get("Order Status"),
                    "announced": False,
                },
            )
            orders[order.order_id] = order
        return orders

    def build_drones(self) -> Dict[str, Drone]:
        drones: Dict[str, Drone] = {}
        for idx in range(self.cfg.drone.count):
            station = self.stations[idx % len(self.stations)]
            drone_id = f"D{idx:02d}"
            drones[drone_id] = Drone(
                drone_id=drone_id,
                home_station_id=station.station_id,
                home_station_loc=station.location,
                position=station.location,
                max_speed_km_min=self.cfg.drone.max_speed_kmph / 60.0,
                max_capacity_kg=self.cfg.drone.max_capacity_kg,
                pickup_service_mean_min=self.cfg.drone.pickup_service_mean_min,
                dropoff_service_mean_min=self.cfg.drone.dropoff_service_mean_min,
                pickup_service_shape=self.cfg.drone.pickup_service_gamma_shape,
                dropoff_service_shape=self.cfg.drone.dropoff_service_gamma_shape,
                swap_time_min=self.cfg.drone.swap_time_min,
                battery_max=self.cfg.drone.battery_max,
                battery_current=self.cfg.drone.battery_max,
                battery_safety_margin=self.cfg.drone.safety_energy,
                cruise_power_empty_w=self.cfg.drone.cruise_power_empty_w,
                cruise_power_full_w=self.cfg.drone.cruise_power_full_w,
                hover_power_empty_w=self.cfg.drone.hover_power_empty_w,
                hover_power_full_w=self.cfg.drone.hover_power_full_w,
                min_speed_ratio=self.cfg.drone.min_speed_ratio,
                speed_load_penalty=self.cfg.drone.speed_load_penalty,
                next_available_time=0.0,
            )
        return drones
