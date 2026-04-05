from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


@dataclass(slots=True)
class PathsConfig:
    root: Path
    raw_dir: Path
    processed_dir: Path
    fitted_dir: Path
    output_dir: Path
    checkpoint_dir: Path
    log_dir: Path
    eval_dir: Path
    figure_dir: Path


@dataclass(slots=True)
class GeoConfig:
    customer_distance_cap_km: float = 5.0
    max_redraws: int = 64
    station_count: int = 2


@dataclass(slots=True)
class OrderConfig:
    business_start_hour: int = 6
    business_end_hour: int = 21
    food_weight_mean: float = 0.12
    dessert_weight_mean: float = 0.08
    drink_weight_mean: float = 0.18
    min_quantity_kg: float = 0.15
    quantity_noise_std: float = 0.08
    prep_gamma_shape: float = 3.0
    prep_base_min: float = 8.0
    prep_complexity_weight: float = 2.0
    prep_peak_bonus_min: float = 5.0
    sla_base_min: float = 15.0
    sla_scale: float = 1.5
    pickup_shape_weight: float = 1.5
    dessert_shape_weight: float = 0.7
    drink_shape_weight: float = 0.5
    peak_hours: Tuple[int, ...] = (11, 12, 17, 18, 19)
    cancel_lambda_max: float = 0.025
    cancel_wait_norm_min: float = 60.0
    cancel_eta_norm_min: float = 45.0
    cancel_late_norm_min: float = 30.0
    cancel_beta_wait: float = 1.25
    cancel_beta_eta: float = 0.65
    cancel_beta_late: float = 1.10
    cancel_beta_unassigned: float = 0.45
    cancel_beta_release: float = 0.35


@dataclass(slots=True)
class DroneConfig:
    count: int = 10
    max_speed_kmph: float = 72.0
    max_capacity_kg: float = 2.5
    pickup_service_gamma_shape: float = 3.0
    pickup_service_mean_min: float = 3.0
    dropoff_service_gamma_shape: float = 3.5
    dropoff_service_mean_min: float = 2.5
    swap_time_min: float = 8.0
    battery_capacity_wh: float = 800.0
    safety_reserve_ratio: float = 0.15
    cruise_power_empty_w: float = 500.0
    cruise_power_full_w: float = 900.0
    hover_power_empty_w: float = 700.0
    hover_power_full_w: float = 1100.0
    min_speed_ratio: float = 0.60
    speed_load_penalty: float = 0.35

    @property
    def battery_max(self) -> float:
        return self.battery_capacity_wh

    @property
    def safety_energy(self) -> float:
        return self.battery_capacity_wh * self.safety_reserve_ratio


@dataclass(slots=True)
class DispatchConfig:
    interval_min: float = 10.0
    archive_limit: int = 32
    particles: int = 24
    iterations: int = 18
    inertia: float = 0.55
    c1: float = 1.45
    c2: float = 1.45
    weight_efficiency: float = 0.30
    weight_risk: float = 0.50
    weight_balance: float = 0.20


@dataclass(slots=True)
class WeatherConfig:
    change_interval_min: float = 60.0
    speed_factor_floor: float = 0.55
    energy_factor_cap: float = 2.0
    hover_factor_cap: float = 1.8


@dataclass(slots=True)
class FitConfig:
    cancel_train_min_samples: int = 32
    random_seed: int = 42


@dataclass(slots=True)
class RLConfig:
    candidate_rules: Tuple[str, ...] = (
        "nearest",
        "earliest_deadline",
        "minimum_slack",
        "deliver_first",
        "pickup_first",
    )
    observation_dim: int = 38
    time_ref_min: float = 10.0
    onboard_norm_ref: float = 6.0
    assigned_norm_ref: float = 8.0
    quantity_ref_kg: float = 5.0
    complete_norm_cap: float = 4.0
    lateness_norm_cap: float = 4.0
    include_workload_feature: bool = False


@dataclass(slots=True)
class RewardConfig:
    delivered: float = 1.0
    canceled: float = 1.1
    time_cost: float = 0.1
    energy_cost: float = 0.05
    lateness_cost: float = 0.8


@dataclass(slots=True)
class TrainingConfig:
    total_timesteps: int = 2500000
    n_steps: int = 512
    batch_size: int = 128
    learning_rate: float = 3e-4
    gamma: float = 0.995
    gae_lambda: float = 0.95
    ent_coef: float = 0.01
    vf_coef: float = 0.50
    seed: int = 42
    eval_episodes: int = 10


@dataclass(slots=True)
class EnvConfig:
    load_scale: float = 1.0
    simulation_horizon_min: float = 1440.0


@dataclass(slots=True)
class AppConfig:
    paths: PathsConfig
    geo: GeoConfig
    order: OrderConfig
    drone: DroneConfig
    dispatch: DispatchConfig
    weather: WeatherConfig
    fit: FitConfig
    rl: RLConfig
    reward: RewardConfig
    training: TrainingConfig
    env: EnvConfig


def build_default_config(root: Path | None = None) -> AppConfig:
    root_dir = Path(root) if root is not None else _project_root()
    paths = PathsConfig(
        root=root_dir,
        raw_dir=root_dir / "data" / "raw",
        processed_dir=root_dir / "data" / "processed",
        fitted_dir=root_dir / "data" / "fitted",
        output_dir=root_dir / "outputs",
        checkpoint_dir=root_dir / "outputs" / "checkpoints",
        log_dir=root_dir / "outputs" / "logs",
        eval_dir=root_dir / "outputs" / "eval",
        figure_dir=root_dir / "outputs" / "figures",
    )
    return AppConfig(
        paths=paths,
        geo=GeoConfig(),
        order=OrderConfig(),
        drone=DroneConfig(),
        dispatch=DispatchConfig(),
        weather=WeatherConfig(),
        fit=FitConfig(),
        rl=RLConfig(),
        reward=RewardConfig(),
        training=TrainingConfig(),
        env=EnvConfig(),
    )
