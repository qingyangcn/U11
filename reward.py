from __future__ import annotations

from config.default import AppConfig
from core.entities import StepMetrics


def compute_step_reward(metrics: StepMetrics, cfg: AppConfig) -> float:
    reward = 0.0
    reward += cfg.reward.delivered * metrics.delivered_count
    reward -= cfg.reward.canceled * metrics.canceled_count
    reward -= cfg.reward.time_cost * (metrics.delta_time / max(cfg.rl.time_ref_min, 1e-6))
    reward -= cfg.reward.energy_cost * (metrics.delta_energy / max(cfg.drone.battery_max, 1e-6))
    reward -= cfg.reward.lateness_cost * (metrics.lateness_total / max(cfg.rl.time_ref_min, 1e-6))
    return float(reward)
