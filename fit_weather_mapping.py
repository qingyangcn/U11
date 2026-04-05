from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd

from config.default import AppConfig, build_default_config
from core.utils import ensure_dir, save_json


def _classify_weather_state(row: pd.Series) -> str:
    summary = str(row.get("Summary", "")).lower()
    precip = str(row.get("Precip Type", "")).lower()
    wind = float(row.get("Wind Speed (km/h)", 0.0))
    if "storm" in summary or wind >= 30:
        return "STORM"
    if "rain" in summary or precip == "rain":
        return "RAIN"
    if wind >= 20:
        return "WINDY"
    if wind <= 8:
        return "CALM"
    return "NORMAL"


def fit_weather_mapping(cfg: AppConfig | None = None) -> Dict[str, Any]:
    cfg = cfg or build_default_config()
    df = pd.read_csv(cfg.paths.raw_dir / "weather_dataset.csv")
    parsed = pd.to_datetime(df["Formatted Date"], errors="coerce", utc=True)
    df["hour"] = parsed.dt.hour.fillna(0).astype(int)
    df["state"] = df.apply(_classify_weather_state, axis=1)

    wind_q95 = float(df["Wind Speed (km/h)"].quantile(0.95))
    wind_q95 = max(wind_q95, 1.0)
    records = []
    for _, row in df.iterrows():
        wind = float(row["Wind Speed (km/h)"])
        humidity = float(row.get("Humidity", 0.0))
        wind_ratio = min(wind / wind_q95, 1.5)
        speed_factor = max(cfg.weather.speed_factor_floor, 1.0 - 0.35 * wind_ratio)
        energy_factor = min(cfg.weather.energy_factor_cap, 1.0 + 0.45 * wind_ratio + 0.10 * humidity)
        hover_factor = min(cfg.weather.hover_factor_cap, 1.0 + 0.30 * wind_ratio + 0.08 * humidity)
        records.append(
            {
                "hour": int(row["hour"]),
                "state": str(row["state"]),
                "summary": str(row.get("Summary", "")),
                "wind_speed_kmph": wind,
                "humidity": humidity,
                "speed_factor": float(speed_factor),
                "energy_factor": float(energy_factor),
                "hover_factor": float(hover_factor),
            }
        )

    result = {
        "change_interval_min": cfg.weather.change_interval_min,
        "records": records,
    }
    return result


def main() -> None:
    cfg = build_default_config()
    ensure_dir(cfg.paths.fitted_dir)
    result = fit_weather_mapping(cfg)
    save_json(cfg.paths.fitted_dir / "weather_mapping.json", result)
    print("saved", cfg.paths.fitted_dir / "weather_mapping.json")


if __name__ == "__main__":
    main()
