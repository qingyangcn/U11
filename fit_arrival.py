from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pandas as pd

from config.default import AppConfig, build_default_config
from core.utils import ensure_dir, save_json


def _load_orders(cfg: AppConfig) -> pd.DataFrame:
    path = cfg.paths.raw_dir / "food_delivery_data.xlsx"
    df = pd.read_excel(path)
    if "Order Date" not in df.columns or "Order Time" not in df.columns:
        raise ValueError("food_delivery_data.xlsx is missing Order Date or Order Time columns.")
    df["Order Date"] = pd.to_datetime(df["Order Date"])
    df["Order Time"] = pd.to_datetime(
        df["Order Time"].astype(str).str.strip(),
        format="%H:%M:%S",
        errors="coerce",
    ).dt.time
    df["order_dt"] = pd.to_datetime(
        df["Order Date"].dt.strftime("%Y-%m-%d") + " " + df["Order Time"].astype(str),
        errors="coerce",
    )
    df = df.dropna(subset=["order_dt"]).copy()
    return df


def fit_arrival_model(cfg: AppConfig | None = None) -> Dict[str, Any]:
    cfg = cfg or build_default_config()
    df = _load_orders(cfg)
    df = df[df["Order Type"].fillna("").astype(str).str.lower().str.strip() == "delivery"].copy()
    df["hour"] = df["order_dt"].dt.hour
    min_hour = int(df["hour"].min())
    max_hour = int(df["hour"].max())
    hours = list(range(min_hour, max_hour + 1))
    hourly_counts = df.groupby("hour").size().reindex(hours, fill_value=0).astype(float)
    rates_per_min = (hourly_counts / 60.0).tolist()

    result = {
        "hours": hours,
        "rates_per_min": rates_per_min,
        "business_start_hour": cfg.order.business_start_hour,
        "business_end_hour": cfg.order.business_end_hour,
        "delivery_order_days": 1,
        "raw_delivery_orders": int(len(df)),
        "assumption": "single_day_dataset",
        "observed_min_hour": min_hour,
        "observed_max_hour": max_hour,
    }
    return result


def main() -> None:
    cfg = build_default_config()
    ensure_dir(cfg.paths.fitted_dir)
    result = fit_arrival_model(cfg)
    save_json(cfg.paths.fitted_dir / "arrival_model.json", result)
    print("saved", cfg.paths.fitted_dir / "arrival_model.json")


if __name__ == "__main__":
    main()
