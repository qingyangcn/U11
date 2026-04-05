from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd

from core.utils import to_jsonable


def build_run_name(prefix: str, scenario: str, seed: int | None = None) -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    seed_part = f"_seed{seed}" if seed is not None else ""
    return f"{prefix}_{scenario}{seed_part}_{stamp}"


def build_eval_seeds(base_seed: int, count: int) -> List[int]:
    return [int(base_seed + offset) for offset in range(max(int(count), 0))]


def episodes_to_frame(episodes: Iterable[Dict]) -> pd.DataFrame:
    frame = pd.DataFrame(list(episodes))
    if frame.empty:
        return frame
    if "episode_index" not in frame.columns:
        frame.insert(0, "episode_index", range(len(frame)))

    def numeric_column(column: str, default: float = 0.0) -> pd.Series:
        if column not in frame.columns:
            return pd.Series(default, index=frame.index, dtype=float)
        return pd.to_numeric(frame[column], errors="coerce").astype(float)

    total_orders = numeric_column("total_orders")
    delivered = numeric_column("delivered_orders")
    canceled = numeric_column("canceled_orders")
    lateness = numeric_column("total_lateness_min")
    energy = numeric_column("total_energy")
    safe_total_orders = total_orders.where(total_orders != 0.0, float("nan"))
    safe_delivered = delivered.where(delivered != 0.0, float("nan"))

    if "total_orders" in frame.columns:
        frame["completion_rate"] = (delivered / safe_total_orders).fillna(0.0)
        frame["cancel_rate"] = (canceled / safe_total_orders).fillna(0.0)
        frame["avg_energy_per_order"] = (energy / safe_total_orders).fillna(0.0)
    if "delivered_orders" in frame.columns:
        frame["avg_lateness_per_delivered"] = (lateness / safe_delivered).fillna(0.0)
    return frame


def summarize_episode_stats(episodes: Iterable[Dict] | pd.DataFrame) -> Dict[str, float]:
    frame = episodes.copy() if isinstance(episodes, pd.DataFrame) else episodes_to_frame(episodes)
    if frame.empty:
        return {}
    summary = {}
    for column in frame.columns:
        if pd.api.types.is_numeric_dtype(frame[column]):
            summary[f"{column}_mean"] = float(frame[column].mean())
            summary[f"{column}_std"] = float(frame[column].std(ddof=0))
    return summary


def summarize_by_group(frame: pd.DataFrame, group_col: str) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    rows = []
    for group_value, group in frame.groupby(group_col):
        row = {group_col: group_value}
        row.update(summarize_episode_stats(group))
        rows.append(row)
    return pd.DataFrame(rows)


def save_episode_table(path: Path, episodes: List[Dict] | pd.DataFrame) -> pd.DataFrame:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = episodes.copy() if isinstance(episodes, pd.DataFrame) else episodes_to_frame(episodes)
    frame.to_csv(path, index=False)
    return frame


def save_summary_json(path: Path, summary: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(to_jsonable(summary), handle, ensure_ascii=False, indent=2)


def save_summary_table(path: Path, summary: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([summary]).to_csv(path, index=False)
