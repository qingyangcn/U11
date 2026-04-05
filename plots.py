from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


def _ensure_frame(frame_or_csv: Path | str | pd.DataFrame) -> pd.DataFrame:
    if isinstance(frame_or_csv, pd.DataFrame):
        return frame_or_csv.copy()
    return pd.read_csv(frame_or_csv)


def plot_metric_curve(frame_or_csv: Path | str | pd.DataFrame, x: str, y: str, output_path: Path, title: str) -> None:
    frame = _ensure_frame(frame_or_csv)
    if frame.empty or x not in frame.columns or y not in frame.columns:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(frame[x], frame[y], linewidth=2.0)
    ax.set_title(title)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_episode_metric_bundle(
    frame_or_csv: Path | str | pd.DataFrame,
    metrics: Sequence[str],
    output_dir: Path,
    prefix: str,
    x_col: str = "episode_index",
) -> None:
    frame = _ensure_frame(frame_or_csv)
    if frame.empty:
        return
    for metric in metrics:
        if metric not in frame.columns:
            continue
        plot_metric_curve(
            frame,
            x=x_col,
            y=metric,
            output_path=output_dir / f"{prefix}_{metric}.png",
            title=f"{prefix}: {metric}",
        )


def plot_grouped_bar(
    frame_or_csv: Path | str | pd.DataFrame,
    category_col: str,
    value_col: str,
    output_path: Path,
    title: str,
) -> None:
    frame = _ensure_frame(frame_or_csv)
    if frame.empty or category_col not in frame.columns or value_col not in frame.columns:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(frame[category_col].astype(str), frame[value_col])
    ax.set_title(title)
    ax.set_xlabel(category_col)
    ax.set_ylabel(value_col)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_grouped_metric_bundle(
    frame_or_csv: Path | str | pd.DataFrame,
    category_col: str,
    metrics: Sequence[str],
    output_dir: Path,
    prefix: str,
) -> None:
    frame = _ensure_frame(frame_or_csv)
    if frame.empty:
        return
    for metric in metrics:
        if metric not in frame.columns:
            continue
        plot_grouped_bar(
            frame,
            category_col=category_col,
            value_col=metric,
            output_path=output_dir / f"{prefix}_{metric}.png",
            title=f"{prefix}: {metric}",
        )
