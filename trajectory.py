from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from core.enums import TaskType
from env.battery import nearest_station


@dataclass(slots=True)
class FlightSegment:
    drone_id: str
    segment_type: str
    order_id: str | None
    start_time: float
    end_time: float
    x0: float
    y0: float
    x1: float
    y1: float

    def as_dict(self) -> Dict[str, float | str | None]:
        return {
            "drone_id": self.drone_id,
            "segment_type": self.segment_type,
            "order_id": self.order_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "x0": self.x0,
            "y0": self.y0,
            "x1": self.x1,
            "y1": self.y1,
        }


def attach_segment_recorder(env) -> List[FlightSegment]:
    simulator = env.simulator
    segments: List[FlightSegment] = []

    original_launch_task = simulator._launch_task
    original_send_to_charge = simulator._send_to_charge

    def launch_wrapper(drone, projection) -> None:
        order = simulator.orders[projection.order_id]
        target = order.merchant_loc if projection.task_type == TaskType.PICKUP else order.customer_loc
        start_time = simulator.current_time
        x0 = float(drone.position.x)
        y0 = float(drone.position.y)
        x1 = float(target.x)
        y1 = float(target.y)
        original_launch_task(drone, projection)
        segments.append(
            FlightSegment(
                drone_id=drone.drone_id,
                segment_type=projection.task_type.value.lower(),
                order_id=projection.order_id,
                start_time=start_time,
                end_time=float(drone.next_available_time),
                x0=x0,
                y0=y0,
                x1=x1,
                y1=y1,
            )
        )

    def charge_wrapper(drone) -> None:
        station = nearest_station(drone.position, simulator.stations)
        start_time = simulator.current_time
        x0 = float(drone.position.x)
        y0 = float(drone.position.y)
        x1 = float(station.location.x)
        y1 = float(station.location.y)
        original_send_to_charge(drone)
        segments.append(
            FlightSegment(
                drone_id=drone.drone_id,
                segment_type="charge",
                order_id=None,
                start_time=start_time,
                end_time=float(drone.next_available_time),
                x0=x0,
                y0=y0,
                x1=x1,
                y1=y1,
            )
        )

    simulator._launch_task = launch_wrapper
    simulator._send_to_charge = charge_wrapper
    return segments


def segments_to_frame(segments: Sequence[FlightSegment]) -> pd.DataFrame:
    return pd.DataFrame([segment.as_dict() for segment in segments])


def _drone_color_map(drone_ids: Iterable[str]) -> Dict[str, tuple]:
    ordered = sorted(set(drone_ids))
    cmap = plt.get_cmap("tab10")
    return {drone_id: cmap(idx % 10) for idx, drone_id in enumerate(ordered)}


def plot_trajectories(
    output_path: Path,
    segments: Sequence[FlightSegment],
    stations,
    orders,
    x_bounds: tuple[float, float],
    y_bounds: tuple[float, float],
    title: str,
    max_time: float | None = None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    filtered = [segment for segment in segments if max_time is None or segment.start_time <= max_time]
    if not filtered:
        return

    fig, ax = plt.subplots(figsize=(11, 8))
    color_map = _drone_color_map(segment.drone_id for segment in filtered)
    executed_order_ids = {segment.order_id for segment in filtered if segment.order_id}
    merchants = [orders[order_id].merchant_loc for order_id in executed_order_ids]
    customers = [orders[order_id].customer_loc for order_id in executed_order_ids]

    if merchants:
        ax.scatter([item.x for item in merchants], [item.y for item in merchants], s=12, c="#d97706", alpha=0.35, label="Merchants")
    if customers:
        ax.scatter([item.x for item in customers], [item.y for item in customers], s=12, c="#059669", alpha=0.35, label="Customers")
    ax.scatter(
        [station.location.x for station in stations],
        [station.location.y for station in stations],
        s=90,
        c="#111827",
        marker="s",
        label="Stations",
    )

    for segment in filtered:
        color = color_map[segment.drone_id]
        linestyle = "--" if segment.segment_type == "charge" else "-"
        ax.plot([segment.x0, segment.x1], [segment.y0, segment.y1], color=color, linewidth=1.3, alpha=0.85, linestyle=linestyle)
        ax.scatter([segment.x0], [segment.y0], color=color, s=8, alpha=0.45)

    drone_handles = [
        plt.Line2D([0], [0], color=color_map[drone_id], lw=2, label=drone_id)
        for drone_id in sorted(color_map)
    ]
    legend_handles = [
        plt.Line2D([0], [0], marker="s", color="w", markerfacecolor="#111827", markersize=8, label="Stations"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#d97706", markersize=6, label="Merchants"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#059669", markersize=6, label="Customers"),
        plt.Line2D([0], [0], color="#374151", lw=2, label="Flight"),
        plt.Line2D([0], [0], color="#374151", lw=2, linestyle="--", label="Charge"),
    ]
    ax.legend(handles=legend_handles + drone_handles, loc="upper left", fontsize=8, ncol=2)
    ax.set_xlim(x_bounds)
    ax.set_ylim(y_bounds)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (km)")
    ax.set_ylabel("y (km)")
    ax.set_title(title)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
