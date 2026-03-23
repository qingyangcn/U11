"""
Trajectory plotting utility for UAV delivery episodes.

Provides a single entry-point function:

    plot_episode_paths(env, save_path, title="", max_drones=20)

which visualises drone flight paths, base locations and merchant locations
from a finished ThreeObjectiveDroneDeliveryEnv episode and saves the result
as a PNG file.
"""

import os

import matplotlib.pyplot as plt
import numpy as np


def plot_episode_paths(
    env,
    save_path: str,
    title: str = "",
    max_drones: int = 20,
) -> None:
    """Plot drone trajectories and key map locations from a finished episode.

    Args:
        env: ``ThreeObjectiveDroneDeliveryEnv`` instance after the episode.
        save_path: Destination PNG file path (parent directory is created if
            it does not exist).
        title: Figure title.  Should include policy/rule name, seed and key
            metrics so the chart is self-describing.
        max_drones: Maximum number of drone trajectories to render (sorted by
            drone id).  Keeping this bounded avoids an illegible legend.
    """
    parent = os.path.dirname(os.path.abspath(save_path))
    os.makedirs(parent, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 9))

    # ------------------------------------------------------------------ #
    # Drone trajectories                                                   #
    # ------------------------------------------------------------------ #
    # path_history[drone_id] is a deque of (step, x, y) tuples and None
    # sentinels.  None values mark boundaries between delivery trips so
    # this loop can break the plotted line there, preventing false
    # cross-map connections caused by joining the end of one trip to the
    # start of the next.  Within each segment the entries are in
    # chronological order (step is non-decreasing) but we sort defensively
    # to guard against any future out-of-order recording.
    path_history = env.path_visualizer.path_history
    drone_ids = sorted(path_history.keys())[:max_drones]

    cmap = plt.cm.tab20
    for i, drone_id in enumerate(drone_ids):
        raw = list(path_history[drone_id])
        if not raw:
            continue

        color = cmap(i % 20)

        # Split the raw deque into continuous segments at None sentinels.
        # Each segment is a list of (step, x, y) tuples that belong to one
        # uninterrupted flight leg.  Sorting by step within each segment is
        # a defensive measure against any recording-order anomaly.
        segments: list = []
        current_seg: list = []
        for entry in raw:
            if entry is None:
                # Segment-break sentinel: commit current segment and start a new one
                if current_seg:
                    segments.append(current_seg)
                current_seg = []
            else:
                current_seg.append(entry)
        if current_seg:
            segments.append(current_seg)

        # Plot each segment as an independent line so trips are not connected.
        first_seg = True
        for seg in segments:
            if len(seg) < 2:
                continue
            # Sort within segment by step to ensure correct visual order
            seg.sort(key=lambda r: r[0])
            xs = [r[1] for r in seg]
            ys = [r[2] for r in seg]
            ax.plot(
                xs, ys,
                "-",
                color=color,
                alpha=0.55,
                linewidth=1.0,
                label=f"Drone {drone_id}" if first_seg else "_nolegend_",
            )
            first_seg = False

        # Mark the last recorded position with a filled circle
        last_entry = next((e for e in reversed(raw) if e is not None), None)
        if last_entry is not None:
            ax.plot(last_entry[1], last_entry[2], "o", color=color, markersize=4, zorder=4)

    # ------------------------------------------------------------------ #
    # Base locations                                                       #
    # ------------------------------------------------------------------ #
    for base_id, base in env.bases.items():
        loc = base["location"]
        ax.plot(
            loc[0], loc[1],
            "s",
            color="black",
            markersize=12,
            zorder=6,
            label="Base" if base_id == 0 else "_nolegend_",
        )
        ax.annotate(
            f"B{base_id}",
            xy=loc,
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
            fontweight="bold",
        )

    # ------------------------------------------------------------------ #
    # Merchant locations (only those active in this episode, top-K cap)   #
    # ------------------------------------------------------------------ #
    # Collect merchant IDs that actually appeared in orders this episode.
    # order_count on the merchant dict may not be populated in all
    # environments, so we derive activity directly from env.orders.
    merchant_order_counts: dict = {}
    for order in env.orders.values():
        mid = order.get("merchant_id")
        if mid is not None:
            merchant_order_counts[mid] = merchant_order_counts.get(mid, 0) + 1

    # Sort by order activity (most-active first), cap at top_k_merchants
    top_k = getattr(env, "top_k_merchants", len(env.merchants))
    sorted_merchants = sorted(
        env.merchants.items(),
        key=lambda kv: merchant_order_counts.get(kv[0], 0),
        reverse=True,
    )
    # Keep only merchants that had at least one order, up to top_k
    active_merchants = [
        (mid, m) for mid, m in sorted_merchants
        if merchant_order_counts.get(mid, 0) > 0
    ][:top_k]

    merchant_plotted = False
    for _mid, merchant in active_merchants:
        loc = merchant["location"]
        ax.plot(
            loc[0], loc[1],
            "^",
            color="coral",
            markersize=7,
            alpha=0.85,
            zorder=5,
            label="Merchant" if not merchant_plotted else "_nolegend_",
        )
        merchant_plotted = True

    # ------------------------------------------------------------------ #
    # Customer / delivery locations                                        #
    # ------------------------------------------------------------------ #
    # Collect unique customer locations from all orders generated this
    # episode (covers READY, ASSIGNED, PICKED_UP, and COMPLETED states).
    customer_xs: list = []
    customer_ys: list = []
    seen_locs: set = set()
    for order in env.orders.values():
        cloc = order.get("customer_location")
        if cloc is not None and cloc not in seen_locs:
            customer_xs.append(cloc[0])
            customer_ys.append(cloc[1])
            seen_locs.add(cloc)

    if customer_xs:
        ax.scatter(
            customer_xs, customer_ys,
            marker="x",
            color="steelblue",
            s=25,
            linewidths=0.8,
            alpha=0.6,
            zorder=4,
            label="Customer",
        )

    # ------------------------------------------------------------------ #
    # Formatting                                                           #
    # ------------------------------------------------------------------ #
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    if title:
        ax.set_title(title, fontsize=9, wrap=True)

    # Deduplicated legend (skip auto-generated "_nolegend_" entries)
    handles, labels = ax.get_legend_handles_labels()
    seen: dict = {}
    for h, lbl in zip(handles, labels):
        if not lbl.startswith("_") and lbl not in seen:
            seen[lbl] = h
    if seen:
        ax.legend(
            seen.values(), seen.keys(),
            loc="upper right",
            fontsize=7,
            ncol=2,
            framealpha=0.8,
        )

    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close(fig)