from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from analysis.metrics import build_run_name, save_summary_json
from analysis.trajectory import attach_segment_recorder, plot_trajectories, segments_to_frame
from config.scenarios.high_load import build_config as build_high_load
from config.scenarios.low_load import build_config as build_low_load
from config.scenarios.medium_load import build_config as build_medium_load
from env.generator import ensure_fitted_artifacts
from gym_env.delivery_env import DeliveryEnv
from rl.rule_actions import ACTION_TO_INDEX


def build_scenario(name: str):
    mapping = {
        "low": build_low_load,
        "medium": build_medium_load,
        "high": build_high_load,
    }
    return mapping[name]()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=["low", "medium", "high"], default="medium")
    parser.add_argument("--rule", default="nearest")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--focus-minutes", type=float, default=180.0)
    args = parser.parse_args()

    cfg = build_scenario(args.scenario)
    ensure_fitted_artifacts(cfg)
    env = DeliveryEnv(cfg, seed=args.seed)
    segments = attach_segment_recorder(env)

    run_name = build_run_name("trajectory", args.scenario, args.seed)
    run_dir = cfg.paths.figure_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    _, info = env.reset(seed=args.seed)
    done = False
    action_idx = ACTION_TO_INDEX[args.rule]
    while not done:
        _, _, terminated, truncated, info = env.step(action_idx)
        done = terminated or truncated

    frame = segments_to_frame(segments)
    frame.to_csv(run_dir / "segments.csv", index=False)
    plot_trajectories(
        output_path=run_dir / "trajectory_full.png",
        segments=segments,
        stations=env.simulator.stations,
        orders=env.simulator.orders,
        x_bounds=env.simulator.x_bounds,
        y_bounds=env.simulator.y_bounds,
        title=f"UAV trajectories | scenario={args.scenario} rule={args.rule} seed={args.seed}",
    )
    plot_trajectories(
        output_path=run_dir / "trajectory_focus.png",
        segments=segments,
        stations=env.simulator.stations,
        orders=env.simulator.orders,
        x_bounds=env.simulator.x_bounds,
        y_bounds=env.simulator.y_bounds,
        title=f"UAV trajectories first {args.focus_minutes:.0f} min | scenario={args.scenario} rule={args.rule} seed={args.seed}",
        max_time=args.focus_minutes,
    )

    summary = {
        "scenario": args.scenario,
        "rule": args.rule,
        "seed": args.seed,
        "segment_count": int(len(frame)),
        "episode_stats": info["episode_stats"],
        "focus_minutes": float(args.focus_minutes),
        "x_bounds": list(env.simulator.x_bounds),
        "y_bounds": list(env.simulator.y_bounds),
    }
    save_summary_json(run_dir / "summary.json", summary)
    print("saved", run_dir)


if __name__ == "__main__":
    main()
