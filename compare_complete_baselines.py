from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from analysis.metrics import build_eval_seeds, build_run_name, save_episode_table, save_summary_json, summarize_by_group
from analysis.plots import plot_grouped_metric_bundle
from baselines.full_rule_schedulers import (
    EarliestDeadlineFullScheduler,
    MinimumSlackFullScheduler,
    NearestFullScheduler,
)
from config.scenarios.high_load import build_config as build_high_load
from config.scenarios.low_load import build_config as build_low_load
from config.scenarios.medium_load import build_config as build_medium_load
from env.generator import ensure_fitted_artifacts
from env.simulator import EventDrivenSimulator
from gym_env.delivery_env import DeliveryEnv
from rl.rule_actions import ACTION_TO_INDEX
from runner.compare_swarm_baselines import (
    evaluate_swarm,
    load_maskable_model,
    resolve_default_model_path,
    resolve_model_path,
)


def build_scenario(name: str):
    mapping = {
        "low": build_low_load,
        "medium": build_medium_load,
        "high": build_high_load,
    }
    return mapping[name]()


def evaluate_complete_rule(cfg, eval_seeds: list[int], method_name: str) -> list[dict]:
    rows = []
    scheduler_cls = {
        "nearest_full": NearestFullScheduler,
        "earliest_deadline_full": EarliestDeadlineFullScheduler,
        "minimum_slack_full": MinimumSlackFullScheduler,
    }[method_name]
    for episode, eval_seed in enumerate(eval_seeds):
        scheduler = scheduler_cls(cfg)
        sim = EventDrivenSimulator(cfg, seed=eval_seed, swarm_scheduler=scheduler)
        sim.reset(seed=eval_seed)
        while not sim.done:
            sim.step_swarm()
        row = {"episode_index": episode, "method": method_name, "eval_seed": eval_seed}
        row.update(sim.episode_stats)
        rows.append(row)
    return rows


def _new_alias_counter() -> dict[str, Counter[str]]:
    return {action_name: Counter() for action_name in ACTION_TO_INDEX}


def update_alias_counter(alias_counter: dict[str, Counter[str]], alias_info: dict[str, dict]) -> None:
    for action_name in ACTION_TO_INDEX:
        detail = alias_info.get(action_name, {})
        if detail.get("available", False):
            alias_counter[action_name]["available"] += 1
        if detail.get("kept", False):
            alias_counter[action_name]["kept"] += 1
        if detail.get("available", False) and not detail.get("kept", False) and detail.get("alias_of") is not None:
            alias_counter[action_name]["alias_masked"] += 1


def build_alias_rows(method_name: str, episode: int, eval_seed: int, alias_counter: dict[str, Counter[str]]) -> list[dict]:
    rows = []
    for action_name in ACTION_TO_INDEX:
        counter = alias_counter[action_name]
        available_count = int(counter.get("available", 0))
        kept_count = int(counter.get("kept", 0))
        alias_masked_count = int(counter.get("alias_masked", 0))
        rows.append(
            {
                "episode_index": episode,
                "method": method_name,
                "eval_seed": eval_seed,
                "action_name": action_name,
                "available_count": available_count,
                "kept_count": kept_count,
                "alias_masked_count": alias_masked_count,
                "kept_rate": float(kept_count / available_count) if available_count else 0.0,
                "alias_mask_rate": float(alias_masked_count / available_count) if available_count else 0.0,
            }
        )
    return rows


def evaluate_model_with_alias(env: DeliveryEnv, eval_seeds: list[int], model, method_name: str) -> tuple[list[dict], list[dict]]:
    rows = []
    alias_rows = []
    for episode, eval_seed in enumerate(eval_seeds):
        obs, info = env.reset(seed=eval_seed)
        done = False
        alias_counter = _new_alias_counter()
        while not done:
            update_alias_counter(alias_counter, env.action_alias_info())
            action, _ = model.predict(obs, deterministic=True, action_masks=env.action_masks())
            obs, _, terminated, truncated, info = env.step(int(action))
            done = terminated or truncated
        row = {"episode_index": episode, "method": method_name, "eval_seed": eval_seed}
        row.update(info["episode_stats"])
        rows.append(row)
        alias_rows.extend(build_alias_rows(method_name, episode, eval_seed, alias_counter))
    return rows, alias_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=["low", "medium", "high"], default="high")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", default="")
    parser.add_argument("--model-label", default="main_method")
    args = parser.parse_args()

    cfg = build_scenario(args.scenario)
    ensure_fitted_artifacts(cfg)
    eval_seeds = build_eval_seeds(args.seed, args.episodes)

    all_rows: list[dict] = []
    alias_rows: list[dict] = []
    for method_name in (
        "nearest_full",
        "earliest_deadline_full",
        "minimum_slack_full",
        "pure_pso",
        "pure_gwo",
    ):
        if method_name in {"pure_pso", "pure_gwo"}:
            all_rows.extend(evaluate_swarm(cfg, eval_seeds, method_name))
        else:
            all_rows.extend(evaluate_complete_rule(cfg, eval_seeds, method_name))

    model_path = resolve_model_path(args.model, cfg.paths.checkpoint_dir) if args.model else resolve_default_model_path(cfg.paths.checkpoint_dir, args.scenario)
    if model_path is not None:
        from sb3_contrib.common.wrappers import ActionMasker

        env = DeliveryEnv(cfg, seed=args.seed)
        wrapped_env = ActionMasker(env, lambda wrapped: wrapped.action_masks())
        model = load_maskable_model(model_path, wrapped_env)
        model_rows, model_alias_rows = evaluate_model_with_alias(env, eval_seeds, model, args.model_label or model_path.stem)
        all_rows.extend(model_rows)
        alias_rows.extend(model_alias_rows)

    run_name = build_run_name("complete_compare", args.scenario, args.seed)
    run_dir = cfg.paths.eval_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    frame = save_episode_table(run_dir / "episodes.csv", all_rows)
    grouped = summarize_by_group(frame, "method")
    grouped["num_eval_seeds"] = float(len(eval_seeds))
    grouped.to_csv(run_dir / "summary_by_method.csv", index=False)
    save_summary_json(run_dir / "summary_by_method.json", grouped.to_dict(orient="records"))
    alias_frame = pd.DataFrame(alias_rows)
    if not alias_frame.empty:
        alias_frame.to_csv(run_dir / "alias_usage.csv", index=False)
        alias_summary = (
            alias_frame.groupby(["method", "action_name"], as_index=False)
            .agg(
                available_count_mean=("available_count", "mean"),
                available_count_std=("available_count", "std"),
                kept_count_mean=("kept_count", "mean"),
                kept_count_std=("kept_count", "std"),
                alias_masked_count_mean=("alias_masked_count", "mean"),
                alias_masked_count_std=("alias_masked_count", "std"),
                kept_rate_mean=("kept_rate", "mean"),
                kept_rate_std=("kept_rate", "std"),
                alias_mask_rate_mean=("alias_mask_rate", "mean"),
                alias_mask_rate_std=("alias_mask_rate", "std"),
            )
            .fillna(0.0)
        )
        alias_summary.to_csv(run_dir / "alias_usage_summary.csv", index=False)
        save_summary_json(run_dir / "alias_usage_summary.json", alias_summary.to_dict(orient="records"))
    plot_grouped_metric_bundle(
        grouped,
        category_col="method",
        metrics=[
            "completion_rate_mean",
            "cancel_rate_mean",
            "total_lateness_min_mean",
            "total_energy_mean",
        ],
        output_dir=run_dir,
        prefix="complete_compare",
    )
    print(grouped)
    print("saved", run_dir)


if __name__ == "__main__":
    main()
