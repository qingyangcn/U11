from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from analysis.metrics import (
    build_eval_seeds,
    build_run_name,
    save_episode_table,
    save_summary_json,
    save_summary_table,
    summarize_by_group,
)
from analysis.plots import plot_grouped_metric_bundle
from baselines.rule_baselines import RULE_BASELINE_POLICIES, select_rule_action
from config.scenarios.high_load import build_config as build_high_load
from config.scenarios.low_load import build_config as build_low_load
from config.scenarios.medium_load import build_config as build_medium_load
from env.generator import ensure_fitted_artifacts
from gym_env.delivery_env import DeliveryEnv
from rl.rule_actions import ACTION_TO_INDEX
from runner.compare_swarm_baselines import load_maskable_model, resolve_default_model_path, resolve_model_path


def build_scenario(name: str):
    mapping = {
        "low": build_low_load,
        "medium": build_medium_load,
        "high": build_high_load,
    }
    return mapping[name]()


def build_action_rows(policy_name: str, episode: int, eval_seed: int, action_counter: Counter[str]) -> list[dict]:
    total_actions = sum(action_counter.values())
    rows = []
    for action_name in ACTION_TO_INDEX:
        count = int(action_counter.get(action_name, 0))
        share = float(count / total_actions) if total_actions else 0.0
        rows.append(
            {
                "episode_index": episode,
                "policy": policy_name,
                "eval_seed": eval_seed,
                "action_name": action_name,
                "count": count,
                "share": share,
            }
        )
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


def build_alias_rows(policy_name: str, episode: int, eval_seed: int, alias_counter: dict[str, Counter[str]]) -> list[dict]:
    rows = []
    for action_name in ACTION_TO_INDEX:
        counter = alias_counter[action_name]
        available_count = int(counter.get("available", 0))
        kept_count = int(counter.get("kept", 0))
        alias_masked_count = int(counter.get("alias_masked", 0))
        rows.append(
            {
                "episode_index": episode,
                "policy": policy_name,
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


def evaluate_policy(env: DeliveryEnv, eval_seeds: list[int], policy_name: str):
    rows = []
    action_rows = []
    alias_rows = []
    for episode, eval_seed in enumerate(eval_seeds):
        _, info = env.reset(seed=eval_seed)
        done = False
        action_counter: Counter[str] = Counter()
        alias_counter = _new_alias_counter()
        while not done:
            update_alias_counter(alias_counter, env.action_alias_info())
            drone_id = env.simulator.current_decision.drone_id
            projections = env.simulator.get_feasible_projections(drone_id)
            action_name = select_rule_action(policy_name, projections, env.simulator.orders)
            action_counter[action_name] += 1
            _, _, terminated, truncated, info = env.step(ACTION_TO_INDEX[action_name])
            done = terminated or truncated
        row = {"episode_index": episode, "policy": policy_name, "eval_seed": eval_seed}
        row.update(info["episode_stats"])
        rows.append(row)
        action_rows.extend(build_action_rows(policy_name, episode, eval_seed, action_counter))
        alias_rows.extend(build_alias_rows(policy_name, episode, eval_seed, alias_counter))
    return rows, action_rows, alias_rows


def evaluate_model(env: DeliveryEnv, eval_seeds: list[int], model, policy_name: str):
    rows = []
    action_rows = []
    alias_rows = []
    for episode, eval_seed in enumerate(eval_seeds):
        obs, info = env.reset(seed=eval_seed)
        done = False
        action_counter: Counter[str] = Counter()
        alias_counter = _new_alias_counter()
        while not done:
            update_alias_counter(alias_counter, env.action_alias_info())
            action, _ = model.predict(obs, deterministic=True, action_masks=env.action_masks())
            action_name = next(name for name, idx in ACTION_TO_INDEX.items() if idx == int(action))
            action_counter[action_name] += 1
            obs, _, terminated, truncated, info = env.step(int(action))
            done = terminated or truncated
        row = {"episode_index": episode, "policy": policy_name, "eval_seed": eval_seed}
        row.update(info["episode_stats"])
        rows.append(row)
        action_rows.extend(build_action_rows(policy_name, episode, eval_seed, action_counter))
        alias_rows.extend(build_alias_rows(policy_name, episode, eval_seed, alias_counter))
    return rows, action_rows, alias_rows


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
    env = DeliveryEnv(cfg, seed=args.seed)
    eval_seeds = build_eval_seeds(args.seed, args.episodes)

    all_rows = []
    action_rows = []
    alias_rows = []
    for policy_name in RULE_BASELINE_POLICIES:
        rows, actions, aliases = evaluate_policy(env, eval_seeds, policy_name)
        all_rows.extend(rows)
        action_rows.extend(actions)
        alias_rows.extend(aliases)
    model_path = resolve_model_path(args.model, cfg.paths.checkpoint_dir) if args.model else resolve_default_model_path(cfg.paths.checkpoint_dir, args.scenario)
    if model_path is not None:
        from sb3_contrib.common.wrappers import ActionMasker

        wrapped_env = ActionMasker(env, lambda wrapped: wrapped.action_masks())
        model = load_maskable_model(model_path, wrapped_env)
        policy_name = args.model_label or model_path.stem
        rows, actions, aliases = evaluate_model(env, eval_seeds, model, policy_name)
        all_rows.extend(rows)
        action_rows.extend(actions)
        alias_rows.extend(aliases)

    run_name = build_run_name("ablation_compare", args.scenario, args.seed)
    run_dir = cfg.paths.eval_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    frame = save_episode_table(run_dir / "episodes.csv", all_rows)
    grouped = summarize_by_group(frame, "policy")
    grouped["num_eval_seeds"] = float(len(eval_seeds))
    grouped.to_csv(run_dir / "summary_by_policy.csv", index=False)
    save_summary_json(run_dir / "summary_by_policy.json", grouped.to_dict(orient="records"))
    action_frame = pd.DataFrame(action_rows)
    if not action_frame.empty:
        action_frame.to_csv(run_dir / "action_usage.csv", index=False)
        action_summary = (
            action_frame.groupby(["policy", "action_name"], as_index=False)
            .agg(
                count_mean=("count", "mean"),
                count_std=("count", "std"),
                share_mean=("share", "mean"),
                share_std=("share", "std"),
            )
            .fillna(0.0)
        )
        action_summary.to_csv(run_dir / "action_usage_summary.csv", index=False)
        save_summary_json(run_dir / "action_usage_summary.json", action_summary.to_dict(orient="records"))
    alias_frame = pd.DataFrame(alias_rows)
    if not alias_frame.empty:
        alias_frame.to_csv(run_dir / "alias_usage.csv", index=False)
        alias_summary = (
            alias_frame.groupby(["policy", "action_name"], as_index=False)
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
        category_col="policy",
        metrics=[
            "completion_rate_mean",
            "cancel_rate_mean",
            "total_lateness_min_mean",
            "total_energy_mean",
        ],
        output_dir=run_dir,
        prefix="ablation_compare",
    )
    print(grouped)
    print("saved", run_dir)


if __name__ == "__main__":
    main()
