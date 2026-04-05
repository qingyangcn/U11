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
    summarize_episode_stats,
)
from analysis.plots import plot_episode_metric_bundle
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


def resolve_model_path(model_arg: str, checkpoint_dir: Path) -> Path:
    raw_path = Path(model_arg)
    candidates = []
    if raw_path.suffix:
        candidates.append(raw_path)
        candidates.append(checkpoint_dir / raw_path.name)
    else:
        candidates.append(raw_path)
        candidates.append(raw_path.with_suffix(".zip"))
        candidates.append(checkpoint_dir / raw_path.name)
        candidates.append(checkpoint_dir / f"{raw_path.name}.zip")
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    searched = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(
        f"Cannot find model '{model_arg}'. Searched: {searched}"
    )


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


def build_alias_rows(tag: str, episode: int, eval_seed: int, alias_counter: dict[str, Counter[str]]) -> list[dict]:
    rows = []
    for action_name in ACTION_TO_INDEX:
        counter = alias_counter[action_name]
        available_count = int(counter.get("available", 0))
        kept_count = int(counter.get("kept", 0))
        alias_masked_count = int(counter.get("alias_masked", 0))
        rows.append(
            {
                "episode_index": episode,
                "policy": tag,
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


def run_fixed_rule(env: DeliveryEnv, eval_seeds: list[int], action_name: str):
    rows = []
    alias_rows = []
    action_idx = ACTION_TO_INDEX[action_name]
    for episode, eval_seed in enumerate(eval_seeds):
        _, info = env.reset(seed=eval_seed)
        done = False
        alias_counter = _new_alias_counter()
        while not done:
            update_alias_counter(alias_counter, env.action_alias_info())
            _, _, terminated, truncated, info = env.step(action_idx)
            done = terminated or truncated
        row = {"episode_index": episode, "policy": action_name, "eval_seed": eval_seed}
        row.update(info["episode_stats"])
        rows.append(row)
        alias_rows.extend(build_alias_rows(action_name, episode, eval_seed, alias_counter))
    return rows, alias_rows


def run_model(env: DeliveryEnv, model, eval_seeds: list[int], tag: str):
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
        row = {"episode_index": episode, "policy": tag, "eval_seed": eval_seed}
        row.update(info["episode_stats"])
        rows.append(row)
        alias_rows.extend(build_alias_rows(tag, episode, eval_seed, alias_counter))
    return rows, alias_rows


def load_maskable_model(model_path: Path, wrapped_env):
    from sb3_contrib import MaskablePPO

    try:
        return MaskablePPO.load(model_path, env=wrapped_env)
    except Exception as exc:  # pragma: no cover - message wrapper
        raise RuntimeError(
            f"Failed to load model '{model_path}'. "
            "The current environment uses a 5-action policy without 'wait'. "
            "Older checkpoints trained with the 6-action policy are not compatible and need retraining."
        ) from exc


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=["low", "medium", "high"], default="high")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--rule", default="nearest")
    parser.add_argument("--model", default="maskableppo_high")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = build_scenario(args.scenario)
    ensure_fitted_artifacts(cfg)
    env = DeliveryEnv(cfg, seed=args.seed)
    eval_seeds = build_eval_seeds(args.seed, args.episodes)

    if args.model:
        from sb3_contrib.common.wrappers import ActionMasker

        wrapped_env = ActionMasker(env, lambda wrapped: wrapped.action_masks())
        model_path = resolve_model_path(args.model, cfg.paths.checkpoint_dir)
        model = load_maskable_model(model_path, wrapped_env)
        tag = model_path.stem
        rows, alias_rows = run_model(env, model, eval_seeds, tag)
        run_name = build_run_name(f"eval_{tag}", args.scenario, args.seed)
    else:
        rows, alias_rows = run_fixed_rule(env, eval_seeds, args.rule)
        run_name = build_run_name(f"eval_{args.rule}", args.scenario, args.seed)

    run_dir = cfg.paths.eval_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    frame = save_episode_table(run_dir / "episodes.csv", rows)
    summary = summarize_episode_stats(frame)
    summary["num_eval_seeds"] = float(len(eval_seeds))
    save_summary_json(run_dir / "summary.json", summary)
    save_summary_table(run_dir / "summary.csv", summary)
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
    plot_episode_metric_bundle(
        frame,
        metrics=[
            "delivered_orders",
            "canceled_orders",
            "completion_rate",
            "cancel_rate",
            "total_lateness_min",
            "total_energy",
        ],
        output_dir=run_dir,
        prefix="episodes",
    )
    print(summary)
    print("saved", run_dir)


if __name__ == "__main__":
    main()
