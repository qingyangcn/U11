from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path

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
from analysis.training_logger import EpisodeStatsCallback
from config.scenarios.high_load import build_config as build_high_load
from config.scenarios.low_load import build_config as build_low_load
from config.scenarios.medium_load import build_config as build_medium_load
from env.generator import ensure_fitted_artifacts
from gym_env.delivery_env import DeliveryEnv


def build_scenario(name: str):
    mapping = {
        "low": build_low_load,
        "medium": build_medium_load,
        "high": build_high_load,
    }
    return mapping[name]()


def evaluate_trained_model(model, cfg, eval_seeds: list[int]):
    rows = []
    env = DeliveryEnv(cfg, seed=eval_seeds[0] if eval_seeds else None)
    for episode, eval_seed in enumerate(eval_seeds):
        obs, info = env.reset(seed=eval_seed)
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True, action_masks=env.action_masks())
            obs, reward, terminated, truncated, info = env.step(int(action))
            done = terminated or truncated
        row = {"episode_index": episode, "eval_seed": eval_seed}
        row.update(info["episode_stats"])
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=["low", "medium", "high"], default="high")
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = build_scenario(args.scenario)
    if args.timesteps is not None:
        cfg.training.total_timesteps = args.timesteps
    ensure_fitted_artifacts(cfg)

    from sb3_contrib import MaskablePPO
    from sb3_contrib.common.wrappers import ActionMasker

    run_name = build_run_name("train", args.scenario, args.seed)
    run_dir = cfg.paths.log_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    save_summary_json(run_dir / "config.json", asdict(cfg))

    rollout_steps = min(cfg.training.n_steps, max(8, cfg.training.total_timesteps))
    batch_size = min(cfg.training.batch_size, rollout_steps)
    env = DeliveryEnv(cfg, seed=args.seed)
    wrapped_env = ActionMasker(env, lambda wrapped: wrapped.action_masks())
    episode_log_path = run_dir / "train_episodes.csv"
    callback = EpisodeStatsCallback(episode_log_path)
    model = MaskablePPO(
        "MlpPolicy",
        wrapped_env,
        learning_rate=cfg.training.learning_rate,
        n_steps=rollout_steps,
        batch_size=batch_size,
        gamma=cfg.training.gamma,
        gae_lambda=cfg.training.gae_lambda,
        ent_coef=cfg.training.ent_coef,
        vf_coef=cfg.training.vf_coef,
        verbose=1,
        seed=args.seed,
        device='cuda',
    )
    model.learn(total_timesteps=cfg.training.total_timesteps, progress_bar=False, callback=callback)

    model_path = cfg.paths.checkpoint_dir / f"maskableppo_{args.scenario}.zip"
    model.save(model_path)

    train_frame = save_episode_table(episode_log_path, callback.rows)
    train_summary = summarize_episode_stats(train_frame)
    save_summary_json(run_dir / "train_summary.json", train_summary)
    save_summary_table(run_dir / "train_summary.csv", train_summary)
    plot_episode_metric_bundle(
        train_frame,
        metrics=[
            "episode_reward",
            "episode_length",
            "delivered_orders",
            "canceled_orders",
            "completion_rate",
            "cancel_rate",
            "total_lateness_min",
            "total_energy",
        ],
        output_dir=run_dir,
        prefix="train",
    )

    eval_seeds = build_eval_seeds(args.seed + 1000, cfg.training.eval_episodes)
    eval_rows = evaluate_trained_model(model, cfg, eval_seeds)
    eval_frame = save_episode_table(run_dir / "eval_episodes.csv", eval_rows)
    eval_summary = summarize_episode_stats(eval_frame)
    eval_summary["num_eval_seeds"] = float(len(eval_seeds))
    save_summary_json(run_dir / "eval_summary.json", eval_summary)
    save_summary_table(run_dir / "eval_summary.csv", eval_summary)
    plot_episode_metric_bundle(
        eval_frame,
        metrics=[
            "delivered_orders",
            "canceled_orders",
            "completion_rate",
            "cancel_rate",
            "total_lateness_min",
            "total_energy",
        ],
        output_dir=run_dir,
        prefix="eval",
    )

    print("saved", model_path)
    print("logs", run_dir)


if __name__ == "__main__":
    main()
