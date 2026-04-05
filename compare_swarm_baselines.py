from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from analysis.metrics import build_eval_seeds, build_run_name, save_episode_table, save_summary_json, summarize_by_group
from analysis.plots import plot_grouped_metric_bundle
from config.scenarios.high_load import build_config as build_high_load
from config.scenarios.low_load import build_config as build_low_load
from config.scenarios.medium_load import build_config as build_medium_load
from env.generator import ensure_fitted_artifacts
from env.simulator import EventDrivenSimulator
from gym_env.delivery_env import DeliveryEnv
from swarm.pure_gwo import PureGWOScheduler
from swarm.pure_pso import PurePSOScheduler


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
    raise FileNotFoundError(f"Cannot find model '{model_arg}'. Searched: {searched}")


def resolve_default_model_path(checkpoint_dir: Path, scenario: str) -> Path | None:
    candidates = [
        checkpoint_dir / f"maskableppo_{scenario}.zip",
        checkpoint_dir / f"maskableppo_{scenario} (reward1.5).zip",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return None


def evaluate_model(env: DeliveryEnv, eval_seeds: list[int], model, method_name: str) -> list[dict]:
    rows = []
    for episode, eval_seed in enumerate(eval_seeds):
        obs, info = env.reset(seed=eval_seed)
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True, action_masks=env.action_masks())
            obs, _, terminated, truncated, info = env.step(int(action))
            done = terminated or truncated
        row = {"episode_index": episode, "method": method_name, "eval_seed": eval_seed}
        row.update(info["episode_stats"])
        rows.append(row)
    return rows


def evaluate_swarm(cfg, eval_seeds: list[int], method_name: str) -> list[dict]:
    rows = []
    scheduler_cls = PurePSOScheduler if method_name == "pure_pso" else PureGWOScheduler
    for episode, eval_seed in enumerate(eval_seeds):
        scheduler = scheduler_cls(cfg, seed=eval_seed)
        sim = EventDrivenSimulator(cfg, seed=eval_seed, swarm_scheduler=scheduler)
        sim.reset(seed=eval_seed)
        while not sim.done:
            sim.step_swarm()
        row = {"episode_index": episode, "method": method_name, "eval_seed": eval_seed}
        row.update(sim.episode_stats)
        rows.append(row)
    return rows


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
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", default="")
    parser.add_argument("--model-label", default="main_method")
    args = parser.parse_args()

    cfg = build_scenario(args.scenario)
    ensure_fitted_artifacts(cfg)
    eval_seeds = build_eval_seeds(args.seed, args.episodes)

    all_rows: list[dict] = []
    for method_name in ("pure_pso", "pure_gwo"):
        all_rows.extend(evaluate_swarm(cfg, eval_seeds, method_name))

    model_path = resolve_model_path(args.model, cfg.paths.checkpoint_dir) if args.model else resolve_default_model_path(cfg.paths.checkpoint_dir, args.scenario)
    if model_path is not None:
        from sb3_contrib.common.wrappers import ActionMasker

        env = DeliveryEnv(cfg, seed=args.seed)
        wrapped_env = ActionMasker(env, lambda wrapped: wrapped.action_masks())
        model = load_maskable_model(model_path, wrapped_env)
        all_rows.extend(evaluate_model(env, eval_seeds, model, args.model_label or model_path.stem))

    run_name = build_run_name("swarm_compare", args.scenario, args.seed)
    run_dir = cfg.paths.eval_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    frame = save_episode_table(run_dir / "episodes.csv", all_rows)
    grouped = summarize_by_group(frame, "method")
    grouped["num_eval_seeds"] = float(len(eval_seeds))
    grouped.to_csv(run_dir / "summary_by_method.csv", index=False)
    save_summary_json(run_dir / "summary_by_method.json", grouped.to_dict(orient="records"))
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
        prefix="swarm_compare",
    )
    print(grouped)
    print("saved", run_dir)


if __name__ == "__main__":
    main()
