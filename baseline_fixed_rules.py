"""
Baseline: Fixed Rule Selection

Implements fixed rule baselines where rule_id is always the same value
(one of 0, 1, 2, 3, 4) for every decision.

Compatible with DecentralizedEventDrivenExecutor.

Rule descriptions (from UAV_ENVIRONMENT_11):
    0: Highest-priority / default rule
    1: Alternative rule 1
    2: EDF (Earliest Deadline First)
    3: Nearest pickup
    4: Slack per distance

Usage:
    # Run with fixed rule 3
    python baseline_fixed_rules.py --rule-id 3 --seed 42

    # Run all 5 fixed rules across multiple seeds
    python baseline_fixed_rules.py --all-rules --seeds 42,43,44

    # Run all 5 fixed rules across multiple seeds and write CSV
    python baseline_fixed_rules.py --all-rules --seeds 42,43,44 --csv-out results.csv

    # Run without MOPSO
    python baseline_fixed_rules.py --rule-id 0 --seed 42 --no-mopso
"""

import argparse
import csv
import os
import sys
from typing import Callable

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from UAV_ENVIRONMENT_11 import ThreeObjectiveDroneDeliveryEnv
from U11_decentralized_execution import DecentralizedEventDrivenExecutor
from U11_ablation import _make_env, _compute_completion_stats

try:
    from U11_plotting import plot_episode_paths as _plot_episode_paths
    _HAS_PLOTTING = True
except ImportError:
    _HAS_PLOTTING = False

try:
    from U10_candidate_generator import MOPSOCandidateGenerator

    _HAS_MOPSO = True
except ImportError:
    _HAS_MOPSO = False


def make_fixed_rule_policy(rule_id: int) -> Callable[[dict], int]:
    """
    Factory function that returns a policy always selecting the given rule_id.

    Args:
        rule_id: Fixed rule identifier in {0, 1, 2, 3, 4}.

    Returns:
        A callable ``policy_fn(local_obs: dict) -> int`` compatible with
        ``DecentralizedEventDrivenExecutor``.
    """
    if rule_id not in range(5):
        raise ValueError(f"rule_id must be in {{0,1,2,3,4}}, got {rule_id}")

    def policy_fn(local_obs: dict) -> int:
        return rule_id

    policy_fn.__name__ = f"fixed_rule_{rule_id}_policy"
    return policy_fn


def run_episode(args, rule_id: int, seed: int) -> dict:
    """Run one episode with a fixed rule policy and return completion stats."""
    np.random.seed(seed)

    env = _make_env(args, order_cutoff_steps=0)

    if args.use_mopso and _HAS_MOPSO:
        try:
            candidate_generator = MOPSOCandidateGenerator(
                candidate_k=args.candidate_k,
                n_particles=10,
                n_iterations=3,
                max_orders=100,
                max_orders_per_drone=10,
                seed=seed,
            )
            env.set_candidate_generator(candidate_generator)
        except (ImportError, Exception):
            pass  # Fall back to built-in candidates when MOPSO unavailable

    fixed_policy = make_fixed_rule_policy(rule_id)

    executor = DecentralizedEventDrivenExecutor(
        env=env,
        policy_fn=fixed_policy,
        max_skip_steps=args.max_skip_steps,
        verbose=False,
    )

    executor.run_episode(max_steps=args.max_steps, seed=seed)

    stats = _compute_completion_stats(env, executor)
    stats['seed'] = seed
    stats['rule_id'] = rule_id
    stats['policy'] = f'fixed_rule_{rule_id}'

    if getattr(args, 'plot_paths', False) and _HAS_PLOTTING:
        gc = stats['general_completion']
        epc = stats['energy_per_completed']
        aw = stats['avg_wait_ready_to_assigned']
        plot_title = (
            f"Fixed rule {rule_id} | seed={seed} | "
            f"GC={gc:.3f}  energy_pc={epc:.3f}  wait_avg={aw:.2f}"
        )
        save_path = os.path.join(
            args.plot_dir, f"baseline_rule_{rule_id}_seed_{seed}.png"
        )
        _plot_episode_paths(
            env, save_path, title=plot_title,
            max_drones=getattr(args, 'plot_max_drones', 20),
        )
        print(f"    Plot saved: {save_path}")

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Baseline: Fixed Rule Selection"
    )

    parser.add_argument("--rule-id", type=int, default=3,
                        help="Fixed rule ID in {0,1,2,3,4} (required unless --all-rules)")
    parser.add_argument("--all-rules", action="store_true", default=False,
                        help="Run all 5 fixed rules (0..4) sequentially")
    parser.add_argument("--seed", type=int, default=21,
                        help="Random seed for a single episode run (default: 21)")
    parser.add_argument("--seeds", type=str, default='21',
                        help="Comma-separated seeds to run multiple episodes "
                             "(overrides --seed when provided)")
    parser.add_argument("--num-drones", type=int, default=1,
                        help="Number of drones (default: 20)")
    parser.add_argument("--obs-max-orders", type=int, default=200,
                        help="Maximum orders in observation (default: 200)")
    parser.add_argument("--top-k-merchants", type=int, default=10,
                        help="Top K merchants (default: 50)")
    parser.add_argument("--candidate-k", type=int, default=10,
                        help="Number of candidates per drone")
    parser.add_argument("--enable-random-events", action="store_true", default=False,
                        help="Enable random events (default: False)")
    parser.add_argument("--max-skip-steps", type=int, default=1,
                        help="Max steps to skip when waiting for decisions (default: 1)")
    parser.add_argument("--max-steps", type=int, default=500,
                        help="Maximum decision steps per episode (default: 500)")
    parser.add_argument("--use-mopso", action="store_true", default=True,
                        help="Use MOPSO candidate generator (default: True)")
    parser.add_argument("--no-mopso", dest="use_mopso", action="store_false",
                        help="Disable MOPSO candidate generator")
    parser.add_argument("--csv-out", type=str, default=None,
                        help="Write per-episode results to this CSV file")
    parser.add_argument("--plot-paths", action="store_true", default=True,
                        help="Save per-episode trajectory PNG plots")
    parser.add_argument("--plot-dir", type=str, default="plots",
                        help="Directory for trajectory plots (default: plots)")
    parser.add_argument("--plot-max-drones", type=int, default=20,
                        help="Maximum number of drone trajectories to render (default: 20)")

    args = parser.parse_args()

    if not args.all_rules and args.rule_id is None:
        parser.error("Provide --rule-id <0..4> or use --all-rules")

    rule_ids = list(range(5)) if args.all_rules else [args.rule_id]
    seeds = ([int(s.strip()) for s in args.seeds.split(',')]
             if args.seeds is not None else [args.seed])

    print("=" * 80)
    print("Baseline: Fixed Rule Selection")
    print(f"  rule_ids={rule_ids}  seeds={seeds}  n_seeds={len(seeds)}")
    print(f"  candidate_k={args.candidate_k}  use_mopso={args.use_mopso and _HAS_MOPSO}")
    print("=" * 80)

    all_stats = []
    # rule_id -> list of per-seed stats
    rule_summary: dict = {}

    for rule_id in rule_ids:
        rule_stats = []
        print(f"\n[rule_id={rule_id}]")
        for seed in seeds:
            stats = run_episode(args, rule_id=rule_id, seed=seed)
            all_stats.append(stats)
            rule_stats.append(stats)
            print(f"  seed={seed:>6}  GC={stats['general_completion']:.4f}  "
                  f"generated={stats['generated_total']}  "
                  f"completed={stats['completed_total']}  "
                  f"reward={stats['cumulative_reward']:.2f}  "
                  f"energy_pc={stats['energy_per_completed']:.3f}  "
                  f"wait_avg={stats['avg_wait_ready_to_assigned']:.2f}")

        gc_values = [s['general_completion'] for s in rule_stats]
        if len(gc_values) > 1:
            print(f"  ── aggregate (n={len(gc_values)}) ──")
            print(f"     mean={float(np.mean(gc_values)):.4f}  "
                  f"std={float(np.std(gc_values)):.4f}  "
                  f"min={float(np.min(gc_values)):.4f}  "
                  f"max={float(np.max(gc_values)):.4f}")
        rule_summary[rule_id] = gc_values

    # Overall comparison table
    if len(rule_ids) > 1:
        print("\n" + "=" * 80)
        print(f"Overall Comparison  (n_seeds={len(seeds)})")
        print(f"  {'rule_id':>7}  {'mean_GC':>9}  {'std_GC':>8}  "
              f"{'min_GC':>8}  {'max_GC':>8}")
        for rule_id in rule_ids:
            gc = rule_summary[rule_id]
            print(f"  {rule_id:>7}  {float(np.mean(gc)):>9.4f}  "
                  f"{float(np.std(gc)):>8.4f}  "
                  f"{float(np.min(gc)):>8.4f}  "
                  f"{float(np.max(gc)):>8.4f}")
        # Highlight best rule by mean GC
        best_rule = max(rule_ids, key=lambda r: float(np.mean(rule_summary[r])))
        print(f"\n  Best rule by mean_GC: rule_id={best_rule}  "
              f"mean_GC={float(np.mean(rule_summary[best_rule])):.4f}")

    # CSV output
    if args.csv_out:
        fieldnames = [
            'rule_id', 'policy', 'seed',
            'generated_total', 'completed_total', 'general_completion',
            'cumulative_reward', 'energy_total', 'energy_per_completed',
            'avg_wait_ready_to_assigned', 'p95_wait_ready_to_assigned',
        ]
        with open(args.csv_out, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in all_stats:
                writer.writerow({k: row[k] for k in fieldnames})
        print(f"\nCSV written to: {args.csv_out}")

    print("=" * 80)
    return all_stats


if __name__ == "__main__":
    main()