"""
U11 Ablation & Sanity Check for Decentralized Event-Driven Execution

K (order_cutoff_steps) definition — **Mode 1 only**:
    The environment stops generating/accepting new orders K steps before the
    business-end step, but continues delivering already-accepted orders until
    the episode finishes.  K=0 means no early cutoff.

Usage:
    # Test with random policy (single episode)
    python U11_ablation.py

    # Test with trained policy
    python U11_ablation.py --model-path ./models/u10/ppo_u10_final.zip

    # Multi-seed evaluation (random policy, 3 seeds)
    python U11_ablation.py --seeds 21,42,43

    # Multi-seed evaluation with trained policy
    python U11_ablation.py --model-path ppo_u11.zip --seeds 21,42,43 --csv-out results.csv

    # Quick test (fewer steps)
    python U11_ablation.py --max-steps 100

    # Ablation: scan environment order_cutoff_steps (K) across multiple seeds
    python U11_ablation.py --ablation-cutoff --cutoff-values 0,6,12,18,24 --seeds 42,43 --csv-out out.csv
"""

import argparse
import csv
import math
import os
import sys
from collections import Counter

import numpy as np

# Add repo root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from UAV_ENVIRONMENT_11 import ThreeObjectiveDroneDeliveryEnv
from U11_decentralized_execution import DecentralizedEventDrivenExecutor

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


def random_policy(local_obs: dict) -> int:
    """Simple random policy for testing."""
    return 3


def load_trained_policy(model_path: str, vecnormalize_path: str = None):
    """
    Load a trained PPO policy.

    Supports the current Box/ndarray observation architecture: ``local_obs`` is a
    flat ``np.ndarray`` and ``obs_rms`` (from VecNormalize) is a single
    ``RunningMeanStd`` instance rather than a per-key dict.

    Args:
        model_path: Path to trained model (.zip file)
        vecnormalize_path: Path to VecNormalize stats (.pkl file)

    Returns:
        Policy function that takes local_obs (np.ndarray) and returns rule_id
    """
    try:
        from stable_baselines3 import PPO
    except ImportError:
        raise RuntimeError("Please install stable-baselines3: pip install stable-baselines3")

    # Load model
    model = PPO.load(model_path)

    # Load VecNormalize stats if provided and apply during inference
    obs_rms = None
    clip_obs = 10.0
    epsilon = 1e-8

    if vecnormalize_path and os.path.exists(vecnormalize_path):
        try:
            try:
                import cloudpickle as _pkl_mod
            except ImportError:
                import pickle as _pkl_mod
            with open(vecnormalize_path, 'rb') as f:
                _vn = _pkl_mod.load(f)
            _obs_rms = getattr(_vn, 'obs_rms', None)
            _clip_obs = float(getattr(_vn, 'clip_obs', 10.0))
            _norm_obs = bool(getattr(_vn, 'norm_obs', True))
            _epsilon = float(getattr(_vn, 'epsilon', 1e-8))
            if _norm_obs and _obs_rms is not None:
                obs_rms = _obs_rms
                clip_obs = _clip_obs
                epsilon = _epsilon
                print(f"VecNormalize stats loaded from: {vecnormalize_path}")
                print(f"  norm_obs enabled, clip_obs={clip_obs}")
                _mean_avg = float(np.array(obs_rms.mean).mean())
                _std_avg = float(np.sqrt(np.array(obs_rms.var).mean()))
                print(f"  obs_rms: mean_avg={_mean_avg:.4f}, std_avg={_std_avg:.4f}")
            else:
                print(f"VecNormalize loaded from: {vecnormalize_path} "
                      f"(norm_obs={_norm_obs}, no normalization applied)")
        except Exception as e:
            print(f"Warning: Failed to load VecNormalize stats from {vecnormalize_path}: {e}")
            print("  Proceeding without observation normalization.")
    elif vecnormalize_path:
        print(f"Warning: VecNormalize stats file not found: {vecnormalize_path}")
        print("  Proceeding without observation normalization.")

    def policy_fn(local_obs: np.ndarray) -> int:
        """Wrapper function for trained policy (Box/ndarray observation space)."""
        if obs_rms is not None:
            # Apply VecNormalize observation normalization to match training preprocessing.
            # Replicates VecNormalize._normalize_obs: clip((obs - mean) / sqrt(var + eps), ±clip_obs)
            obs_to_predict = np.clip(
                (local_obs - obs_rms.mean) / np.sqrt(obs_rms.var + epsilon),
                -clip_obs, clip_obs,
            ).astype(np.float32)
        else:
            obs_to_predict = local_obs
        action, _ = model.predict(obs_to_predict, deterministic=True)
        return int(action)

    return policy_fn


def _make_env(args, order_cutoff_steps: int = 0) -> ThreeObjectiveDroneDeliveryEnv:
    """Create environment with the given configuration."""
    return ThreeObjectiveDroneDeliveryEnv(
        grid_size=16,
        num_drones=args.num_drones,
        max_orders=args.obs_max_orders,
        num_bases=2,
        steps_per_hour=12,
        drone_max_capacity=10,
        top_k_merchants=args.top_k_merchants,
        reward_output_mode="scalar",
        enable_random_events=args.enable_random_events,
        debug_state_warnings=False,
        fixed_objective_weights=(0.3, 0.2, 0.5),
        num_candidates=args.candidate_k,
        rule_count=5,
        enable_diagnostics=False,
        energy_e0=0.1,
        energy_alpha=0.5,
        battery_return_threshold=10.0,
        multi_objective_mode="fixed",
        candidate_update_interval=8,
        candidate_fallback_enabled=False,
        order_cutoff_steps=order_cutoff_steps,
    )


def _compute_completion_stats(
        env: ThreeObjectiveDroneDeliveryEnv,
        executor=None,
) -> dict:
    """Compute unified evaluation metrics from a finished episode.

    Args:
        env: The finished environment.
        executor: Optional ``DecentralizedEventDrivenExecutor`` to read
                  ``cumulative_reward`` from.

    Returns:
        Dict with keys:
            generated_total, completed_total, general_completion,
            cumulative_reward, energy_total, energy_per_completed,
            avg_wait_ready_to_assigned, p95_wait_ready_to_assigned.
    """
    generated_total = env.daily_stats['orders_generated']
    completed_total = env.daily_stats['orders_completed']
    general_completion = completed_total / generated_total if generated_total > 0 else 0.0

    # cumulative_reward
    cumulative_reward = float(executor.cumulative_reward) if executor is not None else float('nan')

    # energy metrics
    energy_total = float(env.daily_stats.get('energy_consumed', 0.0))
    energy_per_completed = (
        energy_total / completed_total if completed_total > 0 else float('nan')
    )

    # wait-ready-to-assigned
    wait_samples = list(env.metrics.get('wait_ready_to_assigned_samples', []))
    if wait_samples:
        avg_wait = float(np.mean(wait_samples))
        p95_wait = float(np.percentile(wait_samples, 95))
    else:
        avg_wait = float('nan')
        p95_wait = float('nan')

    return {
        'generated_total': generated_total,
        'completed_total': completed_total,
        'general_completion': general_completion,
        'cumulative_reward': cumulative_reward,
        'energy_total': energy_total,
        'energy_per_completed': energy_per_completed,
        'avg_wait_ready_to_assigned': avg_wait,
        'p95_wait_ready_to_assigned': p95_wait,
    }


def run_single_episode(args, order_cutoff_steps: int, seed: int) -> dict:
    """Run one episode and return completion stats.

    The environment is created with *order_cutoff_steps* (K) so that order
    generation stops K steps before business-end (Mode 1).  Delivery of
    already-accepted orders continues until the episode finishes.
    """
    env = _make_env(args, order_cutoff_steps=order_cutoff_steps)

    if _HAS_MOPSO:
        try:
            candidate_generator = MOPSOCandidateGenerator(
                candidate_k=args.candidate_k,
                n_particles=30,
                n_iterations=10,
                max_orders=200,
                max_orders_per_drone=10,
                seed=seed,
            )
            env.set_candidate_generator(candidate_generator)
        except (ImportError, Exception):
            pass  # Fall back to built-in candidates when MOPSO unavailable

    policy_fn = random_policy

    executor = DecentralizedEventDrivenExecutor(
        env=env,
        policy_fn=policy_fn,
        max_skip_steps=args.max_skip_steps,
        verbose=False,
        track_action_stats=getattr(args, 'track_action_stats', False),
    )

    executor.run_episode(max_steps=args.max_steps, seed=seed)

    stats = _compute_completion_stats(env, executor)
    stats['order_cutoff_steps'] = order_cutoff_steps
    stats['seed'] = seed

    if getattr(args, 'track_action_stats', False):
        action_stats = executor.get_action_stats()
        pct = {k: f'{v:.1f}%' for k, v in action_stats.to_percent().items()}
        print(f"  [seed={seed} K={order_cutoff_steps}] "
              f"rule_counts: {dict(action_stats.rule_counts)}  "
              f"rule_percent: {pct}  "
              f"n_decisions={action_stats.n_decisions}  "
              f"n_invalid_rule={action_stats.n_invalid_rule}  "
              f"n_empty_candidates={action_stats.n_empty_candidates}")

    if getattr(args, 'plot_paths', False) and _HAS_PLOTTING:
        gc = stats['general_completion']
        epc = stats['energy_per_completed']
        aw = stats['avg_wait_ready_to_assigned']
        plot_title = (
            f"Ablation K={order_cutoff_steps} | seed={seed} | "
            f"GC={gc:.3f}  energy_pc={epc:.3f}  wait_avg={aw:.2f}"
        )
        save_path = os.path.join(
            args.plot_dir, f"ablation_K_{order_cutoff_steps}_seed_{seed}.png"
        )
        _plot_episode_paths(
            env, save_path, title=plot_title,
            max_drones=getattr(args, 'plot_max_drones', 20),
        )
        print(f"    Plot saved: {save_path}")

    return stats


def run_multi_seed_eval(args):
    """Run multi-seed evaluation and aggregate completion statistics.

    Loads the policy once (trained model if ``--model-path`` exists, otherwise
    the built-in random policy) then runs one episode per seed, printing
    per-seed results and a final aggregate summary.

    Args:
        args: Parsed CLI arguments.  Uses ``args.seeds``, ``args.model_path``,
              ``args.vecnormalize_path``, ``args.order_cutoff_steps``,
              ``args.max_steps``, ``args.max_skip_steps``, ``args.csv_out``.
    """
    seeds = [int(s.strip()) for s in args.seeds.split(',')]

    # Load policy once, shared across all seeds
    if args.model_path and os.path.exists(args.model_path):
        policy_fn = load_trained_policy(args.model_path, args.vecnormalize_path)
        policy_name = f"Trained ({os.path.basename(args.model_path)})"
    else:
        policy_fn = random_policy
        policy_name = "Random (rule_id=3)"

    print("=" * 80)
    print("Multi-Seed Evaluation")
    print(f"  Policy:            {policy_name}")
    print(f"  Seeds ({len(seeds)}):        {seeds}")
    print(f"  order_cutoff_steps: {args.order_cutoff_steps}")
    print(f"  max_steps:          {args.max_steps}")
    print("=" * 80)

    rows = []
    fieldnames = [
        'seed', 'generated_total', 'completed_total', 'general_completion',
        'cumulative_reward', 'energy_total', 'energy_per_completed',
        'avg_wait_ready_to_assigned', 'p95_wait_ready_to_assigned',
    ]

    for seed in seeds:
        print(f"  seed={seed:>6} ...", end=' ', flush=True)

        env = _make_env(args, order_cutoff_steps=args.order_cutoff_steps)

        if _HAS_MOPSO:
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

        executor = DecentralizedEventDrivenExecutor(
            env=env,
            policy_fn=policy_fn,
            max_skip_steps=args.max_skip_steps,
            verbose=False,
            track_action_stats=getattr(args, 'track_action_stats', False),
        )
        executor.run_episode(max_steps=args.max_steps, seed=seed)

        stats = _compute_completion_stats(env, executor)
        stats['seed'] = seed
        rows.append(stats)

        print(f"GC={stats['general_completion']:.4f}  "
              f"reward={stats['cumulative_reward']:.2f}  "
              f"energy_total={stats['energy_total']:2f}"
              f"energy_pc={stats['energy_per_completed']:.3f}  "
              f"wait_avg={stats['avg_wait_ready_to_assigned']:.2f}  "
              f"generated={stats['generated_total']}  "
              f"completed={stats['completed_total']}")

        if getattr(args, 'track_action_stats', False):
            action_stats = executor.get_action_stats()
            pct = {k: f'{v:.1f}%' for k, v in action_stats.to_percent().items()}
            print(f"    rule_counts={dict(action_stats.rule_counts)}  "
                  f"rule_pct={pct}  "
                  f"n_decisions={action_stats.n_decisions}  "
                  f"n_empty={action_stats.n_empty_candidates}")

        if getattr(args, 'plot_paths', False) and _HAS_PLOTTING:
            gc = stats['general_completion']
            epc = stats['energy_per_completed']
            aw = stats['avg_wait_ready_to_assigned']
            plot_title = (
                f"{policy_name} | seed={seed} | "
                f"GC={gc:.3f}  energy_pc={epc:.3f}  wait_avg={aw:.2f}"
            )
            save_path = os.path.join(
                args.plot_dir, f"ablation_seed_{seed}.png"
            )
            _plot_episode_paths(
                env, save_path, title=plot_title,
                max_drones=getattr(args, 'plot_max_drones', 20),
            )
            print(f"    Plot saved: {save_path}")

    # Aggregate helpers
    def _agg(key):
        vals = [r[key] for r in rows if not math.isnan(r[key])]
        if not vals:
            return dict(mean=float('nan'), std=float('nan'),
                        min=float('nan'), max=float('nan'))
        return dict(mean=float(np.mean(vals)), std=float(np.std(vals)),
                    min=float(np.min(vals)), max=float(np.max(vals)))

    print("\n" + "=" * 80)
    print(f"Multi-Seed Summary  policy={policy_name}  n={len(seeds)}")
    print(f"  {'metric':<28}  {'mean':>10}  {'std':>10}  {'min':>10}  {'max':>10}")
    for label, key in [
        ('general_completion', 'general_completion'),
        ('cumulative_reward', 'cumulative_reward'),
        ('energy_per_completed', 'energy_per_completed'),
        ('avg_wait_ready_to_assigned', 'avg_wait_ready_to_assigned'),
        ('p95_wait_ready_to_assigned', 'p95_wait_ready_to_assigned'),
    ]:
        a = _agg(key)
        print(f"  {label:<28}  {a['mean']:>10.4f}  {a['std']:>10.4f}"
              f"  {a['min']:>10.4f}  {a['max']:>10.4f}")

    if args.csv_out:
        with open(args.csv_out, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow({k: row[k] for k in fieldnames})
        print(f"\nCSV written to: {args.csv_out}")

    print("=" * 80)
    return rows


def run_ablation_cutoff(args):
    """Run K-sweep ablation and write CSV."""
    cutoff_values = [int(v.strip()) for v in args.cutoff_values.split(',')]
    seeds = [int(s.strip()) for s in args.seeds.split(',')]

    print("=" * 80)
    print("Ablation: Order Cutoff Steps (K) Sweep")
    print(f"  K values: {cutoff_values}")
    print(f"  Seeds:    {seeds}")
    print("=" * 80)

    rows = []
    fieldnames = [
        'order_cutoff_steps', 'seed',
        'generated_total', 'completed_total', 'general_completion',
        'cumulative_reward', 'energy_total', 'energy_per_completed',
        'avg_wait_ready_to_assigned', 'p95_wait_ready_to_assigned',
    ]

    for K in cutoff_values:
        for seed in seeds:
            print(f"  Running K={K}, seed={seed} ...", end=' ', flush=True)
            row = run_single_episode(args, order_cutoff_steps=K, seed=seed)
            rows.append(row)
            print(f"GC={row['general_completion']:.4f}  "
                  f"reward={row['cumulative_reward']:.2f}  "
                  f"energy_pc={row['energy_per_completed']:.3f}  "
                  f"wait_avg={row['avg_wait_ready_to_assigned']:.2f}  "
                  f"generated={row['generated_total']}  completed={row['completed_total']}")

    if args.csv_out:
        with open(args.csv_out, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow({k: row[k] for k in fieldnames})
        print(f"\nCSV written to: {args.csv_out}")

    # Aggregate per K
    from collections import defaultdict

    _AGG_KEYS = [
        'general_completion', 'cumulative_reward',
        'energy_per_completed', 'avg_wait_ready_to_assigned',
    ]

    agg: dict = defaultdict(lambda: {k: [] for k in _AGG_KEYS})
    for row in rows:
        K = row['order_cutoff_steps']
        for key in _AGG_KEYS:
            val = row[key]
            if not math.isnan(val):
                agg[K][key].append(val)

    print("\n" + "=" * 80)
    print("Per-K aggregated means:")
    print(f"  {'K':>6}  {'mean_GC':>10}  {'mean_reward':>12}  "
          f"{'mean_energy_pc':>14}  {'mean_wait':>10}")
    summary = {}
    for K in cutoff_values:
        def _mean(lst):
            return float(np.mean(lst)) if lst else float('nan')

        mean_gc = _mean(agg[K]['general_completion'])
        mean_rw = _mean(agg[K]['cumulative_reward'])
        mean_ep = _mean(agg[K]['energy_per_completed'])
        mean_wt = _mean(agg[K]['avg_wait_ready_to_assigned'])
        summary[K] = {'mean_gc': mean_gc}
        print(f"  {K:>6}  "
              f"{'nan' if math.isnan(mean_gc) else f'{mean_gc:.4f}':>10}  "
              f"{'nan' if math.isnan(mean_rw) else f'{mean_rw:.2f}':>12}  "
              f"{'nan' if math.isnan(mean_ep) else f'{mean_ep:.3f}':>14}  "
              f"{'nan' if math.isnan(mean_wt) else f'{mean_wt:.2f}':>10}")

    # Determine best K by GC
    valid_gc = {K: v['mean_gc'] for K, v in summary.items() if not math.isnan(v['mean_gc'])}

    if not valid_gc:
        print("\nInsufficient data to determine best K.")
        return

    K_g, best_gc_val = min(valid_gc.items(), key=lambda kv: (-kv[1], kv[0]))

    print("\n" + "=" * 80)
    print(f"Recommended K (best mean_GC={best_gc_val:.4f}): K={K_g}")

    print("=" * 80)


def run_sanity_check(args):
    """Run sanity check with specified configuration."""
    print("=" * 80)
    print("U11 Decentralized Execution Sanity Check")
    print("=" * 80)

    # Create environment
    print("\nCreating environment...")
    env = _make_env(args, order_cutoff_steps=args.order_cutoff_steps)

    # Create MOPSO candidate generator
    if _HAS_MOPSO:
        print("Creating MOPSO candidate generator...")
        try:
            candidate_generator = MOPSOCandidateGenerator(
                candidate_k=args.candidate_k,
                n_particles=10,
                n_iterations=3,
                max_orders=100,
                max_orders_per_drone=10,
                seed=args.seed,
            )
            env.set_candidate_generator(candidate_generator)
        except (ImportError, Exception) as _mopso_err:
            print(f"  Warning: MOPSO unavailable, using built-in candidates.")

    # Choose policy
    if args.model_path:
        print(f"\nLoading trained policy from: {args.model_path}")
        policy_fn = load_trained_policy(args.model_path, args.vecnormalize_path)
        policy_name = "Trained Policy"
    else:
        print("\nUsing random policy for testing")
        policy_fn = random_policy
        policy_name = "Random Policy"

    # Create decentralized executor
    print(f"Creating decentralized executor with {policy_name}...")
    executor = DecentralizedEventDrivenExecutor(
        env=env,
        policy_fn=policy_fn,
        max_skip_steps=args.max_skip_steps,
        verbose=args.verbose,
    )

    # Run episode

    stats = executor.run_episode(max_steps=args.max_steps)


def main():
    """Parse arguments and run sanity check."""
    parser = argparse.ArgumentParser(
        description="U11 Decentralized Execution Sanity Check"
    )

    # Environment parameters
    parser.add_argument("--num-drones", type=int, default=20,
                        help="Number of drones (default: 10)")
    parser.add_argument("--obs-max-orders", type=int, default=200,
                        help="Maximum orders in observation (default: 200)")
    parser.add_argument("--top-k-merchants", type=int, default=50,
                        help="Top K merchants (default: 50)")
    parser.add_argument("--candidate-k", type=int, default=10,
                        help="Number of candidates per drone (default: 20)")
    parser.add_argument("--enable-random-events", action="store_true", default=False,
                        help="Enable random events (default: False)")

    # Executor parameters
    parser.add_argument("--max-skip-steps", type=int, default=1,
                        help="Max steps to skip when waiting for decisions (default: 10)")
    parser.add_argument("--max-steps", type=int, default=500,
                        help="Maximum decision steps per episode (default: 500)")

    # Policy parameters
    parser.add_argument("--model-path", type=str, default='ppo_u11_final.zip',
                        help="Path to trained model (.zip file) - if not provided, uses random policy")
    parser.add_argument("--vecnormalize-path", type=str, default='vecnormalize_u11_final.pkl',
                        help="Path to VecNormalize stats (.pkl file)")

    # Order cutoff parameter
    parser.add_argument("--order-cutoff-steps", type=int, default=0,
                        help="Stop accepting orders this many steps before business end (default: 0=disabled)")

    # Ablation parameters
    parser.add_argument("--ablation-cutoff", action="store_true", default=False,
                        help="Enable K-sweep ablation mode: scan environment order_cutoff_steps "
                             "(stops order generation K steps before business end)")
    parser.add_argument("--cutoff-values", type=str, default="0",
                        help="Comma-separated environment order_cutoff_steps (K) values to sweep "
                             "in ablation mode (default: 0..60)")
    parser.add_argument("--csv-out", type=str, default=None,
                        help="Output CSV path for multi-seed or ablation results")

    # Plotting parameters
    parser.add_argument("--plot-paths", action="store_true", default=False,
                        help="Save per-episode trajectory PNG plots")
    parser.add_argument("--plot-dir", type=str, default="plots",
                        help="Directory for trajectory plots (default: plots)")
    parser.add_argument("--plot-max-drones", type=int, default=20,
                        help="Maximum number of drone trajectories to render (default: 20)")

    # Other parameters
    parser.add_argument("--seed", type=int, default=21,
                        help="Random seed for single-episode mode (default: 21); "
                             "overridden when --seeds contains multiple values")
    parser.add_argument("--seeds", type=str, default='21,22,23,35,81,105,135,688,918,515',
                        help="Comma-separated seed list.  When provided in non-ablation mode "
                             "the script runs one episode per seed and prints aggregate stats.  "
                             "In ablation mode seeds are used for the K-sweep.")
    parser.add_argument("--verbose", action="store_true", default=False,
                        help="Print detailed execution logs (default: False)")
    parser.add_argument("--track-action-stats", action="store_true", default=True,
                        help="Track and print rule selection distribution after each episode "
                             "(default: False)")

    args = parser.parse_args()

    # Resolve seeds: --seeds overrides --seed; default to single value of --seed
    if args.seeds is None:
        args.seeds = str(args.seed)

    if args.ablation_cutoff:
        try:
            run_ablation_cutoff(args)
        except Exception as e:
            print("\n" + "=" * 80)
            print("Ablation FAILED ✗")
            print("=" * 80)
            print(f"\nError: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    else:
        # Multi-seed mode when multiple seeds are given; single-episode otherwise
        seeds = [int(s.strip()) for s in args.seeds.split(',')]
        run_fn = run_multi_seed_eval if len(seeds) > 1 else run_sanity_check
        if len(seeds) == 1:
            args.seed = seeds[0]  # keep args.seed in sync for run_sanity_check
        try:
            run_fn(args)
        except Exception as e:
            print("\n" + "=" * 80)
            print(f"{'Multi-Seed Eval' if len(seeds) > 1 else 'Sanity Check'} FAILED ✗")
            print("=" * 80)
            print(f"\nError: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)


if __name__ == "__main__":
    main()