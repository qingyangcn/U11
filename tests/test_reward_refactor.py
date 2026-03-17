"""
最小回归测试：验证奖励函数解耦重构（研究导向）

断言：
  1. Obj0 不包含 progress shaping（obj0_progress_shaping 始终为 0）。
  2. 取消惩罚仅在 Obj2 中生效，Obj0 的 cancelled 分项始终为 0。
  3. backlog 惩罚使用 log(1+backlog) 而非线性。
"""

import math
import sys
import os

# Allow importing from repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
from UAV_ENVIRONMENT_11 import ThreeObjectiveDroneDeliveryEnv as UAVDeliveryEnv


def make_env(**kwargs):
    """Create a minimal environment instance for reward tests."""
    defaults = dict(
        grid_size=10,
        num_drones=2,
        max_orders=20,
        high_load_factor=1.5,
        enable_diagnostics=False,
    )
    defaults.update(kwargs)
    return UAVDeliveryEnv(**defaults)


# ---------------------------------------------------------------------------
# Assertion 1: Obj0 progress_shaping is always 0
# ---------------------------------------------------------------------------

def test_obj0_progress_shaping_is_zero():
    """After several steps, obj0_progress_shaping must always be 0."""
    env = make_env()
    env.reset(seed=42)
    action = np.zeros(env.num_drones, dtype=np.int32)

    for _ in range(20):
        env.step(action)
        ps = env.last_step_reward_components['obj0_progress_shaping']
        assert ps == 0.0, (
            f"obj0_progress_shaping should be 0 (progress shaping removed from Obj0), "
            f"got {ps}"
        )


# ---------------------------------------------------------------------------
# Assertion 2: Cancelled penalty in Obj0 is always 0
# ---------------------------------------------------------------------------

def test_obj0_cancelled_is_zero():
    """obj0_cancelled must always be 0 (cancel penalty moved to Obj2 only)."""
    env = make_env(high_load_factor=2.5)  # High load to trigger cancellations
    env.reset(seed=0)
    action = np.zeros(env.num_drones, dtype=np.int32)

    for _ in range(50):
        env.step(action)
        obj0_cancel = env.last_step_reward_components['obj0_cancelled']
        assert obj0_cancel == 0.0, (
            f"obj0_cancelled should be 0 (cancel penalty is Obj2-only), "
            f"got {obj0_cancel}"
        )


# ---------------------------------------------------------------------------
# Assertion 3: Backlog penalty uses log(1+backlog) not linear
# ---------------------------------------------------------------------------

def test_backlog_penalty_is_log_based():
    """
    Verify that obj2_backlog = -reward_f * log(1+backlog).
    We inject known backlog sizes via active_orders and call the reward function
    directly (using _calculate_three_objective_rewards).
    """
    env = make_env()
    env.reset(seed=7)

    # Patch last_stats to prevent delta confusion: set to current so deltas = 0
    def _zero_deltas():
        env.last_stats['completed'] = env.daily_stats['orders_completed']
        env.last_stats['energy'] = env.daily_stats['energy_consumed']
        env.last_stats['on_time'] = env.daily_stats['on_time_deliveries']
        env.last_stats['cancelled'] = env.daily_stats['orders_cancelled']
        env.last_stats['distance'] = env.daily_stats['total_flight_distance']

    for backlog_size in [0, 5, 20, 100]:
        # Save original active_orders (it's a set of order ids)
        original_orders = set(env.active_orders)
        # Replace with dummy negative ids (won't collide with real orders)
        env.active_orders.clear()
        for i in range(backlog_size):
            env.active_orders.add(-(i + 1))  # dummy order ids

        _zero_deltas()
        rewards = env._calculate_three_objective_rewards()

        expected_penalty = env.reward_f * math.log(1.0 + backlog_size)
        actual_penalty = -env.last_step_reward_components['obj2_backlog']  # stored as negative

        assert abs(actual_penalty - expected_penalty) < 1e-5, (
            f"backlog={backlog_size}: expected log-penalty={expected_penalty:.5f}, "
            f"got {actual_penalty:.5f}"
        )

        # Verify it is NOT linear (except backlog=0 which is the same)
        if backlog_size > 0:
            linear_penalty = env.reward_f * backlog_size
            assert actual_penalty != pytest.approx(linear_penalty, rel=0.01), (
                f"backlog={backlog_size}: penalty looks linear ({actual_penalty:.5f} ≈ {linear_penalty:.5f}); "
                f"expected log-based"
            )

        # Restore
        env.active_orders.clear()
        env.active_orders.update(original_orders)


# ---------------------------------------------------------------------------
# Assertion 4: Obj0 total equals only completed bonus (no shaping, no cancel)
# ---------------------------------------------------------------------------

def test_obj0_total_equals_completed_only():
    """obj0_total should equal obj0_completed when there's no other Obj0 component."""
    env = make_env()
    env.reset(seed=42)
    action = np.zeros(env.num_drones, dtype=np.int32)

    for _ in range(30):
        obs, r_vec, terminated, truncated, info = env.step(action)
        rc = env.last_step_reward_components
        # Obj0 total should equal completed bonus only (shaping and cancel are 0)
        assert rc['obj0_total'] == pytest.approx(rc['obj0_completed'], abs=1e-4), (
            f"Obj0 total {rc['obj0_total']} should equal completed bonus "
            f"{rc['obj0_completed']} (no other Obj0 components)"
        )
        if terminated:
            env.reset(seed=42)


# ---------------------------------------------------------------------------
# Quick smoke test: one full episode runs without errors
# ---------------------------------------------------------------------------

def test_full_episode_smoke():
    """One full episode should run to completion without exceptions."""
    env = make_env()
    env.reset(seed=1)
    action = np.zeros(env.num_drones, dtype=np.int32)

    terminated = truncated = False
    steps = 0
    while not (terminated or truncated) and steps < 5000:
        _, _, terminated, truncated, _ = env.step(action)
        steps += 1

    assert steps > 0, "Episode ended immediately"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
