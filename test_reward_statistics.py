"""
回归测试：验证奖励/统计口径一致性修复

覆盖的测试点：
1. episode_r_vec 每步仅累计一次（不再重复累计）
2. scalar_reward 与 r_vec 在 scalar 模式下关系正确
3. 终止步 final_bonus 只计入 episode_r_vec 一次
"""

import contextlib
import io
import numpy as np
import unittest




class TestEpisodeRVecSingleAccumulation(unittest.TestCase):
    """episode_r_vec 每步仅累计一次"""

    def test_episode_r_vec_matches_manual_sum(self):
        """手动累加每步 r_vec 应与 episode_r_vec 完全一致"""
        from UAV_ENVIRONMENT_11 import ThreeObjectiveDroneDeliveryEnv
        env = ThreeObjectiveDroneDeliveryEnv(
            grid_size=4,
            num_drones=2,
            max_orders=10,
            reward_output_mode="scalar",
            enable_random_events=False,
            enable_sigmoid_hazard_cancellation=False,
            enable_diagnostics=False,
        )
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            env.reset(seed=42)

        manual_ep_r_vec = np.zeros(env.num_objectives, dtype=np.float32)
        for _ in range(20):
            action = env.action_space.sample()
            with contextlib.redirect_stdout(buf):
                obs, reward, terminated, truncated, info = env.step(action)

            manual_ep_r_vec = manual_ep_r_vec + info['r_vec'].astype(np.float32)
            np.testing.assert_allclose(
                info['episode_r_vec'],
                manual_ep_r_vec,
                rtol=1e-5,
                atol=1e-6,
                err_msg=(
                    f"episode_r_vec mismatch at step: "
                    f"env={info['episode_r_vec']}, manual={manual_ep_r_vec}"
                ),
            )
            if terminated or truncated:
                break


class TestScalarRewardConsistency(unittest.TestCase):
    """scalar_reward 在 scalar 模式下应等于 dot(weights, r_vec)"""

    def test_scalar_reward_equals_dot_product(self):
        from UAV_ENVIRONMENT_11 import ThreeObjectiveDroneDeliveryEnv
        env = ThreeObjectiveDroneDeliveryEnv(
            grid_size=4,
            num_drones=2,
            max_orders=10,
            reward_output_mode="scalar",
            enable_random_events=False,
            enable_sigmoid_hazard_cancellation=False,
            enable_diagnostics=False,
        )
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            env.reset(seed=7)

        for _ in range(10):
            action = env.action_space.sample()
            with contextlib.redirect_stdout(buf):
                obs, reward, terminated, truncated, info = env.step(action)

            expected = float(np.dot(info['objective_weights'], info['r_vec']))
            self.assertAlmostEqual(
                info['scalar_reward'],
                expected,
                places=5,
                msg=f"scalar_reward {info['scalar_reward']} != dot(w,r_vec) {expected}",
            )
            self.assertAlmostEqual(
                reward,
                expected,
                places=5,
                msg=f"step() reward {reward} != dot(w,r_vec) {expected}",
            )
            if terminated or truncated:
                break

    def test_obj0_mode(self):
        """obj0 模式下 scalar_reward 应等于 r_vec[0]"""
        from UAV_ENVIRONMENT_11 import ThreeObjectiveDroneDeliveryEnv
        env = ThreeObjectiveDroneDeliveryEnv(
            grid_size=4,
            num_drones=2,
            max_orders=10,
            reward_output_mode="obj0",
            enable_random_events=False,
            enable_sigmoid_hazard_cancellation=False,
            enable_diagnostics=False,
        )
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            env.reset(seed=3)

        for _ in range(10):
            action = env.action_space.sample()
            with contextlib.redirect_stdout(buf):
                obs, reward, terminated, truncated, info = env.step(action)

            self.assertAlmostEqual(
                info['scalar_reward'],
                float(info['r_vec'][0]),
                places=5,
                msg=f"obj0 scalar_reward {info['scalar_reward']} != r_vec[0] {info['r_vec'][0]}",
            )
            if terminated or truncated:
                break

    def test_zero_mode(self):
        """zero 模式下 scalar_reward 应始终为 0"""
        from UAV_ENVIRONMENT_11 import ThreeObjectiveDroneDeliveryEnv
        env = ThreeObjectiveDroneDeliveryEnv(
            grid_size=4,
            num_drones=2,
            max_orders=10,
            reward_output_mode="zero",
            enable_random_events=False,
            enable_sigmoid_hazard_cancellation=False,
            enable_diagnostics=False,
        )
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            env.reset(seed=5)

        for _ in range(10):
            action = env.action_space.sample()
            with contextlib.redirect_stdout(buf):
                obs, reward, terminated, truncated, info = env.step(action)

            self.assertEqual(reward, 0.0, msg="zero mode should always return 0")
            if terminated or truncated:
                break


class TestFinalBonusSingleAccumulation(unittest.TestCase):
    """终止步 final_bonus 只计入 episode_r_vec 一次"""

    def _run_full_episode(self, reward_output_mode="scalar", seed=0, max_steps=500):
        """运行完整 episode，返回最后一步 info"""
        from UAV_ENVIRONMENT_11 import ThreeObjectiveDroneDeliveryEnv
        env = ThreeObjectiveDroneDeliveryEnv(
            grid_size=4,
            num_drones=2,
            max_orders=10,
            reward_output_mode=reward_output_mode,
            enable_random_events=False,
            enable_sigmoid_hazard_cancellation=False,
            enable_diagnostics=False,
        )
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            env.reset(seed=seed)

        manual_ep_r_vec = np.zeros(env.num_objectives, dtype=np.float32)
        final_info = None
        for _ in range(max_steps):
            action = env.action_space.sample()
            with contextlib.redirect_stdout(buf):
                obs, reward, terminated, truncated, info = env.step(action)
            manual_ep_r_vec = manual_ep_r_vec + info['r_vec'].astype(np.float32)
            if terminated or truncated:
                final_info = info
                final_manual = manual_ep_r_vec.copy()
                break
        return final_info, final_manual

    def test_terminal_episode_r_vec_includes_bonus_once(self):
        """终止 episode 的 episode_r_vec 应与手动逐步累加一致（含 final_bonus 一次）"""
        final_info, final_manual = self._run_full_episode(seed=99)
        if final_info is None:
            self.skipTest("Episode did not terminate within max_steps")

        np.testing.assert_allclose(
            final_info['episode_r_vec'],
            final_manual,
            rtol=1e-5,
            atol=1e-6,
            err_msg=(
                "On terminal step, episode_r_vec mismatch. "
                f"env={final_info['episode_r_vec']}, manual={final_manual}. "
                "final_bonus may have been double-counted."
            ),
        )

    def test_episode_r_vec_consistent_across_reset(self):
        """reset 后 episode_r_vec 应清零"""
        from UAV_ENVIRONMENT_11 import ThreeObjectiveDroneDeliveryEnv
        env = ThreeObjectiveDroneDeliveryEnv(
            grid_size=4,
            num_drones=2,
            max_orders=10,
            reward_output_mode="scalar",
            enable_random_events=False,
            enable_sigmoid_hazard_cancellation=False,
            enable_diagnostics=False,
        )
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            env.reset(seed=0)
        # Run a few steps
        for _ in range(5):
            action = env.action_space.sample()
            with contextlib.redirect_stdout(buf):
                env.step(action)
        # Reset and verify episode_r_vec is zeroed
        with contextlib.redirect_stdout(buf):
            env.reset(seed=1)
        np.testing.assert_array_equal(
            env.episode_r_vec,
            np.zeros(env.num_objectives, dtype=np.float32),
            err_msg="episode_r_vec should be zero after reset",
        )


if __name__ == "__main__":
    # 运行时抑制冗长的环境输出
    unittest.main(verbosity=2)
