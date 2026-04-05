from __future__ import annotations

import numpy as np

from config.default import AppConfig
from core.entities import Drone, Order, Station, WeatherSnapshot
from env.constraints import TaskProjection
from swarm.base import BaseSwarmScheduler
from swarm.common import SwarmPlan, build_problem, decode_plan, plan_matches_projection


class PureGWOScheduler(BaseSwarmScheduler):
    def __init__(self, cfg: AppConfig, seed: int | None = None) -> None:
        self.cfg = cfg
        self.base_seed = seed
        self.rng = np.random.default_rng(seed)
        self.current_plan: SwarmPlan | None = None

    def reset(self, seed: int | None = None) -> None:
        actual_seed = self.base_seed if seed is None else seed
        self.rng = np.random.default_rng(actual_seed)
        self.current_plan = None

    def replan(
        self,
        current_time: float,
        drones: dict[str, Drone],
        orders: dict[str, Order],
        stations: list[Station],
        weather: WeatherSnapshot,
        allow_new_assignments: bool,
    ) -> SwarmPlan:
        problem = build_problem(
            current_time=current_time,
            drones=drones,
            orders=orders,
            stations=stations,
            weather=weather,
            cfg=self.cfg,
            allow_new_assignments=allow_new_assignments,
        )
        dim = len(problem.optim_order_ids) * 3
        if dim == 0:
            self.current_plan = decode_plan(np.zeros(0, dtype=float), problem, self.cfg)
            return self.current_plan

        wolves = max(int(self.cfg.dispatch.particles), 4)
        iterations = max(int(self.cfg.dispatch.iterations), 1)
        positions = self.rng.random((wolves, dim))
        costs = np.zeros(wolves, dtype=float)
        plans: list[SwarmPlan] = []
        for idx in range(wolves):
            plan = decode_plan(positions[idx], problem, self.cfg)
            plans.append(plan)
            costs[idx] = plan.evaluation.objective_cost

        best_plan = plans[int(np.argmin(costs))]
        for iteration in range(iterations):
            order = np.argsort(costs)
            alpha = positions[order[0]].copy()
            beta = positions[order[1]].copy()
            delta = positions[order[2]].copy()
            best_plan = plans[int(order[0])]
            a = 2.0 - 2.0 * (iteration / max(iterations, 1))
            for idx in range(wolves):
                r1 = self.rng.random(dim)
                r2 = self.rng.random(dim)
                a1 = 2.0 * a * r1 - a
                c1 = 2.0 * r2
                x1 = alpha - a1 * np.abs(c1 * alpha - positions[idx])

                r1 = self.rng.random(dim)
                r2 = self.rng.random(dim)
                a2 = 2.0 * a * r1 - a
                c2 = 2.0 * r2
                x2 = beta - a2 * np.abs(c2 * beta - positions[idx])

                r1 = self.rng.random(dim)
                r2 = self.rng.random(dim)
                a3 = 2.0 * a * r1 - a
                c3 = 2.0 * r2
                x3 = delta - a3 * np.abs(c3 * delta - positions[idx])

                positions[idx] = np.clip((x1 + x2 + x3) / 3.0, 0.0, 1.0)
                plan = decode_plan(positions[idx], problem, self.cfg)
                plans[idx] = plan
                costs[idx] = plan.evaluation.objective_cost

        best_idx = int(np.argmin(costs))
        self.current_plan = plans[best_idx]
        return self.current_plan

    def next_projection(
        self,
        drone_id: str,
        feasible_projections: dict[str, TaskProjection],
        orders: dict[str, Order],
    ) -> TaskProjection | None:
        del orders
        if self.current_plan is None:
            return None
        for task in self.current_plan.task_sequences.get(drone_id, []):
            projection = feasible_projections.get(task.order_id)
            if projection is not None and plan_matches_projection(task, projection):
                return projection
        return None
