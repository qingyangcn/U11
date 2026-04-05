from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from config.default import AppConfig
from core.entities import Drone, Order, Station, WeatherSnapshot
from env.constraints import TaskProjection
from swarm.base import BaseSwarmScheduler
from swarm.common import SwarmPlan, build_problem, decode_plan, plan_matches_projection


@dataclass(slots=True)
class _ParticleState:
    position: np.ndarray
    velocity: np.ndarray
    best_position: np.ndarray
    best_cost: float
    best_plan: SwarmPlan


class PurePSOScheduler(BaseSwarmScheduler):
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

        particles = max(int(self.cfg.dispatch.particles), 4)
        iterations = max(int(self.cfg.dispatch.iterations), 1)
        swarm: list[_ParticleState] = []
        for _ in range(particles):
            position = self.rng.random(dim)
            velocity = self.rng.uniform(-0.15, 0.15, size=dim)
            plan = decode_plan(position, problem, self.cfg)
            swarm.append(
                _ParticleState(
                    position=position,
                    velocity=velocity,
                    best_position=position.copy(),
                    best_cost=plan.evaluation.objective_cost,
                    best_plan=plan,
                )
            )

        global_best = min(swarm, key=lambda particle: particle.best_cost)
        global_best_position = global_best.best_position.copy()
        global_best_cost = float(global_best.best_cost)
        global_best_plan = global_best.best_plan

        for _ in range(iterations):
            for particle in swarm:
                r1 = self.rng.random(dim)
                r2 = self.rng.random(dim)
                particle.velocity = (
                    self.cfg.dispatch.inertia * particle.velocity
                    + self.cfg.dispatch.c1 * r1 * (particle.best_position - particle.position)
                    + self.cfg.dispatch.c2 * r2 * (global_best_position - particle.position)
                )
                particle.position = np.clip(particle.position + particle.velocity, 0.0, 1.0)
                plan = decode_plan(particle.position, problem, self.cfg)
                cost = plan.evaluation.objective_cost
                if cost < particle.best_cost:
                    particle.best_cost = float(cost)
                    particle.best_position = particle.position.copy()
                    particle.best_plan = plan
                if cost < global_best_cost:
                    global_best_cost = float(cost)
                    global_best_position = particle.position.copy()
                    global_best_plan = plan

        self.current_plan = global_best_plan
        return global_best_plan

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
