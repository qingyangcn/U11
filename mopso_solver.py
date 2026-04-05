from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np

from allocator.base import BaseAllocator
from allocator.decoder import DecodedAllocation, decode_particle
from allocator.objectives import evaluate_solution
from config.default import AppConfig, build_default_config
from core.entities import Drone, Order, Station, WeatherSnapshot
from core.enums import OrderStatus
from swarm.common import project_drone_future_state


@dataclass(slots=True)
class ArchiveEntry:
    particle: np.ndarray
    decoded: DecodedAllocation
    objectives: tuple[float, float, float]


def _dominates(a: tuple[float, float, float], b: tuple[float, float, float]) -> bool:
    return all(x <= y for x, y in zip(a, b)) and any(x < y for x, y in zip(a, b))


class MOPSOSolver(BaseAllocator):
    def __init__(self, cfg: AppConfig | None = None, seed: int | None = None) -> None:
        self.cfg = cfg or build_default_config()
        self.rng = np.random.default_rng(seed if seed is not None else self.cfg.training.seed)

    def _update_archive(self, archive: list[ArchiveEntry], candidate: ArchiveEntry) -> list[ArchiveEntry]:
        filtered = []
        keep_candidate = True
        for entry in archive:
            if _dominates(entry.objectives, candidate.objectives):
                keep_candidate = False
                filtered.append(entry)
            elif _dominates(candidate.objectives, entry.objectives):
                continue
            else:
                filtered.append(entry)
        if keep_candidate:
            filtered.append(candidate)
        return filtered[: self.cfg.dispatch.archive_limit]

    def _choose_archive_entry(self, archive: list[ArchiveEntry]) -> ArchiveEntry:
        if len(archive) == 1:
            return archive[0]
        objective_matrix = np.asarray([entry.objectives for entry in archive], dtype=float)
        mins = objective_matrix.min(axis=0)
        maxs = objective_matrix.max(axis=0)
        weights = np.asarray(
            [
                self.cfg.dispatch.weight_efficiency,
                self.cfg.dispatch.weight_risk,
                self.cfg.dispatch.weight_balance,
            ],
            dtype=float,
        )
        best_idx = 0
        best_score = float("inf")
        for idx, row in enumerate(objective_matrix):
            normalized = (row - mins) / (maxs - mins + 1e-8)
            score = float(np.max(weights * normalized))
            if score < best_score:
                best_score = score
                best_idx = idx
        return archive[best_idx]

    def assign(
        self,
        current_time: float,
        drones: Iterable[Drone],
        orders: Iterable[Order],
        stations: Iterable[Station],
        weather: WeatherSnapshot,
    ) -> Dict[str, list[str]]:
        drones = list(drones)
        orders = list(orders)
        stations = list(stations)
        if not drones:
            return {}
        order_lookup = {order.order_id: order for order in orders}
        candidate_orders = [order for order in orders if order.status == OrderStatus.READY_UNASSIGNED]
        if not candidate_orders:
            return {drone.drone_id: [] for drone in drones}
        future_states = {
            drone.drone_id: project_drone_future_state(current_time, drone, order_lookup, stations, weather, self.cfg)
            for drone in drones
        }
        initial_available_time = {
            drone_id: state.available_time for drone_id, state in future_states.items()
        }
        initial_anchor = {
            drone_id: state.position for drone_id, state in future_states.items()
        }
        initial_payload = {
            drone_id: float(sum(order_lookup[oid].quantity_kg for oid in state.carried_order_ids if oid in order_lookup))
            for drone_id, state in future_states.items()
        }
        initial_battery = {
            drone_id: state.battery_current for drone_id, state in future_states.items()
        }

        n_orders = len(candidate_orders)
        n_particles = self.cfg.dispatch.particles
        positions = self.rng.uniform(0.0, 1.0, size=(n_particles, n_orders, 2))
        velocities = self.rng.normal(0.0, 0.1, size=(n_particles, n_orders, 2))

        pbest_positions = positions.copy()
        pbest_objectives: list[tuple[float, float, float]] = []
        archive: list[ArchiveEntry] = []

        for particle in positions:
            decoded = decode_particle(
                particle,
                current_time,
                drones,
                candidate_orders,
                order_lookup,
                stations,
                weather,
                self.cfg,
                available_time_by_drone=initial_available_time,
                anchor_by_drone=initial_anchor,
                payload_by_drone=initial_payload,
                battery_by_drone=initial_battery,
            )
            objectives = evaluate_solution(current_time, decoded.assignments, decoded.estimates, drones)
            pbest_objectives.append(objectives)
            archive = self._update_archive(
                archive,
                ArchiveEntry(particle=particle.copy(), decoded=decoded, objectives=objectives),
            )

        for _ in range(self.cfg.dispatch.iterations):
            guide = self._choose_archive_entry(archive).particle
            for idx in range(n_particles):
                r1 = self.rng.random(size=(n_orders, 2))
                r2 = self.rng.random(size=(n_orders, 2))
                velocities[idx] = (
                    self.cfg.dispatch.inertia * velocities[idx]
                    + self.cfg.dispatch.c1 * r1 * (pbest_positions[idx] - positions[idx])
                    + self.cfg.dispatch.c2 * r2 * (guide - positions[idx])
                )
                positions[idx] = np.clip(positions[idx] + velocities[idx], 0.0, 1.0)
                decoded = decode_particle(
                    positions[idx],
                    current_time,
                    drones,
                    candidate_orders,
                    order_lookup,
                    stations,
                    weather,
                    self.cfg,
                    available_time_by_drone=initial_available_time,
                    anchor_by_drone=initial_anchor,
                    payload_by_drone=initial_payload,
                    battery_by_drone=initial_battery,
                )
                objectives = evaluate_solution(current_time, decoded.assignments, decoded.estimates, drones)
                if _dominates(objectives, pbest_objectives[idx]) or (
                    not _dominates(pbest_objectives[idx], objectives) and float(np.sum(objectives)) < float(np.sum(pbest_objectives[idx]))
                ):
                    pbest_positions[idx] = positions[idx].copy()
                    pbest_objectives[idx] = objectives
                archive = self._update_archive(
                    archive,
                    ArchiveEntry(particle=positions[idx].copy(), decoded=decoded, objectives=objectives),
                )

        best = self._choose_archive_entry(archive)
        return best.decoded.assignments
