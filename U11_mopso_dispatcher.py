"""
U7 MOPSO Dispatcher – Multi-Objective PSO Order-to-Drone Batch Assignment

This module implements the TASK LAYER of the hierarchical UAV delivery system.
Responsibility: order → drone assignment (READY → ASSIGNED, multiple orders per drone).

Architecture:
    MOPSO (this module) : order → drone assignment (task layer)
    RL                  : path / control for already-assigned tasks (motion layer)
    Environment         : state advancement & constraint checking only

The core class ``MOPSOPlanner`` exposes a single public method::

    assignment = planner._run_mopso(
        ready_orders, drones, merchants, constraints, objective_weights
    )

which returns ``Dict[int, List[int]]`` – drone_id → ordered list of order_ids.

Algorithm
---------
A greedy multi-objective scoring approach is used that:
1. Scores every (drone, order) pair on three objectives (urgency, proximity, route distance).
2. Optionally runs n_iterations rounds of PSO perturbation to escape local optima
   (controlled by ``n_iterations``; set to 0 for pure greedy).
3. Greedily assigns the highest-scoring pairs first, respecting per-drone capacity
   and ``max_orders_per_drone``.

The three objectives align with the environment's reward structure:
    Objective 0 – Delivery timeliness  (maximize slack / deadline proximity)
    Objective 1 – Pickup proximity     (minimize drone→merchant distance)
    Objective 2 – Total route distance (minimize merchant→customer distance)
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Optional, Any


class MOPSOPlanner:
    """
    Multi-Objective PSO planner for order-to-drone batch assignment.

    Parameters
    ----------
    n_particles : int
        Number of PSO particles (population size for perturbation phase).
        Set to 1 to disable perturbation (pure greedy).
    n_iterations : int
        Number of PSO iterations.  Set to 0 for pure greedy assignment.
    max_orders : int
        Maximum number of READY orders passed to the optimiser (others are ignored).
    max_orders_per_drone : int
        Hard cap on how many orders a single drone can be batch-assigned.
    seed : int or None
        Random seed for reproducibility.
    **kwargs
        Ignored – accepted for forward-compatibility with caller code that may
        pass extra keyword arguments (e.g. ``eta_speed_scale_assumption``).
    """

    def __init__(
        self,
        n_particles: int = 30,
        n_iterations: int = 10,
        max_orders: int = 200,
        max_orders_per_drone: int = 10,
        seed: Optional[int] = None,
        **kwargs,
    ) -> None:
        self.n_particles = max(1, n_particles)
        self.n_iterations = max(0, n_iterations)
        self.max_orders = max_orders
        self.max_orders_per_drone = max_orders_per_drone
        self.rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def _run_mopso(
        self,
        ready_orders: List[dict],
        drones: List[dict],
        merchants: Dict[str, dict],
        constraints: dict,
        objective_weights: np.ndarray,
    ) -> Dict[int, List[int]]:
        """
        Run MOPSO to produce a batch order-to-drone assignment.

        Parameters
        ----------
        ready_orders :
            Snapshots of READY, unassigned orders.  Each dict contains at least:
            ``order_id``, ``merchant_location``, ``customer_location``,
            ``deadline_step``, ``urgent``, ``distance``.
        drones :
            Snapshots of all drones.  Each dict contains at least:
            ``drone_id``, ``location``, ``current_load``, ``max_capacity``,
            ``speed``.
        merchants :
            Dict mapping merchant_id → merchant snapshot (unused here but kept
            for API compatibility).
        constraints :
            Dict with ``current_step`` and other env parameters.
        objective_weights :
            1-D array of length 3 summing to 1; weights for the three objectives.

        Returns
        -------
        Dict[int, List[int]]
            Mapping drone_id → list of order_ids (in priority order, highest first).
            Drones with no assignments map to an empty list.
        """
        if not ready_orders or not drones:
            return {d["drone_id"]: [] for d in drones}

        orders = ready_orders[: self.max_orders]
        current_step = int(constraints.get("current_step", 0))

        # Normalise weights
        w = np.asarray(objective_weights, dtype=np.float64)
        if w.sum() > 0:
            w = w / w.sum()
        else:
            w = np.array([1 / 3, 1 / 3, 1 / 3], dtype=np.float64)

        # Build score matrix (n_drones × n_orders); -inf = infeasible pair
        scores = self._build_score_matrix(orders, drones, w, current_step)

        # Greedy best assignment (used as baseline and final result)
        base_assignment = self._greedy_assign(scores, orders, drones)

        if self.n_iterations <= 0 or self.n_particles <= 1:
            return base_assignment

        # PSO perturbation: try random swaps / reallocations to improve total score
        best_assignment = base_assignment
        best_total = self._score_assignment(best_assignment, scores, orders, drones)

        for _ in range(self.n_iterations):
            candidate = self._perturb_assignment(
                best_assignment, scores, orders, drones
            )
            candidate_total = self._score_assignment(candidate, scores, orders, drones)
            if candidate_total > best_total:
                best_assignment = candidate
                best_total = candidate_total

        return best_assignment

    # ------------------------------------------------------------------
    # Score matrix construction
    # ------------------------------------------------------------------

    def _build_score_matrix(
        self,
        orders: List[dict],
        drones: List[dict],
        weights: np.ndarray,
        current_step: int,
    ) -> np.ndarray:
        """Return score matrix (n_drones × n_orders); higher = better assignment."""
        n_drones = len(drones)
        n_orders = len(orders)
        scores = np.full((n_drones, n_orders), -np.inf, dtype=np.float64)

        # Pre-compute order arrays once
        merchant_locs = np.array(
            [o["merchant_location"] for o in orders], dtype=np.float64
        )  # (n_orders, 2)
        customer_locs = np.array(
            [o["customer_location"] for o in orders], dtype=np.float64
        )  # (n_orders, 2)
        deadlines = np.array([o["deadline_step"] for o in orders], dtype=np.float64)
        urgents = np.array([int(o.get("urgent", False)) for o in orders], dtype=np.float64)

        # Merchant→customer leg distance (shared across drones)
        mc_diff = customer_locs - merchant_locs  # (n_orders, 2)
        mc_dist = np.hypot(mc_diff[:, 0], mc_diff[:, 1])  # (n_orders,)

        # Time slack (steps until deadline)
        slacks = deadlines - current_step  # (n_orders,)

        for di, drone in enumerate(drones):
            remaining_capacity = drone["max_capacity"] - drone["current_load"]
            if remaining_capacity <= 0:
                continue  # Drone at capacity – leave all scores as -inf

            drone_loc = np.array(drone["location"], dtype=np.float64)

            # Drone→merchant pickup distances
            dm_diff = merchant_locs - drone_loc  # (n_orders, 2)
            dm_dist = np.hypot(dm_diff[:, 0], dm_diff[:, 1])  # (n_orders,)

            # Feasibility mask: only orders with positive slack
            feasible = slacks > 0  # (n_orders,)

            if not feasible.any():
                continue

            # --- Objective 0: Timeliness (urgency-weighted slack score) ---
            # Higher slack relative to deadline = better; urgent orders boosted
            max_deadline = deadlines.max() if deadlines.max() > current_step else current_step + 1
            normalised_slack = np.clip(slacks / (max_deadline - current_step), 0.0, 1.0)
            obj0 = normalised_slack + 0.3 * urgents  # urgent bonus

            # --- Objective 1: Pickup proximity ---
            max_dm = dm_dist.max() if dm_dist.max() > 0 else 1.0
            obj1 = 1.0 - dm_dist / max_dm  # closer = higher score

            # --- Objective 2: Route efficiency (inverse total distance) ---
            total_dist = dm_dist + mc_dist
            max_total = total_dist.max() if total_dist.max() > 0 else 1.0
            obj2 = 1.0 - total_dist / max_total

            combined = weights[0] * obj0 + weights[1] * obj1 + weights[2] * obj2

            # Set infeasible pairs to -inf
            combined[~feasible] = -np.inf

            scores[di] = combined

        return scores

    # ------------------------------------------------------------------
    # Greedy assignment
    # ------------------------------------------------------------------

    def _greedy_assign(
        self,
        scores: np.ndarray,
        orders: List[dict],
        drones: List[dict],
    ) -> Dict[int, List[int]]:
        """
        Build assignment by greedily taking the best (drone, order) pair at each step.
        """
        n_orders = len(orders)
        assignment: Dict[int, List[int]] = {d["drone_id"]: [] for d in drones}
        drone_loads = {d["drone_id"]: d["current_load"] for d in drones}
        drone_max = {d["drone_id"]: d["max_capacity"] for d in drones}
        drone_idx = {d["drone_id"]: di for di, d in enumerate(drones)}

        order_assigned = np.zeros(n_orders, dtype=bool)

        # Build flat list of (score, di, oi, drone_id) and sort descending
        flat: List[tuple] = []
        for di, drone in enumerate(drones):
            drone_id = drone["drone_id"]
            for oi in range(n_orders):
                s = scores[di, oi]
                if s > -np.inf:
                    flat.append((s, di, oi, drone_id))
        flat.sort(key=lambda x: x[0], reverse=True)

        for s, di, oi, drone_id in flat:
            if order_assigned[oi]:
                continue
            if drone_loads[drone_id] >= drone_max[drone_id]:
                continue
            if len(assignment[drone_id]) >= self.max_orders_per_drone:
                continue
            assignment[drone_id].append(orders[oi]["order_id"])
            drone_loads[drone_id] += 1
            order_assigned[oi] = True

        return assignment

    # ------------------------------------------------------------------
    # PSO helpers
    # ------------------------------------------------------------------

    def _score_assignment(
        self,
        assignment: Dict[int, List[int]],
        scores: np.ndarray,
        orders: List[dict],
        drones: List[dict],
    ) -> float:
        """Sum of scores for all assigned (drone, order) pairs."""
        order_id_to_idx = {o["order_id"]: oi for oi, o in enumerate(orders)}
        drone_id_to_idx = {d["drone_id"]: di for di, d in enumerate(drones)}
        total = 0.0
        for drone_id, order_ids in assignment.items():
            di = drone_id_to_idx.get(drone_id)
            if di is None:
                continue
            for oid in order_ids:
                oi = order_id_to_idx.get(oid)
                if oi is None:
                    continue
                total += scores[di, oi]
        return total

    def _perturb_assignment(
        self,
        assignment: Dict[int, List[int]],
        scores: np.ndarray,
        orders: List[dict],
        drones: List[dict],
    ) -> Dict[int, List[int]]:
        """
        Produce a perturbed neighbour of ``assignment`` by swapping one order
        between two randomly chosen drones (or re-allocating an unassigned order).
        """
        import copy

        new_assign: Dict[int, List[int]] = {k: list(v) for k, v in assignment.items()}
        drone_ids = [d["drone_id"] for d in drones]
        if len(drone_ids) < 2:
            return new_assign

        # Pick two distinct drones
        d1, d2 = self.rng.choice(len(drone_ids), size=2, replace=False)
        id1, id2 = drone_ids[d1], drone_ids[d2]

        list1 = new_assign[id1]
        list2 = new_assign[id2]

        # Try swapping one order between them if both non-empty
        if list1 and list2:
            i1 = int(self.rng.integers(len(list1)))
            i2 = int(self.rng.integers(len(list2)))
            list1[i1], list2[i2] = list2[i2], list1[i1]
        elif list1:
            # Move a random order from d1 to d2 if d2 has capacity
            drone2_data = drones[d2]
            if drone2_data["current_load"] + len(list2) < drone2_data["max_capacity"]:
                i1 = int(self.rng.integers(len(list1)))
                list2.append(list1.pop(i1))
        elif list2:
            drone1_data = drones[d1]
            if drone1_data["current_load"] + len(list1) < drone1_data["max_capacity"]:
                i2 = int(self.rng.integers(len(list2)))
                list1.append(list2.pop(i2))

        new_assign[id1] = list1
        new_assign[id2] = list2
        return new_assign
