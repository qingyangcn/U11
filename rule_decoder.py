from __future__ import annotations

from typing import Dict, Optional

from core.entities import Order
from core.enums import TaskType
from env.constraints import TaskProjection
from rl.rule_actions import NON_WAIT_RULES


def _nearest_projection(candidates: list[TaskProjection], orders: Dict[str, Order]) -> TaskProjection:
    return min(
        candidates,
        key=lambda item: (
            item.travel_time_min,
            orders[item.order_id].deadline,
            item.order_id,
        ),
    )


def select_rule_projection(
    rule_name: str,
    projections: Dict[str, TaskProjection],
    orders: Dict[str, Order],
) -> TaskProjection | None:
    if not projections:
        return None
    candidates = list(projections.values())

    if rule_name == "nearest":
        return _nearest_projection(candidates, orders)
    if rule_name == "earliest_deadline":
        return min(candidates, key=lambda item: (orders[item.order_id].deadline, item.travel_time_min, item.order_id))
    if rule_name == "minimum_slack":
        return min(
            candidates,
            key=lambda item: (
                orders[item.order_id].deadline - item.order_finish_time,
                item.travel_time_min,
                item.order_id,
            ),
        )
    if rule_name == "deliver_first":
        dropoffs = [item for item in candidates if item.task_type == TaskType.DROPOFF]
        if dropoffs:
            return _nearest_projection(dropoffs, orders)
        return _nearest_projection(candidates, orders)
    if rule_name == "pickup_first":
        pickups = [item for item in candidates if item.task_type == TaskType.PICKUP]
        if pickups:
            return _nearest_projection(pickups, orders)
        return _nearest_projection(candidates, orders)
    return None


def build_rule_prototypes(
    projections: Dict[str, TaskProjection],
    orders: Dict[str, Order],
) -> Dict[str, TaskProjection | None]:
    return {rule: select_rule_projection(rule, projections, orders) for rule in NON_WAIT_RULES}
