from __future__ import annotations

from typing import Dict

from core.entities import Order
from env.constraints import TaskProjection


RULE_BASELINE_POLICIES = (
    "nearest",
    "earliest_deadline",
    "minimum_slack",
)


def select_rule_action(policy_name: str, projections: Dict[str, TaskProjection], orders: Dict[str, Order]) -> str:
    if policy_name not in RULE_BASELINE_POLICIES:
        raise KeyError(f"Unknown rule baseline policy: {policy_name}")
    del projections, orders
    return policy_name
