from __future__ import annotations

from typing import Dict

import numpy as np

from env.constraints import TaskProjection
from rl.rule_actions import NON_WAIT_RULES, RULE_ACTIONS


def build_action_alias_details(
    rule_prototypes: Dict[str, TaskProjection | None],
    force_charge: bool,
) -> Dict[str, Dict[str, object]]:
    details: Dict[str, Dict[str, object]] = {}
    if force_charge:
        for rule_name in RULE_ACTIONS:
            details[rule_name] = {
                "available": False,
                "kept": False,
                "alias_of": None,
                "order_id": None,
                "task_type": None,
            }
        return details

    seen_signatures: Dict[tuple[str, str], str] = {}
    for rule_name in RULE_ACTIONS:
        prototype = rule_prototypes.get(rule_name)
        if prototype is None:
            details[rule_name] = {
                "available": False,
                "kept": False,
                "alias_of": None,
                "order_id": None,
                "task_type": None,
            }
            continue
        signature = (prototype.order_id, prototype.task_type.value)
        alias_of = seen_signatures.get(signature)
        kept = alias_of is None
        if kept:
            seen_signatures[signature] = rule_name
        details[rule_name] = {
            "available": True,
            "kept": kept,
            "alias_of": alias_of,
            "order_id": prototype.order_id,
            "task_type": prototype.task_type.value,
        }
    return details


def build_action_mask(
    rule_prototypes: Dict[str, TaskProjection | None],
    force_charge: bool,
) -> np.ndarray:
    alias_details = build_action_alias_details(rule_prototypes, force_charge)
    mask = [bool(alias_details[rule_name]["kept"]) for rule_name in NON_WAIT_RULES]
    return np.asarray(mask, dtype=bool)
