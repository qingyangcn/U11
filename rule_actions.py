from __future__ import annotations

RULE_ACTIONS = (
    "nearest",
    "earliest_deadline",
    "minimum_slack",
    "deliver_first",
    "pickup_first",
)

NON_WAIT_RULES = RULE_ACTIONS
ACTION_TO_INDEX = {name: idx for idx, name in enumerate(RULE_ACTIONS)}
INDEX_TO_ACTION = {idx: name for idx, name in enumerate(RULE_ACTIONS)}
