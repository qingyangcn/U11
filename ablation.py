from __future__ import annotations

from dataclasses import replace
from typing import Dict, List

from config.default import AppConfig


def build_ablation_configs(cfg: AppConfig) -> List[Dict]:
    return [
        {"name": "main", "config": cfg},
        {"name": "higher_dispatch_interval", "config": replace(cfg, dispatch=replace(cfg.dispatch, interval_min=15.0))},
        {"name": "lower_dispatch_interval", "config": replace(cfg, dispatch=replace(cfg.dispatch, interval_min=5.0))},
    ]
