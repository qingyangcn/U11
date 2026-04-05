from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np

from config.default import AppConfig, build_default_config
from core.entities import WeatherSnapshot
from core.enums import WeatherState
from core.utils import load_json


class WeatherProcess:
    def __init__(self, cfg: AppConfig | None = None) -> None:
        self.cfg = cfg or build_default_config()
        payload = load_json(self.cfg.paths.fitted_dir / "weather_mapping.json")
        self.change_interval_min = float(payload["change_interval_min"])
        self.records_by_hour: Dict[int, List[dict]] = defaultdict(list)
        for record in payload["records"]:
            self.records_by_hour[int(record["hour"])].append(record)
        self._schedule: List[Tuple[float, WeatherSnapshot]] = []

    def reset(self, rng: np.random.Generator, total_minutes: float) -> None:
        business_start = self.cfg.order.business_start_hour
        schedule: List[Tuple[float, WeatherSnapshot]] = []
        t = 0.0
        while t <= total_minutes + 1e-6:
            absolute_hour = business_start + int(t // 60.0)
            hour_of_day = absolute_hour % 24
            records = self.records_by_hour.get(hour_of_day) or self.records_by_hour[min(self.records_by_hour)]
            record = records[int(rng.integers(0, len(records)))]
            schedule.append(
                (
                    t,
                    WeatherSnapshot(
                        state=WeatherState(record["state"]),
                        speed_factor=float(record["speed_factor"]),
                        energy_factor=float(record["energy_factor"]),
                        hover_factor=float(record["hover_factor"]),
                        wind_speed_kmph=float(record["wind_speed_kmph"]),
                        humidity=float(record["humidity"]),
                        summary=str(record["summary"]),
                    ),
                )
            )
            t += self.change_interval_min
        self._schedule = schedule

    @property
    def schedule(self) -> List[Tuple[float, WeatherSnapshot]]:
        return self._schedule

    def snapshot_at(self, time_min: float) -> WeatherSnapshot:
        if not self._schedule:
            raise RuntimeError("WeatherProcess.reset must be called before snapshot_at.")
        idx = 0
        for i, (t, _) in enumerate(self._schedule):
            if t <= time_min:
                idx = i
            else:
                break
        return self._schedule[idx][1]
