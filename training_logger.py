from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pandas as pd
from stable_baselines3.common.callbacks import BaseCallback

from analysis.metrics import episodes_to_frame


class EpisodeStatsCallback(BaseCallback):
    def __init__(self, csv_path: Path, verbose: int = 0) -> None:
        super().__init__(verbose=verbose)
        self.csv_path = csv_path
        self.rows: List[Dict] = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])
        for info, done in zip(infos, dones):
            if not done:
                continue
            row: Dict = {
                "episode_index": len(self.rows),
                "timesteps": int(self.num_timesteps),
            }
            episode = info.get("episode")
            if episode:
                row["episode_reward"] = float(episode.get("r", 0.0))
                row["episode_length"] = float(episode.get("l", 0.0))
                row["episode_wall_time_sec"] = float(episode.get("t", 0.0))
            row.update(info.get("episode_stats", {}))
            self.rows.append(row)
        return True

    def _on_training_end(self) -> None:
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        frame = episodes_to_frame(self.rows)
        frame.to_csv(self.csv_path, index=False)
