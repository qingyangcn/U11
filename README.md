# Multi-UAV Hierarchical Scheduler

This project is a first runnable implementation of the design described in
`多无人机调度问题_统一设定与实现说明_完善版.md`.

Main characteristics:

- event-driven multi-UAV simulator
- default simulation horizon is one day (1440 minutes)
- upper-layer rolling MOPSO order assignment
- lower-layer rule-selection RL policy
- Gymnasium environment compatible with SB3 MaskablePPO
- dataset fitting pipeline for arrivals, cancellation, spatial sampling, and weather

The codebase is intentionally modular so individual modules can be replaced or
refined without rewriting the full system.

Current runtime semantics:

- orders are not pre-assigned before `ready_time`
- cancellation is not a standalone `ORDER_CANCEL` event; it is evaluated at trigger points
- the RL action space has 5 rule actions:
  - `nearest`
  - `earliest_deadline`
  - `minimum_slack`
  - `deliver_first`
  - `pickup_first`
- the explicit `wait` action has been removed from the policy space; when no feasible task exists, the simulator waits automatically until the next dispatch epoch or terminal time
- `deliver_first` means:
  - choose the nearest task inside the current `dropoff` subset if any dropoff exists
  - otherwise fall back to global `nearest`
- `pickup_first` means:
  - choose the nearest task inside the current `pickup` subset if any pickup exists
  - otherwise fall back to global `nearest`

Decision semantics at a simulator time slice `t`:

1. process all non-decision events with `timestamp = t`
2. inside that set, use fixed event priority plus a fixed external tie-break key
3. after all those events are settled, collect the local-decision drone set `D_t`
4. execute zero-time serial decisions over `D_t`
5. advance the clock only after `D_t` becomes empty

Important note:

- the order inside `D_t` is not controlled by the policy
- it is determined by a simulator-side seeded pseudo-random ordering based on `seed + current_time + drone_id`
- this avoids always using raw `drone_id` order while keeping the tie-break external and reproducible

Exact alias masking:

- multiple rule actions can decode to the same concrete task
- the environment now masks only exact aliases
- an exact alias means the decoded actions point to the same `order_id` and same task type (`pickup` or `dropoff`)
- canonical keep order is:
  - `nearest > earliest_deadline > minimum_slack > deliver_first > pickup_first`
- alias statistics are exported during evaluation and ablation runs as:
  - `alias_usage.csv`
  - `alias_usage_summary.csv`
  - `alias_usage_summary.json`

Evaluation notes:

- old checkpoints trained with the previous 6-action policy are not compatible with the current 5-action environment and must be retrained
- complete-method comparisons are run with `runner.compare_complete_baselines`
- local-rule replacement studies are run with `runner.compare_ablations`
- the current unit system and constant-closure notes are documented in
  [UNIT_SYSTEM.md](/C:/Users/Lenovo/PycharmProjects/pythonProject/RL-T/UAV/UNIT_SYSTEM.md)
  and now use a `Wh/W/km/min/kg` battery-energy formulation

Recommended execution style:

- set the PyCharm interpreter to `E:\anaconda\envs\qingyang\python.exe`
- use `RL-T` as the project root
- run modules from the parent directory, for example:
  - `python -m UAV.runner.train --scenario medium`
  - `python -m UAV.runner.evaluate --scenario medium --rule nearest`
  - `python -m UAV.runner.compare_ablations --scenario medium --episodes 10`
  - `python -m UAV.runner.compare_complete_baselines --scenario high --episodes 10 --model maskableppo_high --model-label main_method`
  - `python -m UAV.runner.compare_swarm_baselines --scenario high --episodes 10 --model maskableppo_high --model-label main_method`
  - `python -m unittest UAV.tests.test_events UAV.tests.test_cancel`
