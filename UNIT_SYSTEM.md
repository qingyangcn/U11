# Unit System And Constant Closure

This file documents the current runtime unit system after the battery and
energy model was upgraded to a `Wh`-based formulation.

## Core Runtime Units

The simulator now uses the following runtime units:

- planar position: `km`
- distance: `km`
- time: `min`
- speed: `km/min` internally, configured as `km/h`
- payload and capacity: `kg`
- battery energy: `Wh`
- cruise and hover power: `W`
- service durations: `min`
- dispatch interval: `min`
- weather change interval: `min`
- simulation horizon: `min`

## Where The Conversions Happen

- speed is configured as `max_speed_kmph` in
  [default.py](/C:/Users/Lenovo/PycharmProjects/pythonProject/RL-T/UAV/config/default.py)
  and converted to `km/min` when drones are built in
  [generator.py](/C:/Users/Lenovo/PycharmProjects/pythonProject/RL-T/UAV/env/generator.py)
- travel time and travel energy are computed in
  [travel.py](/C:/Users/Lenovo/PycharmProjects/pythonProject/RL-T/UAV/env/travel.py)
- battery lower bounds and return-to-station energy are computed in
  [battery.py](/C:/Users/Lenovo/PycharmProjects/pythonProject/RL-T/UAV/env/battery.py)

## Battery And Energy Semantics

The energy model is now expressed in `Wh`:

- `battery_capacity_wh` is the full pack energy
- `battery_current` is the remaining pack energy in `Wh`
- `safety_reserve_ratio` defines the protected reserve
- `safety_energy = battery_capacity_wh * safety_reserve_ratio`

The default configuration currently uses:

- `battery_capacity_wh = 800`
- `safety_reserve_ratio = 0.15`
- `cruise_power_empty_w = 500`
- `cruise_power_full_w = 900`
- `hover_power_empty_w = 700`
- `hover_power_full_w = 1100`

Cruise power and hover power are both interpolated linearly by payload ratio.

## Closed Physical-Style Formulas

### Flight Time

- effective speed = `max_speed_km_min * weather.speed_factor * load_factor`
- `load_factor = max(min_speed_ratio, 1 - speed_load_penalty * load_ratio)`

### Flight Energy

- `flight_energy_wh = cruise_power_w * flight_time_hours * weather.energy_factor`

### Hover And Service Energy

- `hover_energy_wh = hover_power_w * duration_hours * weather.hover_factor * mode_factor`

These formulas are implemented in
[travel.py](/C:/Users/Lenovo/PycharmProjects/pythonProject/RL-T/UAV/env/travel.py).

## Closed And Enforced Runtime Constraints

The following checks are now closed under the same `Wh` unit system:

- payload must not exceed `max_capacity_kg`
- every feasible task projection must leave enough energy for return to the
  nearest station plus safety reserve
- forced charging uses:
  - current position
  - current payload
  - energy to the nearest station
  - safety reserve

These checks are enforced in:

- [constraints.py](/C:/Users/Lenovo/PycharmProjects/pythonProject/RL-T/UAV/env/constraints.py)
- [battery.py](/C:/Users/Lenovo/PycharmProjects/pythonProject/RL-T/UAV/env/battery.py)
- [simulator.py](/C:/Users/Lenovo/PycharmProjects/pythonProject/RL-T/UAV/env/simulator.py)

## What Is Still Approximate

The model is now unit-closed in `Wh/W/km/min/kg`, but it is still an
engineering approximation rather than a certified aircraft power model.

The main approximations are:

- cruise and hover power are linearly interpolated by payload ratio
- weather acts as multiplicative power and speed factors
- service phases reuse hover-style power rather than a more detailed actuator
  model
- reward coefficients are still task-level engineering preferences, not money
  cost or regulatory cost

So the model is physically interpretable, but still simplified.

## Normalization And Reward Scales

The following constants are not physical units. They are normalization or reward
reference scales:

- `time_ref_min`
- `onboard_norm_ref`
- `assigned_norm_ref`
- `quantity_ref_kg`
- `complete_norm_cap`
- `lateness_norm_cap`

They affect observation and reward scaling, not the physical state transition
itself.

## Current Interpretation

The current implementation should be read as:

- kinematics and timing units are unified
- payload units are unified
- battery and energy units are unified in `Wh`
- return-feasibility and forced-charging logic use the same energy unit
- observation and reward scales are still engineering references

## Remaining Closure Question

One important choice remains methodological rather than physical:

- the upper-layer `MOPSO` objective and lower-layer `RL` reward are now more
  consistent in feasibility semantics
- but they are still not literally the same objective formula

That is a multi-objective design question, not a unit-system inconsistency.
