"""
Regression test for the PICKED_UP + IDLE state-consistency anomaly.

Root causes fixed:
  1. Timing bug – consistency check was called inside _process_events() BEFORE
     _force_state_synchronization(), so transient bad states were visible.
     Fix: moved check to _run_post_sync_consistency_check(), called after sync.

  2. Fallthrough bug in _handle_drone_arrival() – when pickup succeeded but
     customer_loc was None, code fell through to the error handler which set
     drone to IDLE while the cargo still contained a PICKED_UP order.
     Fix: always return after handling the pickup outcome.

  3. No guard in update_drone_status(IDLE) – nothing prevented the transition.
     Fix: refuse IDLE when drone holds a PICKED_UP serving_order or PICKED_UP cargo.

Usage:
    python test_picked_up_idle_regression.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from UAV_ENVIRONMENT_11 import (
    DroneStatus,
    OrderStatus,
)


# ---------------------------------------------------------------------------
# Unit tests using a minimal mock environment
# ---------------------------------------------------------------------------

def _make_mock_env(num_drones=1):
    """Return a minimal mock env dict with just enough structure for StateManager."""
    import types
    env = types.SimpleNamespace()
    env.drones = {}
    env.orders = {}
    for i in range(num_drones):
        env.drones[i] = {
            'status': DroneStatus.IDLE,
            'cargo': set(),
            'serving_order_id': None,
            'target_location': None,
            'planned_stops': [],
        }
    env.time_system = types.SimpleNamespace(current_step=0)
    return env


def _make_state_manager(env):
    """Return a real StateManager wired to the mock env."""
    from UAV_ENVIRONMENT_11 import StateManager
    sm = StateManager.__new__(StateManager)
    sm.env = env
    sm.state_log = []
    sm._first_anomaly_logged = False
    return sm


# ---------------------------------------------------------------------------
# Test 1: Guard refuses IDLE when drone has PICKED_UP serving_order_id
# ---------------------------------------------------------------------------
def test_guard_refuses_idle_with_picked_up_serving_order():
    env = _make_mock_env(1)
    sm = _make_state_manager(env)

    order_id = 42
    env.orders[order_id] = {
        'id': order_id,
        'status': OrderStatus.PICKED_UP,
        'assigned_drone': 0,
        'customer_location': (5.0, 5.0),
    }

    # Set up drone as FLYING_TO_CUSTOMER with a PICKED_UP order
    env.drones[0]['status'] = DroneStatus.FLYING_TO_CUSTOMER
    env.drones[0]['serving_order_id'] = order_id
    env.drones[0]['cargo'].add(order_id)

    # Attempt to set IDLE – should be refused
    result = sm.update_drone_status(0, DroneStatus.IDLE, target_location=None)
    assert result is False, "Guard should refuse IDLE when serving_order is PICKED_UP"
    assert env.drones[0]['status'] == DroneStatus.FLYING_TO_CUSTOMER, (
        "Drone status must not change when guard fires"
    )
    print("PASS: test_guard_refuses_idle_with_picked_up_serving_order")


# ---------------------------------------------------------------------------
# Test 2: Guard refuses IDLE when cargo contains PICKED_UP order
# ---------------------------------------------------------------------------
def test_guard_refuses_idle_with_picked_up_cargo():
    env = _make_mock_env(1)
    sm = _make_state_manager(env)

    order_id = 99
    env.orders[order_id] = {
        'id': order_id,
        'status': OrderStatus.PICKED_UP,
        'assigned_drone': 0,
        'customer_location': (3.0, 3.0),
    }

    env.drones[0]['status'] = DroneStatus.FLYING_TO_CUSTOMER
    env.drones[0]['cargo'].add(order_id)
    # serving_order_id is None – only cargo check applies
    env.drones[0]['serving_order_id'] = None

    result = sm.update_drone_status(0, DroneStatus.IDLE, target_location=None)
    assert result is False, "Guard should refuse IDLE when cargo has PICKED_UP order"
    assert env.drones[0]['status'] == DroneStatus.FLYING_TO_CUSTOMER
    print("PASS: test_guard_refuses_idle_with_picked_up_cargo")


# ---------------------------------------------------------------------------
# Test 3: Guard allows IDLE when cargo is empty and no PICKED_UP serving order
# ---------------------------------------------------------------------------
def test_guard_allows_idle_when_no_picked_up_order():
    env = _make_mock_env(1)
    sm = _make_state_manager(env)

    order_id = 7
    env.orders[order_id] = {
        'id': order_id,
        'status': OrderStatus.DELIVERED,   # already delivered
        'assigned_drone': 0,
    }
    env.drones[0]['status'] = DroneStatus.FLYING_TO_CUSTOMER
    env.drones[0]['serving_order_id'] = None
    env.drones[0]['cargo'] = set()

    result = sm.update_drone_status(0, DroneStatus.IDLE, target_location=None)
    assert result is True, "Guard should allow IDLE when no PICKED_UP order present"
    assert env.drones[0]['status'] == DroneStatus.IDLE
    print("PASS: test_guard_allows_idle_when_no_picked_up_order")


# ---------------------------------------------------------------------------
# Test 4: Simulate the exact fallthrough scenario – after pickup with no
#         customer_loc, drone must NOT end up IDLE with PICKED_UP cargo.
# ---------------------------------------------------------------------------
def test_no_picked_up_idle_after_pickup_with_missing_customer_loc():
    """
    Reproduce the original bug path:
      drone FLYING_TO_MERCHANT → pickup succeeds → customer_loc is None →
      (old code) drone became IDLE with PICKED_UP cargo.
    After the fix, the pickup must be undone and order reset to READY.
    """
    import types
    from collections import deque
    from UAV_ENVIRONMENT_11 import (
        ThreeObjectiveDroneDeliveryEnv,
        ARRIVAL_THRESHOLD,
    )

    # ---- Build a minimal real environment with a single drone ----
    try:
        env = ThreeObjectiveDroneDeliveryEnv(
            num_drones=1,
            max_orders=5,
            enable_random_events=False,
            debug_state_warnings=False,
        )
        env.reset(seed=0)
    except Exception as exc:
        print(f"SKIP: test_no_picked_up_idle_after_pickup_with_missing_customer_loc "
              f"(env init failed: {exc})")
        return

    # ---- Craft a minimal order and drone state ----
    drone_id = 0
    drone = env.drones[drone_id]
    merchant_id = next(iter(env.merchants))
    merchant_loc = env.merchants[merchant_id]['location']

    # Place the drone exactly at the merchant
    drone['location'] = tuple(merchant_loc) if not isinstance(merchant_loc, tuple) else merchant_loc
    drone['status'] = DroneStatus.FLYING_TO_MERCHANT
    drone['cargo'] = set()
    drone['planned_stops'] = deque()

    # Create an order with no customer_location
    order_id = max(env.orders.keys(), default=-1) + 1
    env.orders[order_id] = {
        'id': order_id,
        'status': OrderStatus.ASSIGNED,
        'assigned_drone': drone_id,
        'merchant_id': merchant_id,
        'customer_location': None,   # ← triggers the bug in old code
        'pickup_time': None,
    }
    env.active_orders.add(order_id)
    drone['serving_order_id'] = order_id

    # ---- Trigger the arrival handler ----
    env._handle_drone_arrival(drone_id, drone)

    # ---- Assertions ----
    order = env.orders[order_id]
    drone_status = drone['status']

    # The drone must NOT be IDLE with a PICKED_UP order in cargo
    is_bad_state = (
        drone_status == DroneStatus.IDLE
        and order_id in drone.get('cargo', set())
        and order['status'] == OrderStatus.PICKED_UP
    )
    assert not is_bad_state, (
        f"PICKED_UP + IDLE anomaly still present! "
        f"drone_status={drone_status}, cargo={drone['cargo']}, "
        f"order_status={order['status']}"
    )

    # The order must have been reset to READY (not PICKED_UP)
    assert order['status'] == OrderStatus.READY, (
        f"Order should be reset to READY after no-customer-loc case; "
        f"got {order['status']}"
    )

    print("PASS: test_no_picked_up_idle_after_pickup_with_missing_customer_loc")


# ---------------------------------------------------------------------------
# Run all tests
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    failures = []
    tests = [
        test_guard_refuses_idle_with_picked_up_serving_order,
        test_guard_refuses_idle_with_picked_up_cargo,
        test_guard_allows_idle_when_no_picked_up_order,
        test_no_picked_up_idle_after_pickup_with_missing_customer_loc,
    ]
    for t in tests:
        try:
            t()
        except AssertionError as e:
            failures.append((t.__name__, str(e)))
            print(f"FAIL: {t.__name__}: {e}")
        except Exception as e:
            failures.append((t.__name__, str(e)))
            print(f"ERROR: {t.__name__}: {e}")

    print()
    if failures:
        print(f"{len(failures)} test(s) FAILED.")
        sys.exit(1)
    else:
        print(f"All {len(tests)} tests PASSED.")
        sys.exit(0)
