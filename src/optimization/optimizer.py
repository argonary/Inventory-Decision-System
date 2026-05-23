from typing import Dict, Optional
import numpy as np
from scipy.optimize import linprog


def optimize_proportional_allocation(
    demand: Dict[int, float],
    capacity: int,
    service_floor_ratio: float = 0.0,
    perishable_flags: Optional[Dict[int, bool]] = None,
    perishable_weight: float = 1.0,
    fill_capacity: bool = False,
) -> Dict[int, int]:
    """
    Proportionally allocate inventory with:
    - capacity as MAX constraint (default)
    - optional exact-capacity fill
    - adaptive service-level floors
    - optional perishable weighting
    """

    if capacity <= 0 or not demand:
        return {k: 0 for k in demand}

    # ---- Clean demand ----
    demand = {k: max(v, 0.0) for k, v in demand.items()}

    # ---- Apply perishable weighting ----
    if perishable_flags:
        weighted_demand = {
            k: (
                demand[k] * perishable_weight
                if perishable_flags.get(k, False)
                else demand[k]
            )
            for k in demand
        }
    else:
        weighted_demand = demand.copy()

    total_weighted_demand = sum(weighted_demand.values())
    if total_weighted_demand == 0:
        return {k: 0 for k in demand}

    # ---- Effective capacity (KEY FIX) ----
    effective_capacity = (
        capacity if fill_capacity
        else min(capacity, total_weighted_demand)
    )

    # ---- Service floors ----
    raw_floors = {
        k: service_floor_ratio * weighted_demand[k]
        for k in weighted_demand
    }

    total_floor = sum(raw_floors.values())

    if total_floor > effective_capacity:
        # Floor guarantees degrade proportionally when capacity is too tight
        # to honor all minimums. This is intentional: a scaled floor is
        # better than an infeasible allocation.
        scale = effective_capacity / total_floor
        floors = {k: v * scale for k, v in raw_floors.items()}
    else:
        floors = raw_floors

    remaining_capacity = effective_capacity - sum(floors.values())

    # ---- Residual demand ----
    residual_demand = {
        k: max(weighted_demand[k] - floors[k], 0.0)
        for k in weighted_demand
    }

    residual_total = sum(residual_demand.values())

    # ---- Proportional allocation ----
    continuous = {}
    for k in weighted_demand:
        if residual_total > 0:
            alloc = floors[k] + remaining_capacity * (
                residual_demand[k] / residual_total
            )
        else:
            alloc = floors[k]
        continuous[k] = alloc

    # ---- Rounding (largest remainder) ----
    rounded = {k: int(np.floor(v)) for k, v in continuous.items()}
    current_total = sum(rounded.values())
    gap = int(round(effective_capacity - current_total))

    remainders = {
        k: continuous[k] - rounded[k]
        for k in continuous
    }

    for k, _ in sorted(remainders.items(), key=lambda x: x[1], reverse=True):
        if gap <= 0:
            break
        rounded[k] += 1
        gap -= 1

    return rounded


def optimize_lp_allocation(
    demand: Dict[int, float],
    capacity: int,
    service_floor_ratio: float = 0.0,
    perishable_flags: Optional[Dict[int, bool]] = None,
    perishable_weight: float = 1.0,
    fill_capacity: bool = False,
) -> Dict[int, int]:
    """
    LP-based allocation using scipy.optimize.linprog.

    Minimizes total underage (equivalent to maximizing units served) subject to:
      - sum(x_i) <= effective_capacity
      - x_i >= floor_i
      - x_i <= weighted_demand_i

    Returns integer allocations via largest-remainder rounding. Signature matches
    optimize_proportional_allocation so callers can swap freely.
    """

    if capacity <= 0 or not demand:
        return {k: 0 for k in demand}

    demand = {k: max(v, 0.0) for k, v in demand.items()}

    if perishable_flags:
        weighted_demand = {
            k: (
                demand[k] * perishable_weight
                if perishable_flags.get(k, False)
                else demand[k]
            )
            for k in demand
        }
    else:
        weighted_demand = demand.copy()

    total_weighted_demand = sum(weighted_demand.values())
    if total_weighted_demand == 0:
        return {k: 0 for k in demand}

    effective_capacity = (
        capacity if fill_capacity
        else min(capacity, total_weighted_demand)
    )

    raw_floors = {
        k: service_floor_ratio * weighted_demand[k]
        for k in weighted_demand
    }
    total_floor = sum(raw_floors.values())

    if total_floor > effective_capacity:
        scale = effective_capacity / total_floor
        floors = {k: v * scale for k, v in raw_floors.items()}
    else:
        floors = raw_floors

    keys = list(weighted_demand.keys())
    n = len(keys)

    c = np.full(n, -1.0)
    A_ub = np.ones((1, n))
    b_ub = np.array([effective_capacity], dtype=float)
    bounds = [(floors[k], weighted_demand[k]) for k in keys]

    result = linprog(
        c=c,
        A_ub=A_ub,
        b_ub=b_ub,
        bounds=bounds,
        method="highs",
    )

    if result.success:
        continuous = {keys[i]: float(result.x[i]) for i in range(n)}
    else:
        # Construction guarantees feasibility; fall back to floors if solver fails.
        continuous = {k: floors[k] for k in keys}

    rounded = {k: int(np.floor(v)) for k, v in continuous.items()}
    current_total = sum(rounded.values())
    gap = int(round(effective_capacity - current_total))

    remainders = {k: continuous[k] - rounded[k] for k in continuous}

    for k, _ in sorted(remainders.items(), key=lambda x: x[1], reverse=True):
        if gap <= 0:
            break
        rounded[k] += 1
        gap -= 1

    return rounded
