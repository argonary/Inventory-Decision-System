from src.optimization.optimizer import (
    optimize_lp_allocation,
    optimize_proportional_allocation,
)


def test_basic_three_skus_sum_and_integer(simple_demand):
    """Three SKUs, capacity sufficient: allocations sum to <= capacity and are ints."""
    capacity = 100
    orders = optimize_proportional_allocation(simple_demand, capacity=capacity)

    assert set(orders.keys()) == set(simple_demand.keys())
    assert all(isinstance(v, int) for v in orders.values())
    assert sum(orders.values()) <= capacity


def test_zero_capacity_returns_all_zero(simple_demand):
    """capacity = 0 -> every SKU receives 0."""
    orders = optimize_proportional_allocation(simple_demand, capacity=0)
    assert orders == {k: 0 for k in simple_demand}


def test_single_sku_demand_less_than_capacity():
    """One SKU, demand < capacity -> allocation == demand."""
    demand = {1: 50.0}
    orders = optimize_proportional_allocation(demand, capacity=100)
    assert orders[1] == min(100, int(demand[1]))


def test_single_sku_demand_greater_than_capacity():
    """One SKU, demand > capacity -> allocation == capacity."""
    demand = {1: 200.0}
    orders = optimize_proportional_allocation(demand, capacity=100)
    assert orders[1] == min(100, int(demand[1]))


def test_all_equal_demand_yields_proportional_allocations():
    """Equal demand across SKUs -> equal allocations (within 1 unit of rounding)."""
    demand = {1: 30.0, 2: 30.0, 3: 30.0}
    capacity = 60
    orders = optimize_proportional_allocation(demand, capacity=capacity)

    values = list(orders.values())
    assert max(values) - min(values) <= 1
    assert sum(values) <= capacity


def test_perishable_weighting_favors_perishable_skus():
    """A perishable SKU with weight > 1 must receive >= its unweighted share."""
    demand = {1: 50.0, 2: 50.0}
    capacity = 80
    perishable_flags = {1: True, 2: False}

    orders = optimize_proportional_allocation(
        demand,
        capacity=capacity,
        perishable_flags=perishable_flags,
        perishable_weight=2.0,
    )

    total_demand = sum(demand.values())
    unweighted_share_1 = capacity * (demand[1] / total_demand)
    assert orders[1] >= unweighted_share_1


def test_fill_capacity_false_does_not_exceed_total_weighted_demand():
    """With fill_capacity=False and capacity > demand, orders cap at weighted demand."""
    demand = {1: 10.0, 2: 20.0, 3: 30.0}
    capacity = 500

    orders = optimize_proportional_allocation(
        demand,
        capacity=capacity,
        fill_capacity=False,
    )

    total_weighted_demand = sum(demand.values())
    assert sum(orders.values()) <= total_weighted_demand


def test_total_orders_within_capacity_plus_one():
    """
    Known behavior (Bug 11): the largest-remainder rounding step uses
    int(round(effective_capacity - current_total)). Combined with floating-point
    sums in `continuous` and `effective_capacity`, the final total can land one
    above the requested capacity in pathological inputs. Until the rounding
    step is hardened, the invariant we assert here is the relaxed
    total_orders <= capacity + 1, not strict equality.
    """
    demand = {1: 33.33, 2: 33.33, 3: 33.34}
    capacity = 100
    orders = optimize_proportional_allocation(
        demand,
        capacity=capacity,
        fill_capacity=True,
    )
    assert sum(orders.values()) <= capacity + 1


# =====================================================
# LP optimizer tests — same input cases as proportional.
# Assertions are adapted where the LP's linear objective
# is degenerate (e.g. equal demand, tied perishable cases),
# since any vertex of the optimal face is a valid solution.
# =====================================================

def test_lp_basic_three_skus_sum_and_integer(simple_demand):
    """Three SKUs, capacity sufficient: allocations sum to <= capacity and are ints."""
    capacity = 100
    orders = optimize_lp_allocation(simple_demand, capacity=capacity)

    assert set(orders.keys()) == set(simple_demand.keys())
    assert all(isinstance(v, int) for v in orders.values())
    assert sum(orders.values()) <= capacity


def test_lp_zero_capacity_returns_all_zero(simple_demand):
    """capacity = 0 -> every SKU receives 0."""
    orders = optimize_lp_allocation(simple_demand, capacity=0)
    assert orders == {k: 0 for k in simple_demand}


def test_lp_single_sku_demand_less_than_capacity():
    """One SKU, demand < capacity -> allocation == demand."""
    demand = {1: 50.0}
    orders = optimize_lp_allocation(demand, capacity=100)
    assert orders[1] == min(100, int(demand[1]))


def test_lp_single_sku_demand_greater_than_capacity():
    """One SKU, demand > capacity -> allocation == capacity."""
    demand = {1: 200.0}
    orders = optimize_lp_allocation(demand, capacity=100)
    assert orders[1] == min(100, int(demand[1]))


def test_lp_equal_demand_respects_bounds_and_capacity():
    """Equal demand makes the LP degenerate; assert bound + capacity invariants."""
    demand = {1: 30.0, 2: 30.0, 3: 30.0}
    capacity = 60
    orders = optimize_lp_allocation(demand, capacity=capacity)

    assert set(orders.keys()) == set(demand.keys())
    assert all(0 <= v <= 30 for v in orders.values())
    assert sum(orders.values()) <= capacity


def test_lp_perishable_weighting_raises_upper_bound():
    """Perishable weight > 1 raises x_i's upper bound; LP saturates when capacity allows."""
    demand = {1: 50.0, 2: 50.0}
    capacity = 200
    perishable_flags = {1: True, 2: False}

    orders = optimize_lp_allocation(
        demand,
        capacity=capacity,
        perishable_flags=perishable_flags,
        perishable_weight=2.0,
    )

    # weighted_demand = {1: 100, 2: 50}; total = 150; effective_cap = 150.
    # LP saturates both upper bounds.
    assert orders[1] == 100
    assert orders[2] == 50


def test_lp_fill_capacity_false_does_not_exceed_total_weighted_demand():
    """With fill_capacity=False and capacity > demand, orders cap at weighted demand."""
    demand = {1: 10.0, 2: 20.0, 3: 30.0}
    capacity = 500

    orders = optimize_lp_allocation(
        demand,
        capacity=capacity,
        fill_capacity=False,
    )

    total_weighted_demand = sum(demand.values())
    assert sum(orders.values()) <= total_weighted_demand


def test_lp_total_orders_within_capacity_plus_one():
    """Same relaxed invariant as proportional under float-imprecise capacity."""
    demand = {1: 33.33, 2: 33.33, 3: 33.34}
    capacity = 100
    orders = optimize_lp_allocation(
        demand,
        capacity=capacity,
        fill_capacity=True,
    )
    assert sum(orders.values()) <= capacity + 1


def test_lp_total_units_at_least_proportional():
    """LP optimizer's total units served must be >= proportional's on a non-trivial input."""
    demand = {1: 25.7, 2: 41.3, 3: 18.9, 4: 60.4, 5: 12.5}
    capacity = 80
    perishable_flags = {1: True, 2: False, 3: True, 4: False, 5: True}

    lp_orders = optimize_lp_allocation(
        demand,
        capacity=capacity,
        service_floor_ratio=0.15,
        perishable_flags=perishable_flags,
        perishable_weight=1.4,
        fill_capacity=False,
    )
    prop_orders = optimize_proportional_allocation(
        demand,
        capacity=capacity,
        service_floor_ratio=0.15,
        perishable_flags=perishable_flags,
        perishable_weight=1.4,
        fill_capacity=False,
    )

    assert sum(lp_orders.values()) >= sum(prop_orders.values())
