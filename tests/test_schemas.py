import pytest
from pydantic import ValidationError

from api.schemas import ForecastToOrdersRequest


def _base_payload() -> dict:
    return {
        "date": "2016-04-21",
        "store_nbr": 44,
        "service_level": "p90",
        "items": [
            {"item_nbr": 769314, "onpromotion": False},
        ],
        "capacity_units": 100,
        "service_floor_ratio": 0.1,
        "perishable_weight": 1.2,
    }


def test_valid_payload_constructs():
    """A well-formed payload constructs without error."""
    req = ForecastToOrdersRequest(**_base_payload())
    assert req.service_level == "p90"
    assert req.capacity_units == 100
    assert len(req.items) == 1


def test_invalid_service_level_raises():
    """service_level outside {'p90', 'p95'} is rejected."""
    payload = _base_payload()
    payload["service_level"] = "p80"
    with pytest.raises(ValidationError):
        ForecastToOrdersRequest(**payload)


@pytest.mark.parametrize("bad_capacity", [0, -10])
def test_non_positive_capacity_raises(bad_capacity):
    """capacity_units must be > 0."""
    payload = _base_payload()
    payload["capacity_units"] = bad_capacity
    with pytest.raises(ValidationError):
        ForecastToOrdersRequest(**payload)


def test_empty_items_list_raises():
    """items list must be non-empty."""
    payload = _base_payload()
    payload["items"] = []
    with pytest.raises(ValidationError):
        ForecastToOrdersRequest(**payload)


@pytest.mark.parametrize("bad_ratio", [-0.01, 1.01, 5.0])
def test_service_floor_ratio_out_of_range_raises(bad_ratio):
    """service_floor_ratio must lie in [0.0, 1.0]."""
    payload = _base_payload()
    payload["service_floor_ratio"] = bad_ratio
    with pytest.raises(ValidationError):
        ForecastToOrdersRequest(**payload)
