import pytest

from api.schemas import BatchItem, ForecastToOrdersRequest


@pytest.fixture
def valid_request() -> ForecastToOrdersRequest:
    """Minimal valid ForecastToOrdersRequest for reuse across tests."""
    return ForecastToOrdersRequest(
        date="2016-04-21",
        store_nbr=44,
        service_level="p90",
        items=[
            BatchItem(item_nbr=769314, onpromotion=False),
            BatchItem(item_nbr=502331, onpromotion=True),
        ],
        capacity_units=100,
        service_floor_ratio=0.1,
        perishable_weight=1.2,
    )


@pytest.fixture
def simple_demand() -> dict:
    """Minimal demand dict suitable for optimizer tests."""
    return {1: 50.0, 2: 30.0, 3: 20.0}
