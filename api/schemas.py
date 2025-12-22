from pydantic import BaseModel, Field
from typing import List, Optional


# =====================================================
# Request schemas
# =====================================================

class BatchItem(BaseModel):
    item_nbr: int = Field(
        ...,
        description="SKU identifier",
        example=769314,
    )
    onpromotion: bool = Field(
        ...,
        description="Whether SKU is on promotion",
        example=True,
    )


class ForecastToOrdersRequest(BaseModel):
    date: str = Field(
        ...,
        description="Decision date (YYYY-MM-DD)",
        example="2016-04-21",
    )
    store_nbr: int = Field(
        ...,
        description="Store number",
        example=44,
    )
    service_level: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Quantile service level (e.g. 0.9, 0.95)",
        example=0.9,
    )
    items: List[BatchItem] = Field(
        ...,
        description="List of SKUs to consider",
    )

    capacity_units: int = Field(
        ...,
        gt=0,
        description="Maximum total order quantity (capacity cap)",
        example=100,
    )
    service_floor_ratio: Optional[float] = Field(
        0.0,
        ge=0.0,
        le=1.0,
        description="Minimum fraction of forecast per SKU",
        example=0.0,
    )
    perishable_weight: Optional[float] = Field(
        1.0,
        gt=0.0,
        description="Weight multiplier for perishable items",
        example=1.2,
    )


# =====================================================
# Response schemas
# =====================================================

class ForecastResult(BaseModel):
    item_nbr: int = Field(..., description="SKU identifier")
    forecast: float = Field(..., description="Forecasted demand at chosen quantile")
    order_qty: int = Field(..., description="Allocated order quantity")


class ForecastSummary(BaseModel):
    total_forecast: float = Field(
        ...,
        description="Sum of forecasted demand across SKUs",
    )
    total_orders: int = Field(
        ...,
        description="Sum of allocated order quantities",
    )


class ForecastToOrdersResponse(BaseModel):
    store_nbr: int
    date: str
    service_level: float
    capacity_units: int
    fill_capacity: bool

    model_version: str = Field(
        ...,
        description="Model version used for inference",
    )
    dataset_mode: str = Field(
        ...,
        description="Dataset mode used by API (train or test)",
        example="test",
    )
    snapshot: str = Field(
        ...,
        description="Featured snapshot file used for inference",
        example="favorita_test_featured_2016Q1.parquet",
    )

    summary: ForecastSummary
    results: List[ForecastResult]
