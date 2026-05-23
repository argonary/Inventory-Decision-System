from pydantic import BaseModel, Field, field_validator
from typing import List, Literal, Optional


# =====================================================
# Request schemas
# =====================================================

class BatchItem(BaseModel):
    item_nbr: int = Field(
        ...,
        description="SKU identifier",
        json_schema_extra={"example": 769314},
    )
    onpromotion: bool = Field(
        ...,
        description="Whether SKU is on promotion",
        json_schema_extra={"example": True},
    )


class ForecastToOrdersRequest(BaseModel):
    date: str = Field(
        ...,
        description="Decision date (YYYY-MM-DD)",
        json_schema_extra={"example": "2016-04-21"},
    )
    store_nbr: int = Field(
        ...,
        description="Store number",
        json_schema_extra={"example": 44},
    )
    service_level: str = Field(
        ...,
        description="Quantile service level: 'p90' or 'p95' (case-insensitive)",
        json_schema_extra={"example": "p90"},
    )
    items: List[BatchItem] = Field(
        ...,
        description="List of SKUs to consider",
    )

    capacity_units: int = Field(
        ...,
        gt=0,
        description="Maximum total order quantity (capacity cap)",
        json_schema_extra={"example": 100},
    )
    service_floor_ratio: Optional[float] = Field(
        0.0,
        ge=0.0,
        le=1.0,
        description="Minimum fraction of forecast per SKU",
        json_schema_extra={"example": 0.0},
    )
    perishable_weight: Optional[float] = Field(
        1.0,
        gt=0.0,
        description="Weight multiplier for perishable items",
        json_schema_extra={"example": 1.2},
    )
    optimizer: Literal["proportional", "lp"] = Field(
        "proportional",
        description="Optimizer backend: 'proportional' (default) or 'lp'",
        json_schema_extra={"example": "proportional"},
    )

    @field_validator("service_level")
    @classmethod
    def _validate_service_level(cls, v):
        if not isinstance(v, str):
            raise ValueError("service_level must be 'p90' or 'p95' (case-insensitive)")
        normalized = v.strip().lower()
        if normalized not in ("p90", "p95"):
            raise ValueError("service_level must be 'p90' or 'p95' (case-insensitive)")
        return normalized

    @field_validator("capacity_units")
    @classmethod
    def _validate_capacity_units(cls, v):
        if v <= 0:
            raise ValueError("capacity_units must be a positive integer greater than 0")
        return v

    @field_validator("items")
    @classmethod
    def _validate_items_non_empty(cls, v):
        if not v:
            raise ValueError("items list must not be empty")
        return v

    @field_validator("service_floor_ratio")
    @classmethod
    def _validate_service_floor_ratio(cls, v):
        if v is None:
            return v
        if not (0.0 <= v <= 1.0):
            raise ValueError("service_floor_ratio must be between 0.0 and 1.0 inclusive")
        return v


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
    service_level: str
    capacity_units: int
    fill_capacity: bool

    model_version: str = Field(
        ...,
        description="Model version used for inference",
    )
    dataset_mode: str = Field(
        ...,
        description="Dataset mode used by API (train or test)",
        json_schema_extra={"example": "test"},
    )
    snapshot: str = Field(
        ...,
        description="Featured snapshot file used for inference",
        json_schema_extra={"example": "favorita_test_featured_2016Q1.parquet"},
    )

    summary: ForecastSummary
    results: List[ForecastResult]
    not_found: List[int] = Field(
        default_factory=list,
        description="Requested item_nbr values not present in the snapshot for the given store/date",
    )
