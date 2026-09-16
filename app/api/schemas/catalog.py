from datetime import datetime

from pydantic import Field

from app.api.schemas.common import ApiModel


class CoverageGap(ApiModel):
    node_id: str
    product_count: int = Field(ge=0)
    bands_covered: int = Field(ge=0)
    priority: str


class CatalogCoverageResponse(ApiModel):
    generated_at: datetime
    total_products: int = Field(ge=0)
    total_nodes: int = Field(ge=0)
    node_coverage: float = Field(ge=0.0, le=1.0)
    gaps: list[CoverageGap]
