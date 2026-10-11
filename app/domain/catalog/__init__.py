"""Catalog product and embedding-space rules."""

from app.domain.catalog.labels import (
    CATEGORY_LABELS,
    THEME_LABELS,
    CategorySource,
    GiftTheme,
    ProductCategory,
    is_gift_available,
    split_labels,
)
from app.domain.catalog.models import (
    DOCUMENT_VERSION,
    MAX_DOCUMENT_LENGTH,
    REQUIRED_SPACES,
    DocumentEvidence,
    DocumentSourceProduct,
    DocumentValidationError,
    PriceBand,
    ProductDocument,
    ProductDocumentSet,
    price_band,
    validate_document_text,
)
from app.domain.catalog.sync import ProcessingStatus, SourceProduct, SyncMode

__all__ = [
    "CATEGORY_LABELS",
    "DOCUMENT_VERSION",
    "REQUIRED_SPACES",
    "THEME_LABELS",
    "CategorySource",
    "GiftTheme",
    "ProductCategory",
    "is_gift_available",
    "split_labels",
    "MAX_DOCUMENT_LENGTH",
    "DocumentEvidence",
    "DocumentSourceProduct",
    "DocumentValidationError",
    "PriceBand",
    "ProcessingStatus",
    "ProductDocument",
    "ProductDocumentSet",
    "SourceProduct",
    "SyncMode",
    "price_band",
    "validate_document_text",
]
