"""Catalog product and embedding-space rules."""

from app.domain.catalog.models import (
    DOCUMENT_VERSION,
    MAX_DOCUMENT_LENGTH,
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
    "DOCUMENT_VERSION",
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
