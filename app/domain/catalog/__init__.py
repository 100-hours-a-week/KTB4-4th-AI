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

__all__ = [
    "DOCUMENT_VERSION",
    "MAX_DOCUMENT_LENGTH",
    "DocumentEvidence",
    "DocumentSourceProduct",
    "DocumentValidationError",
    "PriceBand",
    "ProductDocument",
    "ProductDocumentSet",
    "price_band",
    "validate_document_text",
]
