from fastapi import APIRouter, Depends

from app.api.routes.v1_catalog import router as catalog_router
from app.api.routes.v1_chat import router as chat_router
from app.api.routes.v1_recommendations import router as recommendations_router
from app.core.security import require_service_token

router = APIRouter(dependencies=[Depends(require_service_token)])
router.include_router(chat_router, prefix="/chat", tags=["chat"])
router.include_router(recommendations_router, prefix="/recommendations", tags=["recommendations"])
router.include_router(catalog_router, prefix="/catalog", tags=["catalog"])
