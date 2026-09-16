from fastapi import APIRouter

from app.api.routes.health import router as health_router
from app.api.routes.v1 import router as v1_router

api_router = APIRouter()
api_router.include_router(health_router)
api_router.include_router(v1_router, prefix="/v1")
