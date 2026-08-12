from fastapi import APIRouter, Depends

from app.domains.health.services.service import HealthService, get_health_service

router = APIRouter(tags=["health"])


@router.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@router.get("/health/ready")
async def readiness(service: HealthService = Depends(get_health_service)) -> dict:
    return await service.check_readiness()
