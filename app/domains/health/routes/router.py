from fastapi import APIRouter, Depends

from app.domains.health.services.service import HealthService, get_health_service

router = APIRouter(tags=["health"])


@router.get("/health")
async def health() -> dict:
    """Liveness: the process is up. No dependency checks."""
    return {"status": "ok"}


@router.get("/health/ready")
async def readiness(service: HealthService = Depends(get_health_service)) -> dict:
    """Readiness: can we actually reach Mongo and Redis."""
    return await service.check_readiness()
