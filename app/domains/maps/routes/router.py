from fastapi import APIRouter, Depends, Form, UploadFile, status

from app.domains.maps.dtos.api import Beatmap, MapJobCreateResponse, MapJobOut
from app.domains.maps.routes.validators.file_size import validate_file_size
from app.domains.maps.services.service import MapService, get_map_service

router = APIRouter(prefix="/api/v1/maps", tags=["maps"])


@router.post("", response_model=MapJobCreateResponse, status_code=status.HTTP_202_ACCEPTED)
async def create_map(
    file: UploadFile = Depends(validate_file_size),
    lane_count: int | None = Form(None, description="Number of lanes to spread notes across"),
    service: MapService = Depends(get_map_service),
) -> MapJobCreateResponse:
    return await service.create_map_job(file=file, lane_count=lane_count)


@router.get("", response_model=list[Beatmap])
async def list_beatmaps(service: MapService = Depends(get_map_service)) -> list[Beatmap]:
    return await service.list_beatmaps()


@router.get("/jobs/{job_id}", response_model=MapJobOut)
async def get_job(job_id: str, service: MapService = Depends(get_map_service)) -> MapJobOut:
    return await service.get_job_status(job_id)


@router.get("/{beatmap_id}", response_model=Beatmap)
async def get_beatmap(beatmap_id: str, service: MapService = Depends(get_map_service)) -> Beatmap:
    return await service.get_beatmap(beatmap_id)
