from app.domains.maps.dtos.internal import NewBeatmap
from app.domains.maps.repositories.repository import MapRepository


class MapJobService:
    def __init__(self, repository: MapRepository) -> None:
        self._repository = repository

    async def mark_processing(self, job_id: str) -> None:
        await self._repository.mark_job_processing(job_id)

    async def mark_failed(self, job_id: str, error: str) -> None:
        await self._repository.mark_job_failed(job_id, error)

    async def finalize_beatmap(self, beatmap: NewBeatmap) -> str:
        beatmap_id = await self._repository.insert_beatmap(beatmap)
        await self._repository.mark_job_completed(beatmap.job_id, beatmap_id)
        return beatmap_id
