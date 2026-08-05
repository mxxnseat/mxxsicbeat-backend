from io import BytesIO

import pytest
from httpx import ASGITransport, AsyncClient
from mongomock_motor import AsyncMongoMockClient

from app.core.config import Config, get_config
from app.core.db import get_db
from app.domains.maps.configs.storage import MapsStorageConfig, get_maps_storage_config
from app.domains.maps.jobs.queues.queue import get_flow_producer
from app.domains.maps.services.storage import get_maps_storage
from app.main import create_app


class FakeStorage:
    """Stands in for MapsStorage - same key + fileobj surface, backed by an in-memory dict
    instead of MinIO."""

    def __init__(self, config: MapsStorageConfig | None = None) -> None:
        self.objects: dict[str, bytes] = {}
        self._config = config or MapsStorageConfig(_env_file=None)

    async def ensure_bucket(self) -> None:
        return None

    async def upload(self, key: str, fileobj: BytesIO) -> None:
        self.objects[key] = fileobj.read()

    async def download(self, key: str, destination) -> None:
        destination.write_bytes(self.objects[key])

    def original_key(self, job_id: str, filename: str) -> str:
        return self._config.original_key(job_id, filename)

    def drum_key(self, job_id: str) -> str:
        return self._config.drum_key(job_id)

    def melody_key(self, job_id: str) -> str:
        return self._config.melody_key(job_id)

    def audio_url(self, object_key: str | None) -> str | None:
        base_url = self._config.public_base_url
        if not base_url or not object_key:
            return None
        return f"{base_url.rstrip('/')}/{object_key}"


class FakeFlowProducer:
    def __init__(self) -> None:
        self.added_flows: list[dict] = []

    async def add(self, flow: dict, opts: dict | None = None) -> dict:
        self.added_flows.append(flow)
        return flow


@pytest.fixture
def fake_db():
    client = AsyncMongoMockClient()
    return client.get_database("test")


@pytest.fixture
def test_maps_storage_config() -> MapsStorageConfig:
    return MapsStorageConfig(_env_file=None)


@pytest.fixture
def fake_storage(test_maps_storage_config) -> FakeStorage:
    return FakeStorage(test_maps_storage_config)


@pytest.fixture
def fake_flow_producer() -> FakeFlowProducer:
    return FakeFlowProducer()


@pytest.fixture
def test_config() -> Config:
    return Config(_env_file=None)


@pytest.fixture
async def client(fake_db, fake_storage, fake_flow_producer, test_config, test_maps_storage_config):
    app = create_app()
    app.dependency_overrides[get_db] = lambda: fake_db
    app.dependency_overrides[get_maps_storage] = lambda: fake_storage
    app.dependency_overrides[get_maps_storage_config] = lambda: test_maps_storage_config
    app.dependency_overrides[get_flow_producer] = lambda: fake_flow_producer
    app.dependency_overrides[get_config] = lambda: test_config

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as async_client:
        yield async_client
