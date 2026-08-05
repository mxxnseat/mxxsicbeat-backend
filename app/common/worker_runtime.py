import asyncio
import signal
from collections.abc import Awaitable, Callable

from bullmq import Job, Worker

from app.core.config import Config
from app.core.logging import get_logger

logger = get_logger(__name__)

Processor = Callable[[Job, str], Awaitable[dict]]


class WorkerRuntime:
    """Owns a single BullMQ `Worker`'s lifecycle for a queue + processor pair. Has no knowledge
    of Mongo, repositories, or storage - each handler process builds its own dependencies and
    hands this the finished processor plus whatever cleanup it needs on shutdown, so the same
    runtime is reused unchanged across every handler process."""

    def __init__(
        self,
        config: Config,
        queue_name: str,
        processor: Processor,
        *,
        on_stop: list[Callable[[], None]] | None = None,
    ) -> None:
        self._config = config
        self._queue_name = queue_name
        self._processor = processor
        self._on_stop = on_stop or []
        self._bullmq_worker: Worker | None = None

    async def start(self) -> None:
        self._bullmq_worker = Worker(
            self._queue_name, self._processor, {"connection": self._config.redis_url}
        )
        logger.info("worker.started", queue=self._queue_name)

    async def run_until_stopped(self) -> None:
        stop_event = asyncio.Event()
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, stop_event.set)
        await stop_event.wait()

    async def stop(self) -> None:
        logger.info("worker.stopping", queue=self._queue_name)
        if self._bullmq_worker is not None:
            await self._bullmq_worker.close()
        for close in self._on_stop:
            close()
