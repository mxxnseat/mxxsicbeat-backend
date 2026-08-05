"""Applies pending MongoDB migrations.

Each migration is a module in this package named `NNNN_description.py` exposing an
async `up(db)` function. Applied migrations are recorded by filename stem in the
`_migrations` collection so re-running this is a no-op once everything is applied.

Migrations run forward-only - there's no `down()`. To reverse one, write a new migration
that undoes it.

Usage:
    uv run python -m migrations.runner
"""

import asyncio
import importlib.util
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType

from motor.motor_asyncio import AsyncIOMotorDatabase

from app.core.config import get_config
from app.core.db import create_mongo_client, get_database
from app.core.logging import configure_logging, get_logger

logger = get_logger(__name__)

MIGRATIONS_DIR = Path(__file__).parent
MIGRATIONS_COLLECTION = "_migrations"


def _discover() -> list[Path]:
    return sorted(MIGRATIONS_DIR.glob("[0-9]*.py"))


def _load(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


async def run_migrations(db: AsyncIOMotorDatabase) -> list[str]:
    """Applies every not-yet-recorded migration in filename order, returning the names applied."""
    already_applied = {doc["_id"] async for doc in db[MIGRATIONS_COLLECTION].find({}, {"_id": 1})}

    applied_now = []
    for path in _discover():
        name = path.stem
        if name in already_applied:
            continue

        module = _load(path)
        logger.info("migration.applying", name=name)
        await module.up(db)
        await db[MIGRATIONS_COLLECTION].insert_one({"_id": name, "applied_at": datetime.now(UTC)})
        logger.info("migration.applied", name=name)
        applied_now.append(name)

    return applied_now


async def main() -> None:
    config = get_config()
    configure_logging(config.log_level)

    client = create_mongo_client(config)
    db = get_database(client, config)
    try:
        applied = await run_migrations(db)
    finally:
        client.close()

    if applied:
        logger.info("migrations.done", count=len(applied), applied=applied)
    else:
        logger.info("migrations.done", count=0)


if __name__ == "__main__":
    asyncio.run(main())
