# Migrations

MongoDB index/schema changes, applied explicitly - never at api/worker startup.

```bash
uv run python -m migrations.runner
```

Applied migrations are recorded by filename in the `_migrations` collection, so re-running is a
no-op once everything is up to date. `scripts/dev.sh` runs this automatically before starting the
api/worker locally; in other environments (CI, deploys) run it once against the target database
before starting the app.

## Adding a migration

Create the next-numbered file in this folder, `NNNN_description.py`, exposing an async `up(db)`:

```python
from motor.motor_asyncio import AsyncIOMotorDatabase


async def up(db: AsyncIOMotorDatabase) -> None:
    await db.some_collection.create_index("some_field")
```

Migrations run forward-only, in filename order - there's no `down()`. To reverse one, write a new
migration that undoes it, rather than editing or deleting an already-applied file.
