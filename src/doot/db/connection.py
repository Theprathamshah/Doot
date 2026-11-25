"""Async SQLite connection management."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import aiosqlite

from doot.db.schema import apply_pragmas


@asynccontextmanager
async def open_connection(db_path: Path) -> AsyncIterator[aiosqlite.Connection]:
    """Open an async SQLite connection with WAL, FK, and row factory configured."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    async with aiosqlite.connect(db_path) as conn:
        conn.row_factory = aiosqlite.Row
        await apply_pragmas(conn)
        yield conn
