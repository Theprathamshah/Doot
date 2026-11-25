"""SQLite schema, connection management, and persistence repository."""

from doot.db.connection import open_connection
from doot.db.repository import DootRepository, InvalidStateTransitionError
from doot.db.schema import initialize_schema

__all__ = [
    "DootRepository",
    "InvalidStateTransitionError",
    "initialize_schema",
    "open_connection",
]
