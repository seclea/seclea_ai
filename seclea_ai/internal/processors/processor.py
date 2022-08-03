from abc import ABC
from pathlib import Path

from peewee import SqliteDatabase


def _assemble_key(record) -> str:
    return f"{record['username']}-{record['project_id']}-{record['entity_id']}"


class Processor(ABC):
    def __init__(self, settings, **kwargs):
        self._settings = settings
        self._db = SqliteDatabase(
            Path.home() / ".seclea" / "seclea_ai.db",
            thread_safe=True,
            pragmas={"journal_mode": "wal"},
        )
