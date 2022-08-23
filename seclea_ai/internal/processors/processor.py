import os.path
from abc import ABC
from pathlib import Path

from peewee import SqliteDatabase


def _assemble_key(record) -> str:
    return f"{record['username']}-{record['project_id']}-{record['entity_id']}"


class Processor(ABC):
    db_name: str = 'seclea_ai.db'
    cache_dir: str = '.'

    def __init__(self, cache_dir, **kwargs):
        os.makedirs(cache_dir, exist_ok=True)
        self.cache_dir = cache_dir
        self._db = SqliteDatabase(
            os.path.join(cache_dir, self.db_name),
            thread_safe=True,
            pragmas={"journal_mode": "wal"},
        )
