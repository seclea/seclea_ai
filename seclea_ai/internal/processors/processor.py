import os.path
from abc import ABC
from pathlib import Path

from peewee import SqliteDatabase


def _assemble_key(record) -> str:
    return f"{record['username']}-{record['project_id']}-{record['entity_id']}"


class Processor(ABC):
    cache_dir: str = '.'

    def __init__(self, cache_dir, **kwargs):
        os.makedirs(cache_dir, exist_ok=True)
        self.cache_dir = cache_dir
