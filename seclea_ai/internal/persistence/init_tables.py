from pathlib import Path

from peewee import SqliteDatabase

from .auth_credentials import AuthCredentials
from .record import Record


def init_tables():
    db = SqliteDatabase(
        Path.home() / ".seclea" / "seclea_ai.db",
        thread_safe=True,
        pragmas={"journal_mode": "wal"},
    )

    db.connect()
    db.create_tables([Record, AuthCredentials])
    db.close()
