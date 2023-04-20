from pathlib import Path

from peewee import Model, SqliteDatabase

# TODO improve auth and pragmas etc.
db = SqliteDatabase(
    Path.home() / ".seclea" / "seclea_ai.db",
    thread_safe=True,
    pragmas={"journal_mode": "wal"},
)


class BaseModel(Model):
    class Meta:
        database = db
