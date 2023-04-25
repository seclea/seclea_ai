from pathlib import Path

from peewee import SqliteDatabase

from .auth_credentials import AuthCredentials
from .models import Project, Dataset, DatasetTransformation, Model, TrainingRun, ModelState
from .record import Record


def init_tables():
    db = SqliteDatabase(
        Path.home() / ".seclea" / "seclea_ai.db",
        thread_safe=True,
        pragmas={"journal_mode": "wal"},
    )

    with db.atomic():
        db.create_tables(
            [
                Record,
                AuthCredentials,
                Project,
                Dataset,
                DatasetTransformation,
                Model,
                TrainingRun,
                ModelState,
            ]
        )


# TODO add migrations - maybe to add constraints to tables after creation?
