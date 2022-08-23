import datetime
import json
from enum import Enum
from json import JSONDecodeError
from pathlib import Path

from peewee import CharField, DateTimeField, Field, IntegerField, Model, SqliteDatabase

# TODO improve auth and pragmas etc.
db = SqliteDatabase(
    Path.home() / ".seclea" / "seclea_ai.db",
    thread_safe=True,
    pragmas={"journal_mode": "wal"},
)


class RecordStatus(Enum):
    IN_MEMORY = "in_memory"
    STORED = "stored"
    SENT = "sent"
    STORE_FAIL = "store_fail"
    SEND_FAIL = "send_fail"


class JsonField(Field):
    def db_value(self, value):
        return json.dumps(value)

    def python_value(self, value):
        return json.loads(value)


class BaseModel(Model):
    class Meta:
        database = db


# TODO rethink this - may be better split up
class Record(BaseModel):
    remote_id = IntegerField(null=True)  # TODO this may change to string for uuids
    metadata = JsonField(null=False, default=dict())
    object_ser = JsonField(null=False, default=dict())
    dependencies = JsonField(null=False, default=list())  # this will be a list of ids
    status = CharField()  # TODO change to enum
    timestamp = DateTimeField(default=datetime.datetime.now)
    path: CharField()


class AuthService(BaseModel):
    key = CharField()
    value = CharField()


db.connect()
db.create_tables([Record, AuthService])
db.close()
