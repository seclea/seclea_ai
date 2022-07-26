import datetime
import json
from enum import Enum
from json import JSONDecodeError
from pathlib import Path

from peewee import CharField, DateTimeField, Field, IntegerField, Model, SqliteDatabase

# TODO improve auth and pragmas etc.
db = SqliteDatabase(Path.home() / ".seclea" / "seclea_ai.db", thread_safe=True)


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
        try:
            value = json.loads(value)
        except JSONDecodeError:
            value = None
        finally:
            return value


class BaseModel(Model):
    class Meta:
        database = db


# TODO rethink this - may be better split up
class Record(BaseModel):

    remote_id = IntegerField(null=True)  # TODO this may change to string for uuids
    entity = CharField(
        null=True
    )  # TODO remove or convert to ForeignKey - here for debugging for now.
    key = CharField(null=True)  # mainly for tracking datasets and training runs may need to remove
    dependencies = JsonField(null=True)  # this will be a list of ids
    status = CharField()  # TODO change to enum
    timestamp = DateTimeField(default=datetime.datetime.now)
    # only used for datasets and modelstates.
    path = CharField(null=True)
    # only used for datasets - probably need to factor out a lot of this.
    dataset_metadata = JsonField(null=True)


class AuthService(BaseModel):

    key = CharField()
    value = CharField()


db.connect()
db.create_tables([Record, AuthService])
db.close()
