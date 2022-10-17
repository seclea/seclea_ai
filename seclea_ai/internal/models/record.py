import datetime
from enum import Enum

from peewee import IntegerField, CharField, DateTimeField

from .db import BaseModel
from .fields import JsonField


class RecordStatus(Enum):
    IN_MEMORY = "in_memory"
    STORED = "stored"
    SENT = "sent"
    STORE_FAIL = "store_fail"
    SEND_FAIL = "send_fail"


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
    size = IntegerField(null=False, default=0)
    # only used for datasets - probably need to factor out a lot of this.
    dataset_metadata = JsonField(null=True)
