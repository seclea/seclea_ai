import datetime
from enum import Enum

from peewee import IntegerField, CharField, DateTimeField

from .db import BaseModel
from .fields import JsonField, EnumField


class RecordEntity(Enum):
    DATASET = "dataset"
    DATASET_TRANSFORMATION = "transformation"
    TRAINING_RUN = "training_run"
    MODEL_STATE = "model_state"


class RecordStatus(Enum):
    IN_MEMORY = "in_memory"
    STORED = "stored"
    SENT = "sent"
    STORE_FAIL = "store_fail"
    SEND_FAIL = "send_fail"


# TODO rethink this - may be better split up
class Record(BaseModel):

    project_id = CharField()
    name = CharField(null=True)
    remote_id = IntegerField(null=True)  # TODO this may change to string for uuids
    entity = EnumField(
        enum_class=RecordEntity
    )  # TODO remove or convert to ForeignKey - here for debugging for now.
    key = CharField(null=True)  # mainly for tracking datasets and training runs may need to remove
    dependencies = JsonField(null=True)  # this will be a list of ids
    status = EnumField(enum_class=RecordStatus)
    created_timestamp = DateTimeField(default=datetime.datetime.now)
    # only used for datasets and modelstates.
    path = CharField(null=True)
    size = IntegerField(null=False, default=0)
    # only used for datasets - probably need to factor out a lot of this.
    dataset_metadata = JsonField(null=True)
