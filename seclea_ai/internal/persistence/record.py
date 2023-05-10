import datetime
from enum import Enum

from peewee import IntegerField, DateTimeField

from .db import BaseModel
from .fields import EnumField


# TODO remove this after restructure of Record.
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


class Record(BaseModel):
    status = EnumField(enum_class=RecordStatus)
    size = IntegerField(null=False, default=0)
    created_timestamp = DateTimeField(default=datetime.datetime.now)
