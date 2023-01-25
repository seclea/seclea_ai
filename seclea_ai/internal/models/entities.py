from peewee import (
    IntegerField,
    CharField,
    UUIDField,
    TextField,
    ForeignKeyField,
    ManyToManyField,
    SQL,
)

from .db import BaseModel
from .fields import JsonField


class Project(BaseModel):

    uuid = UUIDField(null=True)  # uuid - remote id

    name = CharField(max_length=250)
    description = CharField(default="My project description", max_length=5000)


class Dataset(BaseModel):

    uuid = UUIDField(null=True)  # uuid - remote id

    name = CharField(max_length=256)
    hash = CharField(max_length=200)
    metadata = JsonField()
    dataset = CharField(max_length=400)  # file where stored locally

    project = ForeignKeyField(Project)
    parent = ForeignKeyField("self", null=True, backref="children")

    class Meta:
        constraints = [SQL("UNIQUE (project, hash)")]


class DatasetTransformation(BaseModel):

    uuid = UUIDField(null=True)  # uuid - remote id

    name = CharField(max_length=50)
    code_raw = TextField()
    code_encoded = TextField()

    dataset = ForeignKeyField(Dataset, backref="dataset_transformations")


class Model(BaseModel):

    uuid = UUIDField(null=True)  # uuid - remote id

    name = CharField(max_length=100)
    framework = CharField(max_length=100)

    class Meta:
        constraints = [SQL("UNIQUE (name, framework)")]


class TrainingRun(BaseModel):

    uuid = UUIDField(null=False)  # uuid - remote id

    name = CharField(max_length=256)
    metadata = JsonField()
    params = JsonField()

    project = ForeignKeyField(Project)
    model = ForeignKeyField(Model, backref="training_runs")
    datasets = ManyToManyField(Dataset)


class ModelState(BaseModel):

    uuid = UUIDField(null=True)

    sequence_num = IntegerField()
    state = CharField(max_length=400)  # file where stored locally

    training_run = ForeignKeyField(TrainingRun, backref="model_states")

    class Meta:
        constraints = [SQL("UNIQUE (training_run, sequence_num)")]
