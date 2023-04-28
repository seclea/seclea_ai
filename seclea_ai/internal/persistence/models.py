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
from .record import Record


class RecordedModel(BaseModel):

    record = ForeignKeyField(Record, null=True)


class Project(BaseModel):

    uuid = UUIDField(null=True)  # uuid - remote id

    name = CharField(max_length=250)
    description = CharField(default="Please update project description", max_length=5000)


class Dataset(RecordedModel):

    uuid = UUIDField(null=True)  # uuid - remote id

    name = CharField(max_length=256)
    hash = CharField(max_length=200)
    metadata = JsonField()
    dataset = CharField(max_length=400, null=True)  # file where stored locally

    project = ForeignKeyField(Project, backref="datasets")
    parent = ForeignKeyField("self", null=True, backref="children")

    # class Meta:
    #     constraints = [SQL("UNIQUE (project, hash)"), SQL("UNIQUE (project, name)")]


class DatasetTransformation(RecordedModel):

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


class TrainingRun(RecordedModel):

    uuid = UUIDField(null=False)  # uuid - remote id

    name = CharField(max_length=256)
    metadata = JsonField()
    params = JsonField()

    # TODO think about on_delete and on_update
    project = ForeignKeyField(Project, backref="training_runs")
    model = ForeignKeyField(Model, backref="training_runs")
    datasets = ManyToManyField(Dataset, backref="training_runs")


TrainingRunDataset = TrainingRun.datasets.get_through_model()


class ModelState(RecordedModel):

    uuid = UUIDField(null=True)

    sequence_num = IntegerField()
    state = CharField(max_length=400, null=True)  # file where stored locally

    training_run = ForeignKeyField(TrainingRun, backref="model_states")

    # TODO add constraints on FK in a way that still can be initialised.
    #  currently throws OperationalError - training_run column doesn't exist.
    # class Meta:
    #     constraints = [SQL("UNIQUE (training_run, sequence_num)")]
