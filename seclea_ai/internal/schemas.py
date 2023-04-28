from __future__ import annotations

from typing import Optional, Dict, List, Any
from uuid import UUID

import peewee


# the idea would be to have an interface to the persistable entities that separates them
# and allows us to pass them around more easily.

# maybe use pydantic models.
from pydantic import BaseModel
from pydantic.utils import GetterDict


class PeeweeGetterDict(GetterDict):
    def get(self, key: Any, default: Any = None):
        res = getattr(self._obj, key, default)
        if isinstance(res, peewee.ModelSelect):
            return list(res)
        return res


class ProjectSchema(BaseModel):

    uuid: Optional[UUID] = None

    name: str
    description: str = "Please update project description"
    # TODO add validators on length etc.

    class Config:
        orm_mode = True
        getter_dict = PeeweeGetterDict


class ProjectDBSchema(ProjectSchema):

    id: int

    class Config:
        orm_mode = True
        getter_dict = PeeweeGetterDict


class DatasetSchema(BaseModel):

    uuid: Optional[UUID] = None

    name: str
    hash: str
    metadata: Dict
    dataset: Optional[str] = None

    project: ProjectSchema
    parent: Optional[DatasetSchema] = None
    # TODO add validators

    class Config:
        orm_mode = True
        getter_dict = PeeweeGetterDict


class DatasetTransformationSchema(BaseModel):

    uuid: Optional[UUID] = None

    name: str
    code_raw: str
    code_encoded: str

    dataset: DatasetSchema

    class Config:
        orm_mode = True
        getter_dict = PeeweeGetterDict


class ModelSchema(BaseModel):

    uuid: Optional[UUID] = None

    name: str
    framework: str
    # TODO add validation

    class Config:
        orm_mode = True
        getter_dict = PeeweeGetterDict


class TrainingRunSchema(BaseModel):
    uuid: Optional[UUID] = None

    name: str
    metadata: Dict
    params: Dict

    project: ProjectSchema
    model: ModelSchema
    datasets: List[DatasetSchema]

    class Config:
        orm_mode = True
        getter_dict = PeeweeGetterDict


class ModelStateSchema(BaseModel):
    uuid: Optional[UUID] = None

    sequence_num: int
    state: Optional[str]

    training_run: TrainingRunSchema

    class Config:
        orm_mode = True
        getter_dict = PeeweeGetterDict
