import os.path
from typing import List

from requests import Session
from .base import BaseModelApi
from seclea_ai.lib.seclea_utils.object_management.mixin import Organization, Project, Dataset, DatasetTransformation, \
    Model, \
    ModelState, TrainingRun
from .base import BaseModel


class ProjectApi(BaseModelApi):
    model_url = 'projects'
    model = Project
    file_keys = []


class OrganizationApi(BaseModelApi):
    model_url = 'organization'
    model = Organization
    file_keys = []


class DatasetApi(BaseModelApi):
    model_url = 'datasets'
    model = Dataset
    file_keys = ['dataset']


class DatasetTransformationApi(BaseModelApi):
    model_url = 'dataset-transformations'
    model = DatasetTransformation
    file_keys = []


class TrainingRunApi(BaseModelApi):
    model_url = 'training-runs'
    model = TrainingRun
    file_keys = []


class ModelApi(BaseModelApi):
    model_url = 'models'
    model = Model
    file_keys = []


class ModelStateApi(BaseModelApi):
    model_url = 'model-states'
    model = ModelState
    file_keys = []
