from requests import Session
from .base import BaseModelApi
from seclea_ai.lib.seclea_utils.object_management.mixin import Project, Dataset, DatasetTransformation, Model, \
    ModelState, TrainingRun


class ProjectApi(BaseModelApi):
    model_url = 'projects/'
    model = Project


class DatasetApi(BaseModelApi):
    model_url = 'datasets/'
    model = Dataset


class DatasetTransformationApi(BaseModelApi):
    model_url = 'dataset-transformations/'
    model = DatasetTransformation


class TrainingRunApi(BaseModelApi):
    model_url = 'training-runs/'
    model = TrainingRun


class ModelApi(BaseModelApi):
    model_url = 'models/'
    model = Model


class ModelStateApi(BaseModelApi):
    model_url = 'model-states/'
    model = ModelState
