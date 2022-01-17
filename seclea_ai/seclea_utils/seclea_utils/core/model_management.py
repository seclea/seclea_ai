from abc import ABC, abstractmethod
from typing import Any

from seclea_utils.core import DataManager


class ModelManager(ABC):
    def __init__(self, data_manager: DataManager):
        self.data_manager = data_manager

    @abstractmethod
    def save_model(self, model: Any, reference: str) -> str:
        """
        Saves a model at the specified reference.
        :param model: model to save
        :param reference: where to save it
        :return: Where it was saved.
        """
        pass

    @abstractmethod
    def load_model(self, reference: str):
        """
        Loads a model from the specified reference.
        :param reference:
        :return: The model.
        """
        pass

    @staticmethod
    @abstractmethod
    def get_params(model):
        """
        Gets the model parameters for a given model.
        """
        pass
