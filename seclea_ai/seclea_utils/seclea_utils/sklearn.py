import pickle  # nosec
from typing import Any, Dict

from seclea_utils.core.data_management import DataManager
from seclea_utils.core.model_management import ModelManager


class SKLearnModelManager(ModelManager):
    def __init__(self, data_manager: DataManager):
        super(SKLearnModelManager, self).__init__(data_manager)

    def save_model(self, model: Any, reference: str) -> str:
        data = pickle.dumps(model)  # nosec
        return self.data_manager.save_object(data, reference)

    def load_model(self, reference: str):
        """
        Loads a stored SKLearn model. As the model is stored with pickle and a certain version of SKLearn, there
        may be inconsistencies where different versions of SKLearn are used for pickling and unpickling.
        :param reference:
        :return:
        """
        data = self.data_manager.load_object(reference)
        return pickle.loads(data)  # nosec

    @staticmethod
    def get_params(model) -> Dict:
        """
        Extracts the parameters of the model.
        :param model: The model
        """

        return model.get_params()
