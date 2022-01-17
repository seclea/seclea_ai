import copy
from typing import Any, Dict

import lightgbm as lgb

from seclea_utils.core import DataManager, ModelManager


class LightGBMModelManager(ModelManager):
    def __init__(self, data_manager: DataManager):
        super(LightGBMModelManager, self).__init__(data_manager)

    def save_model(self, model: Any, reference: str) -> str:
        # saves either the best or all iterations.. TODO think
        if isinstance(model, lgb.Booster):
            model_data = model.model_to_string().encode("utf-8")
        else:
            model_data = model.booster_.model_to_string().encode("utf-8")
        return self.data_manager.save_object(model_data, reference)

    def load_model(self, reference: str):
        """
        Loads a stored LightGBM model. Note this will always return a LightGBM Booster, even if the original model was
        an SKLearn model. This will impact the methods available on the returned model.
        :param reference:
        :return: LightGBM.Booster model
        """
        data = self.data_manager.load_object(reference).decode("utf-8")
        model = lgb.Booster(model_str=data)
        return model

    @staticmethod
    def get_params(model) -> Dict:
        """
        Extracts the parameters of the model.
        :param model: The model
        """

        return copy.deepcopy(model.params)
