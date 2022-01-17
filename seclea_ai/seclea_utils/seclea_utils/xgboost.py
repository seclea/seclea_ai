from typing import Any, Dict

import xgboost as xgb

from seclea_utils.core import CustomNamedTemporaryFile, DataManager, ModelManager


class XGBoostModelManager(ModelManager):
    def __init__(self, data_manager: DataManager):
        super(XGBoostModelManager, self).__init__(data_manager)

    def save_model(self, model: Any, reference: str) -> str:
        # save to temp file then use manager to store.
        with CustomNamedTemporaryFile() as temp:
            # this could be from SKlearnAPI or LearningAPI which have significant differences
            model.save_model(temp.name)
            with open(temp.name, "rb") as read_temp:
                return self.data_manager.save_object(read_temp.read(), reference)

    def load_model(self, reference: str) -> Any:
        """
        Loads a stored XGBoost model. Note this will always return a Booster (LearningAPI model) even if the original
        model was an SKLearn model. This will impact the methods available on the returned model.
        :param reference:
        :return: XGBoost.Booster model.
        """
        # need to know what kind of model -
        with CustomNamedTemporaryFile() as temp:
            data = self.data_manager.load_object(reference)
            temp.write(data)
            temp.flush()
            model = (
                xgb.Booster()
            )  # TODO need to be careful about customer usage - ie. do they use the best iteration for their model or not....
            model.load_model(temp.name)
        return model

    @staticmethod
    def get_params(model) -> Dict:
        """
        Extracts the parameters of the model.
        :param model: The model
        """

        return model.save_config()
