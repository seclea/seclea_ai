from enum import Enum

from seclea_utils.core.data_management import DataManager

try:
    from seclea_utils.sklearn import SKLearnModelManager
except ValueError:
    SKLearnModelManager = None
try:
    from seclea_utils.lightgbm import LightGBMModelManager
except ValueError:
    LightGBMModelManager = None
except ModuleNotFoundError:
    LightGBMModelManager = None
try:
    from seclea_utils.xgboost import XGBoostModelManager
except ValueError:
    XGBoostModelManager = None
except ModuleNotFoundError:
    XGBoostModelManager = None


class Frameworks(Enum):
    XGBOOST = XGBoostModelManager
    LIGHTGBM = LightGBMModelManager
    SKLEARN = SKLearnModelManager
    NOT_IMPORTED = None


def get_model_manager(framework: Frameworks, data_manager: DataManager):
    if framework.value is None:
        raise ValueError("Framework not installed")
    else:
        return framework.value(data_manager)
