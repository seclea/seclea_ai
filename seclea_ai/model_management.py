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


def get_model_manager(framework, manager):
    frameworks = {
        "xgboost": XGBoostModelManager,
        "lightgbm": LightGBMModelManager,
        "sklearn": SKLearnModelManager,
    }
    if framework in frameworks:
        try:
            return frameworks[framework](manager)
        except TypeError:
            raise ImportError(f"{framework} is not installed. Please install and try again")
    else:
        raise ValueError(f"Framework not supported, must be one of {frameworks.keys()}")
