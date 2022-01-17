from unittest import TestCase

from seclea_utils.core.data_management import FileManager
from seclea_utils.get_model_manager import (
    Frameworks,
    LightGBMModelManager,
    SKLearnModelManager,
    XGBoostModelManager,
    get_model_manager,
)


class Test(TestCase):
    def test_get_model_manager_framework_not_installed_exception(self):
        self.assertRaises(ValueError, get_model_manager, Frameworks.NOT_IMPORTED, FileManager())

    def test_get_model_manager_sklearn_installed(self):
        self.assertIsInstance(
            get_model_manager(Frameworks.SKLEARN, FileManager()),
            SKLearnModelManager,
            msg="Failed to instantiate framework",
        )

    def test_get_model_manager_xgboost_installed(self):
        self.assertIsInstance(
            get_model_manager(Frameworks.XGBOOST, FileManager()),
            XGBoostModelManager,
            msg="Failed to instantiate framework",
        )

    def test_get_model_manager_lightgbm_installed(self):
        self.assertIsInstance(
            get_model_manager(Frameworks.LIGHTGBM, FileManager()),
            LightGBMModelManager,
            msg="Failed to instantiate framework",
        )
