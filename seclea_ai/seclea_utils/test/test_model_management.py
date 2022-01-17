import os
import tempfile
from unittest import TestCase

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import GradientBoostingClassifier

from seclea_utils.core import FileManager
from seclea_utils.lightgbm import LightGBMModelManager
from seclea_utils.sklearn import SKLearnModelManager
from seclea_utils.xgboost import XGBoostModelManager


class TestSKLearnModelManager(TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        # read data
        data = pd.read_csv("test/example_files/filled_na_train.csv")
        label = data["isFraud"].copy(deep=True)
        data = data.drop("isFraud", axis=1)
        # setup training params
        params = {
            "learning_rate": 0.2,
            "max_depth": 10,
            "min_samples_leaf": 24,
            "max_leaf_nodes": 496,
        }
        # train model
        cls._booster_skl = GradientBoostingClassifier(**params)
        cls._booster_skl.fit(data, label)
        cls.sample = data.iloc[[0]].to_numpy()

    def setUp(self) -> None:
        self.manager = SKLearnModelManager(FileManager())

    def test_save_model(self):
        temp = os.path.join(tempfile.gettempdir(), os.urandom(24).hex())
        stored = self.manager.save_model(model=self._booster_skl, reference=temp)
        self.assertTrue(os.path.exists(stored))
        os.remove(stored)
        # os.remove(temp)  # uncomment if not using FileManager

    def test_load_model(self):
        temp = os.path.join(tempfile.gettempdir(), os.urandom(24).hex())
        stored = self.manager.save_model(model=self._booster_skl, reference=temp)
        model = self.manager.load_model(stored)
        # os.remove(new_file)
        # test that it executes as expected
        print(
            f"SKL - Loaded: {model.predict_proba(self.sample)} - Original: {self._booster_skl.predict_proba(self.sample)}"
        )
        self.assertTrue(
            np.allclose(
                model.predict_proba(self.sample), self._booster_skl.predict_proba(self.sample)
            )
        )
        os.remove(stored)
        # os.remove(temp)  # uncomment if not using FileManager


class TestXGBoostModelManager(TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        # read core
        data = pd.read_csv("test/example_files/filled_na_train.csv")
        label = data["isFraud"].copy(deep=True)
        data = data.drop("isFraud", axis=1)
        cls.data = data
        # load to XGBoost format
        dtrain = xgb.DMatrix(data=data, label=label)
        # setup training params
        params = dict(max_depth=2, eta=1, objective="binary:logistic", nthread=4, eval_metric="auc")
        num_rounds = 5
        # train model
        cls._booster = xgb.train(params=params, dtrain=dtrain, num_boost_round=num_rounds)
        cls._booster_skl = xgb.XGBClassifier(objective="binary:logistic", use_label_encoder=False)
        cls._booster_skl.fit(data, label)
        cls.sample = data.iloc[[0]].to_numpy()

    def setUp(self) -> None:
        self.manager = XGBoostModelManager(FileManager())

    def test_save_learning_api(self):
        temp = os.path.join(tempfile.gettempdir(), os.urandom(24).hex())
        stored = self.manager.save_model(model=self._booster, reference=temp)
        self.assertTrue(os.path.exists(stored))
        os.remove(stored)
        # os.remove(temp)  # uncomment if not using FileManager

    def test_load_from_learning_api(self):
        temp = os.path.join(tempfile.gettempdir(), os.urandom(24).hex())
        stored = self.manager.save_model(model=self._booster, reference=temp)
        model = self.manager.load_model(stored)
        # os.remove(new_file)
        # test that it executes as expected
        print(
            f"XGBoost from learning - Loaded: {model.inplace_predict(self.sample)} - Original: {self._booster.inplace_predict(self.sample)}"
        )
        self.assertTrue(
            np.allclose(
                model.inplace_predict(self.sample), self._booster.inplace_predict(self.sample)
            )
        )
        os.remove(stored)
        # os.remove(temp)  # uncomment if not using FileManager

    def test_save_sklearn_api(self):
        temp = os.path.join(tempfile.gettempdir(), os.urandom(24).hex())
        stored = self.manager.save_model(model=self._booster_skl, reference=temp)
        self.assertTrue(os.path.exists(stored))
        os.remove(stored)
        # os.remove(temp)  # uncomment if not using FileManager

    def test_load_from_sklearn_api(self):
        temp = os.path.join(tempfile.gettempdir(), os.urandom(24).hex())
        stored = self.manager.save_model(model=self._booster_skl, reference=temp)
        model = self.manager.load_model(stored)
        # os.remove(new_file)
        # test that it executes as expected
        print(
            f"XGBoost from SKL - Loaded: {model.inplace_predict(self.sample)} - Original: {np.delete(self._booster_skl.predict_proba(self.sample), 0)}"
        )
        self.assertTrue(
            np.allclose(
                model.inplace_predict(self.sample),
                np.delete(self._booster_skl.predict_proba(self.sample), 0),
            )
        )
        os.remove(stored)
        # os.remove(temp)  # uncomment if not using FileManager

    # maybe add DASK as well - for the future at least.


class TestLightGBMModelManager(TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        # read data
        data = pd.read_csv("test/example_files/filled_na_train.csv")
        label = data["isFraud"].copy(deep=True)
        data = data.drop("isFraud", axis=1)
        # load to XGBoost format
        dtrain = lgb.Dataset(data=data, label=label, free_raw_data=True)
        # setup training params
        params = dict(max_depth=2, eta=1, objective="binary", num_threads=4, metric="auc")
        num_rounds = 5
        # train model
        cls._booster = lgb.train(params=params, train_set=dtrain, num_boost_round=num_rounds)
        cls._booster_skl = lgb.LGBMClassifier(objective="binary")
        cls._booster_skl.fit(data, label)
        cls.sample = data.iloc[[0]]

    def setUp(self) -> None:
        self.manager = LightGBMModelManager(FileManager())

    def test_save_learning_api(self):
        temp = os.path.join(tempfile.gettempdir(), os.urandom(24).hex())
        stored = self.manager.save_model(model=self._booster, reference=temp)
        self.assertTrue(os.path.exists(stored))
        os.remove(stored)
        # os.remove(temp)  # uncomment if not using FileManager

    def test_load_from_learning_api(self):
        temp = os.path.join(tempfile.gettempdir(), os.urandom(24).hex())
        stored = self.manager.save_model(model=self._booster, reference=temp)
        model = self.manager.load_model(stored)
        # os.remove(new_file)
        # test that it executes as expected
        print(
            f"LGBM from learning - Loaded: {model.predict(self.sample)} - Original: {self._booster.predict(self.sample)}"
        )
        self.assertTrue(np.allclose(model.predict(self.sample), self._booster.predict(self.sample)))
        os.remove(stored)
        # os.remove(temp)  # uncomment if not using FileManager

    def test_save_sklearn_api(self):
        temp = os.path.join(tempfile.gettempdir(), os.urandom(24).hex())
        stored = self.manager.save_model(model=self._booster_skl, reference=temp)
        self.assertTrue(os.path.exists(stored))
        os.remove(stored)
        # os.remove(temp)  # uncomment if not using FileManager

    def test_load_from_sklearn_api(self):
        temp = os.path.join(tempfile.gettempdir(), os.urandom(24).hex())
        stored = self.manager.save_model(model=self._booster_skl, reference=temp)
        model = self.manager.load_model(stored)
        # os.remove(new_file)
        # test that it executes as expected
        print(
            f"LGBM from SKL Loaded: {model.predict(self.sample)} - Original: {np.delete(self._booster_skl.predict_proba(self.sample), 0)}"
        )
        self.assertTrue(
            np.allclose(
                model.predict(self.sample), np.delete(self._booster_skl.predict(self.sample), 0)
            )
        )
        os.remove(stored)
        # os.remove(temp)  # uncomment if not using FileManager
