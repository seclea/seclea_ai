from unittest import TestCase

import xgboost
from xgboost import Booster

from seclea_utils.core import get_module, get_type


class TestImports(TestCase):
    def test_get_module(self):
        xgb = get_module("xgboost")
        booster = xgb.Booster()
        self.assertTrue(isinstance(booster, Booster), "Should be xgboost Booster but is not.")

    def test_get_module_fail(self):
        django = get_module("django")
        self.assertIsNone(
            django, "Django is imported but should be None. Check in clean environment."
        )

    def test_get_module_fail_required(self):
        with self.assertRaises(ValueError):
            _ = get_module("django", "Django required")

    def test_get_type(self):
        booster = xgboost.Booster()
        t = get_type(booster)
        self.assertEqual(t, "xgboost.core.Booster")
