import datetime
import os
import time
import uuid
from unittest import TestCase

import pandas as pd

from seclea_ai import SecleaAI
from seclea_ai.lib.seclea_utils.object_management import Tracked

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
folder_path = os.path.join(base_dir, "test_integration_portal")


class TestImageDatasetUpload(TestCase):
    """
    Monolithic testing of the Seclea AI file
    order of functions is preserved.

    NOTE: a random project is named on each tests this is to
    pseudo reset the database.

    Please reset database upon completing work

    Workflow to reset db on each test run to be investigated.
    """

    def setUp(self) -> None:
        self.controller = SecleaAI(
            project_name=f"test-project-{uuid.uuid4()}",
            organization="Onespan",
            platform_url="http://localhost:8000",
            auth_url="http://localhost:8010",
            username="admin",
            password="asdf",
        )

    def step_1_upload_dataset(self):
        self.df1 = Tracked(pd.read_csv(f"{folder_path}/adult_data.csv", index_col=0))
        self.df1_name = "Census dataset"
        self.df1_meta = {
            "outcome_name": "income-per-year",
            "favourable_outcome": ">50k",
            "unfavourable_outcome": "<=50k",
            "continuous_features": [
                "age",
                "fnlwgt",
                "education-num",
                "capital-gain",
                "capital-loss",
                "hours-per-week",
            ],
        }
        self.controller.upload_dataset(self.df1)
        self.controller.complete()

    def _steps(self):
        for name in dir(self):  # dir() result is implicitly sorted
            if name.startswith("step"):
                yield name, getattr(self, name)

    def test_steps(self):
        for name, step in self._steps():
            try:
                step()
                print("STEP COMPLETE")
            except Exception as e:
                self.fail(f"{step} failed ({type(e)}: {e})")
