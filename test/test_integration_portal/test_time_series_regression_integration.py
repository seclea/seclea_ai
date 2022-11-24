import os
import uuid
from unittest import TestCase

import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

from seclea_ai import SecleaAI
from seclea_ai.transformations import DatasetTransformation

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
folder_path = os.path.join(base_dir, "test_integration_portal")


class TestIntegrationTimeSeriesClassification(TestCase):
    """
    Monolithic testing of the Seclea AI file
    order of functions is preserved.

    NOTE: a random project is named on each tests this is to
    pseudo reset the database.

    Please reset database upon completing work

    Workflow to reset db on each test run to be investigated.
    """

    def step_0_project_setup(self):
        self.password = "2o4K@Vc1R(V0BpP?jVZ!"  # nosec
        self.username = "janetomlins"  # nosec
        self.organization = "Interspot"
        self.project_name = f"test-project-{uuid.uuid4()}"
        self.portal_url = "http://localhost:8000"
        self.auth_url = "http://localhost:8010"
        self.controller = SecleaAI(
            self.project_name,
            self.organization,
            self.portal_url,
            self.auth_url,
            username=self.username,
            password=self.password,
        )

    def step_1_upload_dataset(self):
        self.sample_df = pd.read_csv(f"{folder_path}/energydata_categorical.csv", index_col="date")
        self.sample_df_name = "Energy Data"
        self.sample_df_meta = {
            "outcome_name": "appliances",
            "continuous_features": [
                "lights",
                "T1",
                "T1",
                "RH_1",
                "T2",
                "RH_2",
                "T3",
                "RH_3",
                "T4",
                "RH_4",
                "T5",
                "RH_5",
                "T6",
                "RH_6",
                "T7",
                "RH_7",
                "T8",
                "RH_8",
                "T9",
                "RH_9",
                "T_out",
                "Press_mm_hg",
                "RH_out",
                "Windspeed",
                "Visibility",
                "Tdewpoint",
                "rv1",
                "rv2",
            ],
        }
        self.controller.upload_dataset(self.sample_df, self.sample_df_name, self.sample_df_meta)

    def step_2_define_transformations(self):
        def get_samples_labels(df, output_col):
            X = df.drop(output_col, axis=1)
            y = df[output_col]

            return X, y

        def get_test_train_splits(X, y, test_size, random_state):

            return train_test_split(
                X, y, test_size=test_size, stratify=y, random_state=random_state
            )
            # returns X_train, X_test, y_train, y_test

        ##############################

        output_col = "appliances"
        X, y = get_samples_labels(self.sample_df, output_col=output_col)

        test_size = 0.2
        random_state = 42
        self.X_train, self.X_test, self.y_train, self.y_test = get_test_train_splits(
            X, y, test_size=test_size, random_state=random_state
        )

        self.transformations_train = [
            DatasetTransformation(
                get_test_train_splits,
                {"X": X, "y": y},
                {
                    "test_size": 0.2,
                    "random_state": 42,
                    # this becomes v important bc we are re-running functions for uploading different branches...
                },
                ["X", None, "y", None],
                split="train",
            ),
        ]

        # upload dataset here
        self.controller.upload_dataset_split(
            X=self.X_train,
            y=self.y_train,
            dataset_name=f"{self.sample_df_name} - Train",
            metadata={},
            transformations=self.transformations_train,
        )

        self.complicated_transformations_test = [
            DatasetTransformation(
                get_test_train_splits,
                {"X": X, "y": y},
                {
                    "test_size": 0.2,
                    "random_state": 42,
                },
                [None, "X", None, "y"],
                split="test",
            ),
        ]

        # upload dataset here
        self.controller.upload_dataset_split(
            X=self.X_test,
            y=self.y_test,
            dataset_name=f"{self.sample_df_name} - Test",
            metadata={},
            transformations=self.complicated_transformations_test,
        )

    def step_3_upload_training_run_4(self):
        # define model
        # Train
        import xgboost as xgb
        model = xgb.XGBRegressor("reg:linear", random_state=42)
        model.fit(self.X_train, self.y_train)
        # preds = model.predict(self.X_test_scaled)

        self.controller.upload_training_run_split(
            model,
            X_train=self.X_train,
            y_train=self.y_train,
            X_test=self.X_test,
            y_test=self.y_test,
        )

    def step_3_upload_training_run_5(self):
        # define model

        from lightgbm import LGBMRegressor

        # from sklearn.metrics import accuracy_score
        # Train
        model = LGBMRegressor(random_state=5)
        model.fit(self.X_train, self.y_train)
        # preds = model.predict(self.X_test_scaled)

        self.controller.upload_training_run_split(
            model,
            X_train=self.X_train,
            y_train=self.y_train,
            X_test=self.X_test,
            y_test=self.y_test,
        )

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
