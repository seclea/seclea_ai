import os
from unittest import TestCase

import pandas as pd
import pytest
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from seclea_ai import SecleaAI
from seclea_ai.transformations import DatasetTransformation

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
folder_path = os.path.join(base_dir, "test_integration_portal")


@pytest.mark.skipif(
    "not config.getoption('--run-slow')",
    reason="Only run when --run-slow is given",
)
class TestMultilabelDataIntegration(TestCase):
    """
    Monolithic testing of the Seclea AI file
    order of functions is preserved.

    NOTE: a random project is named on each tests this is to
    pseudo reset the database.

    Please reset database upon completing work

    Workflow to reset db on each test run to be investigated.
    """

    def step_0_project_setup(self):
        self.password = "asdf"  # nosec
        self.username = "admin-dev"  # nosec
        self.organization = "Onespan"
        self.project_name = "Multilabel Classification Project"
        self.portal_url = "http://localhost:8000"
        self.auth_url = "http://localhost:8010"
        self.controller = SecleaAI(
            project_name=self.project_name,
            organization=self.organization,
            platform_url=self.portal_url,
            auth_url=self.auth_url,
            username=self.username,
            password=self.password,
            create_project=True,
        )

    def step_1_upload_dataset(self):

        # multilabel
        self.multilabel_data = pd.read_csv(
            f"{folder_path}/data/energydata_multilabel.csv", index_col="date"
        )
        self.multilabel_outputs = ["appliances", "lights"]

        self.multilabel_name = "Energy dataset - multilabel"
        self.multilabel_metadata = {
            "outputs": self.multilabel_outputs,
            "continuous_features": [
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
        self.controller.upload_dataset(
            self.multilabel_data,
            dataset_name=self.multilabel_name,
            metadata=self.multilabel_metadata,
        )

    def step_2_define_transformations(self):
        def get_samples_labels(df, output_cols):
            X = df.drop(output_cols, axis=1)
            y = df[output_cols]

            return X, y

        def get_test_train_splits(X, y, test_size, random_state):

            return train_test_split(
                X, y, test_size=test_size, stratify=y, random_state=random_state
            )
            # returns X_train, X_test, y_train, y_test

        ##############################

        X, y = get_samples_labels(self.multilabel_data, output_cols=self.multilabel_outputs)

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
            dataset_name=f"{self.multilabel_name} - Train",
            metadata={},
            transformations=self.transformations_train,
        )

        self.transformations_test = [
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
            dataset_name=f"{self.multilabel_name} - Test",
            metadata={},
            transformations=self.transformations_test,
        )

    def step_3_upload_sklearn_training_run(self):
        # sklearn
        models = [RandomForestClassifier, DecisionTreeClassifier]
        configs = [{"random_state": 42}, {"random_state": 42, "max_depth": 3}]

        for model_class in models:
            for config in configs:
                model = model_class(**config)
                model.fit(self.X_train, self.y_train)

                self.controller.upload_training_run_split(
                    model,
                    X_train=self.X_train,
                    y_train=self.y_train,
                    X_test=self.X_test,
                    y_test=self.y_test,
                )

    def step_4_upload_xgboost_training_run(self):
        # xgboost multioutput classification is still experimental - todo add to this section when they add support
        pass

    def step_5_upload_lgbm_training_run(self):
        # lightgbm currently doesn't support multioutput classification - todo add to this section when they add support
        pass

    def step_6_upload_tensorflow_training_run(self):

        multilabel_normalize = tf.keras.layers.Normalization(axis=-1)
        multilabel_normalize.adapt(self.X_train)

        tf_model: tf.keras.Sequential = tf.keras.Sequential(
            [
                multilabel_normalize,
                tf.keras.layers.Dense(128, activation="relu", input_shape=(26,)),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(2),
            ]
        )

        tf_model.compile(
            optimizer="adam",
            loss=tf.keras.losses.KLDivergence(),
            metrics=[tf.keras.metrics.CategoricalAccuracy()],
        )

        tf_model.fit(x=self.X_train, y=self.y_train, epochs=5)
        self.controller.upload_training_run_split(
            tf_model,
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
