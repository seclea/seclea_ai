import datetime
import os
import uuid
from pathlib import Path
from unittest import TestCase

import pandas as pd
from peewee import SqliteDatabase
from sklearn.model_selection import train_test_split
import tensorflow as tf

from seclea_ai import SecleaAI
from seclea_ai.internal.local_db import Record, RecordStatus
from seclea_ai.transformations import DatasetTransformation

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
folder_path = os.path.join(base_dir, "test_integration_portal")


class TestIntegrationTensorflow(TestCase):
    """
    Monolithic testing of the Seclea AI file
    order of functions is preserved.

    NOTE: a random project is named on each tests this is to
    pseudo reset the database.

    Please reset database upon completing work

    Workflow to reset db on each test run to be investigated.
    """

    def step_0_project_setup(self):
        self.start_timestamp = datetime.datetime.now()
        print(self.start_timestamp)
        self.password = "asdf"  # nosec
        self.username = "admin"  # nosec
        self.organization = "Onespan"
        self.project = f"test-project-{uuid.uuid4()}"
        self.portal_url = "http://localhost:8000"
        self.auth_url = "http://localhost:8010"
        self.controller = SecleaAI(
            project_name=self.project,
            organization=self.organization,
            platform_url=self.portal_url,
            auth_url=self.auth_url,
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
            x=self.X_train,
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
            x=self.X_test,
            y=self.y_test,
            dataset_name=f"{self.sample_df_name} - Test",
            metadata={},
            transformations=self.complicated_transformations_test,
        )

    def step_3_upload_training_run(self):

        # define model
        normalizer = tf.keras.layers.Normalization(axis=-1)
        normalizer.adapt(self.X_train)

        model: tf.keras.Sequential = tf.keras.Sequential(
            [
                normalizer,
                tf.keras.layers.Dense(512, activation="relu", input_shape=(111,)),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(4),
            ]
        )

        # TODO find multiclass classification loss and metrics etc.
        model.compile(
            optimizer="adam",
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
        )

        model.fit(x=self.X_train, y=self.y_train, epochs=5)

        self.controller.upload_training_run_split(
            model,
            X_train=self.X_train,
            y_train=self.y_train,
            X_test=self.X_test,
            y_test=self.y_test,
        )
        self.controller.complete()

    def step_4_check_all_sent(self):
        # check that all record statuses are RecordStatus.SENT.value

        records = Record.select().where(Record.timestamp > self.start_timestamp)
        for idx, record in enumerate(records):
            self.assertEqual(
                record.status,
                RecordStatus.SENT.value,
                f"Entity {record.entity} at position {idx}, with id {record.id} not sent, current status: {record.status}",
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
