import datetime
import os
import uuid
from pathlib import Path
from unittest import TestCase

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from peewee import SqliteDatabase
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from seclea_ai import SecleaAI
from seclea_ai.internal.local_db import Record, RecordStatus
from seclea_ai.transformations import DatasetTransformation
from seclea_ai.lib.seclea_utils.object_management import Tracked

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
folder_path = os.path.join(base_dir, "test_integration_portal")


class TestIntegrationSKLearn(TestCase):
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
        dataset_file_name = "insurance_claims.csv"
        dataset_name = "Insurance Fraud Dataset"
        dataset = pd.read_csv(os.path.join(folder_path, dataset_file_name))
        self.df_1 = Tracked(dataset)
        self.df_1.object_manager.file_name = dataset_file_name[:-4]
        self.df_1.object_manager.path = folder_path
        self.df_1.object_manager.metadata.update({"dataset_name": dataset_name})
        self.df_1.object_manager.metadata.update({
            "outcome_name": "fraud_reported",
            "favourable_outcome": "N",
            "unfavourable_outcome": "Y",
            "continuous_features": [
                "total_claim_amount",
                "policy_annual_premium",
                "capital-gains",
                "capital-loss",
                "injury_claim",
                "property_claim",
                "incident_hour_of_the_day",
            ],
        })
        self.controller.upload_dataset(self.df_1)

    def step_2_define_transformations(self):
        def encode_nans(df):

            new_df = df.copy(deep=True)
            # dealing with special character
            new_df["collision_type"] = df["collision_type"].replace("?", np.NaN, inplace=False)
            new_df["property_damage"] = df["property_damage"].replace("?", np.NaN, inplace=False)
            new_df["police_report_available"] = df["police_report_available"].replace(
                "?",
                "NO",
                inplace=False,
            )  # default to no police report present if previously ?
            return new_df

        def drop_correlated(data, thresh):

            # calculate correlations
            corr_matrix = data.corr().abs()
            # get the upper part of correlation matrix
            upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

            # columns with correlation above threshold
            redundant = [column for column in upper.columns if any(upper[column] >= thresh)]
            print(f"Columns to drop with correlation > {thresh}: {redundant}")
            data.drop(columns=redundant, inplace=True)
            return data

        def drop_nulls(df, threshold):
            cols = [x for x in df.columns if df[x].isnull().sum() / df.shape[0] > threshold]
            return df.drop(columns=cols)

        def encode_categorical(df):

            cat_cols = df.select_dtypes(include=["object"]).columns.tolist()

            for col in cat_cols:
                if col in df.columns:
                    le = LabelEncoder()
                    le.fit(list(df[col].astype(str).values))
                    df[col] = le.transform(list(df[col].astype(str).values))
            return df

        def fill_na_by_col(df, fill_values: dict):
            return df.fillna(fill_values)

        def get_samples_labels(df, output_col):
            X = df.drop(output_col, axis=1)
            y = df[output_col]

            return X, y

        def get_test_train_splits(X, y, test_size, random_state):

            return train_test_split(
                X, y, test_size=test_size, stratify=y, random_state=random_state
            )
            # returns X_train, X_test, y_train, y_test

        def smote_balance(X, y, random_state):

            sm = SMOTE(random_state=random_state)

            X_sm, y_sm = sm.fit_resample(X, y)

            return X_sm, y_sm
            # returns X, y

        def fit_and_scale(X, y):

            scaler = StandardScaler()

            scaler.fit(X)
            X_transformed = X.copy()
            X_transformed[:] = scaler.transform(X_transformed[:])
            return X_transformed, y, scaler

        def fit(X):  # how do we handle these that don't affect directly the dataset..

            # ie. the scaler (as the input to another function) but that's not general..
            scaler = StandardScaler()

            # MAJOR question is. could we identify if they fitted it over the whole dataset... let's test
            scaler.fit(X)
            return scaler

        def scale(X, y, scaler):
            X_transformed = X.copy()
            X_transformed[:] = scaler.transform(X_transformed[:])
            return X_transformed, y

        df = encode_nans(self.df_1)

        corr_thresh = 0.97
        df = drop_correlated(df, corr_thresh)

        null_thresh = 0.9
        df = drop_nulls(df, threshold=null_thresh)

        df = encode_categorical(df)

        ##############################

        output_col = "fraud_reported"
        X, y = get_samples_labels(df, output_col=output_col)

        test_size = 0.2
        random_state = 42
        X_train, self.X_test, y_train, self.y_test = get_test_train_splits(
            X, y, test_size=test_size, random_state=random_state
        )
        self.X_sm, self.y_sm = smote_balance(X_train, y_train, random_state=random_state)
        # deliberate test of datasnooping.
        scaler = fit(pd.concat([self.X_sm, self.X_test], axis=0))
        self.X_sm_scaled, _ = scale(self.X_sm, self.y_sm, scaler)
        self.X_test_scaled, _ = scale(self.X_test, self.y_test, scaler)

        self.complicated_transformations = [
            DatasetTransformation(
                encode_nans, data_kwargs={"df": self.df_1}, kwargs={}, outputs=["data"]
            ),
            DatasetTransformation(
                drop_correlated, {"data": "inherit"}, {"thresh": corr_thresh}, ["df"]
            ),
            DatasetTransformation(
                drop_nulls, {"df": "inherit"}, {"threshold": null_thresh}, ["df"]
            ),
            DatasetTransformation(encode_categorical, {"df": "inherit"}, {}, ["df"]),
        ]

        # upload dataset here
        self.controller.upload_dataset_split(
            x=X,
            y=y,
            dataset_name=f"{self.df_1_name} - Cleaned",
            metadata={"favourable_outcome": 1, "unfavourable_outcome": 0},
            transformations=self.complicated_transformations,
        )

        self.complicated_transformations_train = [
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
            DatasetTransformation(
                smote_balance,
                {"X": "inherit", "y": "inherit"},
                {"random_state": random_state},
                ["X", "y"],
            ),
            DatasetTransformation(
                scale, {"X": "inherit", "y": "inherit"}, {"scaler": scaler}, ["X", "y", None]
            ),
        ]

        # upload dataset here
        self.controller.upload_dataset_split(
            x=self.X_sm_scaled,
            y=self.y_sm,
            dataset_name=f"{self.df_1_name} Train - Balanced - Scaled",
            metadata={},
            transformations=self.complicated_transformations_train,
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
            DatasetTransformation(
                scale, {"X": "inherit", "y": "inherit"}, {"scaler": scaler}, ["X", "y"]
            ),
        ]

        # upload dataset here
        self.controller.upload_dataset_split(
            x=self.X_test_scaled,
            y=self.y_test,
            dataset_name=f"{self.df_1_name} Test - Scaled",
            metadata={},
            transformations=self.complicated_transformations_test,
        )

        self.df_1_transformed = df

    def step_3_upload_training_run(self):
        # define model

        from sklearn.ensemble import RandomForestClassifier

        # from sklearn.metrics import accuracy_score
        # Train
        model = RandomForestClassifier(random_state=42)
        model.fit(self.X_sm_scaled, self.y_sm)
        # preds = model.predict(self.X_test_scaled)

        self.controller.upload_training_run_split(
            model,
            X_train=self.X_sm_scaled,
            y_train=self.y_sm,
            X_test=self.X_test_scaled,
            y_test=self.y_test,
        )

        model1 = RandomForestClassifier(random_state=42, n_estimators=32)
        model1.fit(self.X_sm, self.y_sm)
        self.controller.upload_training_run_split(
            model1, X_train=self.X_sm, y_train=self.y_sm, X_test=self.X_test, y_test=self.y_test
        )
        self.controller.complete()

    def step_4_check_all_sent(self):
        # check that all record statuses are RecordStatus.SENT.value
        db = SqliteDatabase(
            Path.home() / ".seclea" / "seclea_ai.db",
            thread_safe=True,
            pragmas={"journal_mode": "wal"},
        )
        db.connect()
        records = Record.select().where(Record.timestamp > self.start_timestamp)
        for idx, record in enumerate(records):
            self.assertEqual(
                record.status,
                RecordStatus.SENT.value,
                f"Entity {record.entity} at position {idx}, with id {record.id} not sent, current status: {record.status}",
            )
        db.close()

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
