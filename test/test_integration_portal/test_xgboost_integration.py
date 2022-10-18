import datetime
import os
import uuid
from pathlib import Path
from unittest import TestCase

import numpy as np
import pandas as pd
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from peewee import SqliteDatabase
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import DMatrix

from seclea_ai import SecleaAI
from seclea_ai.internal.models.record import Record, RecordStatus
from seclea_ai.transformations import DatasetTransformation

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
folder_path = os.path.join(base_dir, "test_integration_portal")


class TestIntegrationXGBoost(TestCase):
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
        self.username = "admin-dev"  # nosec
        self.organization = "Onespan"
        self.project_name = f"test-project-{uuid.uuid4()}"
        self.portal_url = "http://localhost:8000"
        self.auth_url = "http://localhost:8010"
        self.controller = SecleaAI(
            project_name=self.project_name,
            organization=self.organization,
            platform_url=self.portal_url,
            auth_url=self.auth_url,
            username=self.username,
            password=self.password,
        )

    def step_1_upload_dataset(self):
        self.sample_df_1 = pd.read_csv(f"{folder_path}/insurance_claims.csv")
        self.sample_df_1_name = "Insurance Fraud Dataset"
        self.sample_df_1_meta = {
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
        }
        self.controller.upload_dataset(
            self.sample_df_1, self.sample_df_1_name, self.sample_df_1_meta
        )

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

        df = encode_nans(self.sample_df_1)

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
        self.X_train, self.X_test, self.y_train, self.y_test = get_test_train_splits(
            X, y, test_size=test_size, random_state=random_state
        )
        self.X_sm, self.y_sm = smote_balance(self.X_train, self.y_train, random_state=random_state)

        self.complicated_transformations = [
            DatasetTransformation(
                encode_nans, data_kwargs={"df": self.sample_df_1}, kwargs={}, outputs=["data"]
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
            X=X,
            y=y,
            dataset_name=f"{self.sample_df_1_name} - Cleaned",
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
        ]

        # upload dataset here
        self.controller.upload_dataset_split(
            X=self.X_sm,
            y=self.y_sm,
            dataset_name=f"{self.sample_df_1_name} Train - Balanced",
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
        ]

        # upload dataset here
        self.controller.upload_dataset_split(
            X=self.X_test,
            y=self.y_test,
            dataset_name=f"{self.sample_df_1_name} Test - Scaled",
            metadata={},
            transformations=self.complicated_transformations_test,
        )

        self.sample_df_1_transformed = df

    def step_3_upload_trainingrun(self):
        # define model
        # setup training
        dtrain = DMatrix(data=self.X_sm, label=self.y_sm, enable_categorical=True)
        params = dict(max_depth=2, eta=1, objective="binary:logistic", nthread=4, eval_metric="auc")
        num_rounds = 5
        model = xgb.train(params=params, dtrain=dtrain, num_boost_round=num_rounds)

        self.controller.upload_training_run_split(
            model,
            X_train=self.X_sm,
            y_train=self.y_sm,
            X_test=self.X_test,
            y_test=self.y_test,
        )
        self.controller.complete()

    def step_4_check_all_sent(self):
        # check that all record statuses are RecordStatus.SENT
        db = SqliteDatabase(
            Path.home() / ".seclea" / "seclea_ai.db",
            thread_safe=True,
            pragmas={"journal_mode": "wal"},
        )
        db.connect()
        records = Record.select().where(Record.created_timestamp > self.start_timestamp)
        for idx, record in enumerate(records):
            self.assertEqual(
                record.status,
                RecordStatus.SENT,
                f"Entity {record.entity} at position {idx}, with id {record.id} not sent, current status: {record.status}",
            )
        db.close()

    def _steps(self):
        for name in dir(self):  # dir() result is implicitly sorted
            if name.startswith("step"):
                yield name, getattr(self, name)

    def test_steps(self):
        for name, step in self._steps():
            step()
            print("STEP COMPLETE")
