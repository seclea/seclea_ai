import datetime
import os
from pathlib import Path
from unittest import TestCase

import lightgbm as lgb
import numpy as np
import pandas as pd
import tensorflow as tf
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from peewee import SqliteDatabase
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import DMatrix

from seclea_ai import SecleaAI
from seclea_ai.internal.persistence.record import Record, RecordStatus
from seclea_ai.transformations import DatasetTransformation

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
folder_path = os.path.join(base_dir, "test_integration_portal")


class TestBinaryDataIntegration(TestCase):
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
        self.project_name = "Binary Classification Project"
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
        self.sample_df_1 = pd.read_csv(f"{folder_path}/data/insurance_claims.csv")
        self.sample_df_1_name = "Insurance Fraud Dataset"
        self.sample_df_1_meta = {
            "outputs": ["fraud_reported"],
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
            DatasetTransformation(
                scale, {"X": "inherit", "y": "inherit"}, {"scaler": scaler}, ["X", "y", None]
            ),
        ]

        # upload dataset here
        self.controller.upload_dataset_split(
            X=self.X_sm_scaled,
            y=self.y_sm,
            dataset_name=f"{self.sample_df_1_name} Train - Balanced - Scaled",
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
            X=self.X_test_scaled,
            y=self.y_test,
            dataset_name=f"{self.sample_df_1_name} Test - Scaled",
            metadata={},
            transformations=self.complicated_transformations_test,
        )

        self.sample_df_1_transformed = df

    def step_3_upload_sklearn_training_run(self):
        # sklearn
        sklearn_model = RandomForestClassifier(random_state=42)
        sklearn_model.fit(self.X_sm_scaled, self.y_sm)

        self.controller.upload_training_run_split(
            sklearn_model,
            X_train=self.X_sm_scaled,
            y_train=self.y_sm,
            X_test=self.X_test_scaled,
            y_test=self.y_test,
        )

        sklearn_model = RandomForestClassifier(random_state=42, n_estimators=32)
        sklearn_model.fit(self.X_sm, self.y_sm)
        self.controller.upload_training_run_split(
            sklearn_model,
            X_train=self.X_sm,
            y_train=self.y_sm,
            X_test=self.X_test,
            y_test=self.y_test,
        )

    def step_4_upload_xgboost_training_run(self):
        # xgboost
        xgb_dtrain = DMatrix(data=self.X_sm, label=self.y_sm, enable_categorical=True)
        params = dict(max_depth=3, eta=1, objective="binary:logistic")
        num_rounds = 5
        xgb_model = xgb.train(params=params, dtrain=xgb_dtrain, num_boost_round=num_rounds)

        self.controller.upload_training_run_split(
            xgb_model,
            X_train=self.X_sm,
            y_train=self.y_sm,
            X_test=self.X_test,
            y_test=self.y_test,
        )

    def step_5_upload_lgbm_training_run(self):
        # lightgbm
        lgb_dtrain = lgb.Dataset(
            data=self.X_sm,
            label=self.y_sm,
            free_raw_data=True,
            categorical_feature=[
                "policy_csl",
                "policy_deductable",
                "umbrella_limit",
                "collision_type",
                "incident_type",
                "months_as_customer",
                "witnesses",
                "insured_zip",
                "insured_education_level",
                "auto_make",
                "insured_hobbies",
                "policy_bind_date",
                "incident_city",
                "incident_date",
                "insured_sex",
                "bodily_injuries",
                "auto_model",
                "authorities_contacted",
                "number_of_vehicles_involved",
                "insured_occupation",
                "age",
                "policy_number",
                "police_report_available",
                "incident_state",
                "property_damage",
                "incident_location",
                "incident_severity",
                "auto_year",
                "policy_state",
                "insured_relationship",
            ],
        )

        # setup training params
        params = dict(max_depth=3, eta=1, objective="binary")
        num_rounds = 5
        # train model
        lgb_model = lgb.train(
            params=params,
            train_set=lgb_dtrain,
            num_boost_round=num_rounds,
        )

        # upload model
        self.controller.upload_training_run_split(
            lgb_model,
            X_train=self.X_sm,
            y_train=self.y_sm,
            X_test=self.X_test,
            y_test=self.y_test,
        )

    def step_6_upload_tensorflow_training_run(self):
        # define tensorflow model
        normalizer = tf.keras.layers.Normalization(axis=-1)
        normalizer.adapt(self.X_sm)

        tf_model: tf.keras.Sequential = tf.keras.Sequential(
            [
                normalizer,
                tf.keras.layers.Dense(128, activation="relu", input_shape=(37,)),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(1),
            ]
        )

        tf_model.compile(
            optimizer="adam",
            loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
            metrics=[tf.keras.metrics.BinaryAccuracy()],
        )

        tf_model.fit(x=self.X_sm, y=self.y_sm, epochs=5)

        self.controller.upload_training_run_split(
            tf_model,
            X_train=self.X_sm,
            y_train=self.y_sm,
            X_test=self.X_test,
            y_test=self.y_test,
        )
        self.controller.complete()

    def step_7_check_all_sent(self):
        # check that all record statuses are RecordStatus.SENT
        db = SqliteDatabase(
            Path.home() / ".seclea" / "seclea_ai.db",
            thread_safe=True,
            pragmas={"journal_mode": "wal"},
        )
        with db.atomic():
            records = Record.select().where(Record.created_timestamp > self.start_timestamp)
            for idx, record in enumerate(records):
                self.assertEqual(
                    record.status,
                    RecordStatus.SENT,
                    f"Entity at position {idx}, with id {record.id} not sent, "
                    f"current status: {record.status}",
                )

    def _steps(self):
        for name in dir(self):  # dir() result is implicitly sorted
            if name.startswith("step"):
                yield name, getattr(self, name)

    def test_steps(self):
        for name, step in self._steps():
            step()
            print("STEP COMPLETE")
