import os
import uuid
from unittest import TestCase

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import xgboost as xgb
from seclea_ai import SecleaAI
from seclea_ai.transformations import DatasetTransformation

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
folder_path = os.path.join(base_dir, "test_integration_portal")


class TestIntegrationCensus(TestCase):
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
        self.project_name_1 = f"test-project-{uuid.uuid4()}"
        self.portal_url = "http://localhost:8000"
        self.auth_url = "http://localhost:8010"
        self.controller = SecleaAI(
            self.project_name_1,
            self.organization,
            self.portal_url,
            self.auth_url,
            username=self.username,
            password=self.password,
        )

    def step_1_upload_dataset(self):

        self.sample_df = pd.read_csv(f"{folder_path}/adult_data.csv", index_col=0)
        self.sample_df_name = "Census dataset"
        self.sample_df_meta = {
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

        def encode_categorical(df):
            new_df = df.copy(deep=True)

            cat_cols = new_df.select_dtypes(include=["object"]).columns.tolist()

            for col in cat_cols:
                le = LabelEncoder()
                le.fit(list(new_df[col].astype(str).values))
                new_df[col] = le.transform(list(new_df[col].astype(str).values))
            return new_df

        ##############################

        df = encode_categorical(self.sample_df)

        self.transformations_encode = [
            DatasetTransformation(
                func=encode_categorical,
                data_kwargs={"df": self.sample_df},
                kwargs={},
                outputs=["df"],
            ),
        ]

        # upload dataset here
        self.controller.upload_dataset(
            dataset=df,
            dataset_name=f"{self.sample_df_name} - Encoded",
            metadata={},
            transformations=self.transformations_encode,
        )

        output_col = "income-per-year"
        X, y = get_samples_labels(df, output_col=output_col)

        test_size = 0.2
        random_state = 42
        self.X_train, self.X_test, self.y_train, self.y_test = get_test_train_splits(
            X, y, test_size=test_size, random_state=random_state
        )

        for col in self.X_train.select_dtypes(include=["object"]).columns.tolist():
            self.X_train[col] = self.X_train[col].astype("category")
            self.X_test[col] = self.X_test[col].astype("category")

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
            dataset_name=f"{self.sample_df_name} - Test",
            metadata={},
            transformations=self.transformations_test,
        )

    def step_3_upload_training_run_1(self):
        # define model

        from sklearn.ensemble import RandomForestClassifier

        # from sklearn.metrics import accuracy_score
        # Train
        model = RandomForestClassifier(random_state=42)
        model.fit(self.X_train, self.y_train)
        # preds = model.predict(self.X_test_scaled)

        self.controller.upload_training_run_split(
            model,
            X_train=self.X_train,
            y_train=self.y_train,
            X_test=self.X_test,
            y_test=self.y_test,
        )

    def step_3_upload_training_run_2(self):
        # define model

        from sklearn.tree import DecisionTreeClassifier

        # from sklearn.metrics import accuracy_score
        # Train
        model = DecisionTreeClassifier(random_state=42)
        model.fit(self.X_train, self.y_train)
        # preds = model.predict(self.X_test_scaled)

        self.controller.upload_training_run_split(
            model,
            X_train=self.X_train,
            y_train=self.y_train,
            X_test=self.X_test,
            y_test=self.y_test,
        )

    def step_3_upload_training_run_3(self):
        # define model

        from sklearn.linear_model import LogisticRegression

        # from sklearn.metrics import accuracy_score
        # Train
        model = LogisticRegression(random_state=42)
        model.fit(self.X_train, self.y_train)
        # preds = model.predict(self.X_test_scaled)

        self.controller.upload_training_run_split(
            model,
            X_train=self.X_train,
            y_train=self.y_train,
            X_test=self.X_test,
            y_test=self.y_test,
        )

    def step_3_upload_training_run_4(self):
        # define model
        # Train
        model = xgb.XGBClassifier(objective="binary:logistic", random_state=42)
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

        from lightgbm import LGBMClassifier

        # from sklearn.metrics import accuracy_score
        # Train
        model = LGBMClassifier(objective='binary', random_state=5)
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
