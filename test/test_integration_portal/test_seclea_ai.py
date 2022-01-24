import os
import uuid
from unittest import TestCase

import numpy as np
import pandas as pd

from seclea_ai import Frameworks, SecleaAI
from seclea_ai.transformations import DatasetTransformation

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
folder_path = os.path.join(base_dir, "test_integration_portal")
print(folder_path)


class TestIntegrationSecleaAIPortal(TestCase):
    """
    Monolithic testing of the Seclea AI file
    order of functions is preserved.

    NOTE: a random project is named on each tests this is to
    pseudo reset the database.

    Please reset database upon completing work

    Workflow to reset db on each test run to be investigated.
    """

    def step_0_project_setup(self):
        self.password = "asdf"
        self.username = "onespanadmin"
        self.organization = "Onespan"
        self.project_name_1 = f"test-project-{uuid.uuid4()}"
        self.project_name_2 = f"test-project-{uuid.uuid4()}"
        self.portal_url = "http://localhost:8000"
        self.auth_url = "http://localhost:8010"
        self.controller_1 = SecleaAI(
            self.project_name_1,
            self.organization,
            self.portal_url,
            self.auth_url,
            username=self.username,
            password=self.password,
        )
        # create second project for second dataset
        self.controller_2 = SecleaAI(
            self.project_name_2,
            self.organization,
            self.portal_url,
            self.auth_url,
            username=self.username,
            password=self.password,
        )

    def step_1_upload_dataset(self):
        self.sample_df_1 = pd.read_csv(f"{folder_path}/insurance_claims.csv")
        self.sample_df_1_name = "Insurance Fraud Dataset"
        self.sample_df_1_meta = {
            "index": None,
            "outcome_name": "fraud_reported",
            "continuous_features": [
                "total_claim_amount",
                "policy_annual_premium",
                "capital-gains",
                "capital-loss",
                "injury_claim",
                "property_claim",
                "vehicle_claim",
                "incident_hour_of_the_day",
            ],
        }
        self.controller_1.upload_dataset(
            self.sample_df_1, self.sample_df_1_name, self.sample_df_1_meta
        )

        self.sample_df_2 = pd.read_csv(f"{folder_path}/adult_data.csv")
        self.sample_df_2_name = "Census dataset"
        self.sample_df_2_meta = {
            "index": None,
            "outcome_name": "income-per-year",
            "continuous_features": [
                "age",
                "fnlwgt",
                "education-num",
                "capital-gain",
                "capital-loss",
                "hours-per-week",
            ],
        }
        self.controller_2.upload_dataset(
            self.sample_df_2, self.sample_df_2_name, self.sample_df_2_meta
        )

    def step_2_define_transformations(self):
        def encode_nans(df):
            import numpy as np

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
            import numpy as np

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
            from sklearn.preprocessing import LabelEncoder

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
            from sklearn.model_selection import train_test_split

            return train_test_split(
                X, y, test_size=test_size, stratify=y, random_state=random_state
            )
            # returns X_train, X_test, y_train, y_test

        def smote_balance(X, y, random_state):
            from imblearn.over_sampling import SMOTE

            sm = SMOTE(random_state=random_state)

            X_sm, y_sm = sm.fit_resample(X, y)

            print(
                f"""Shape of X before SMOTE: {X.shape}
            Shape of X after SMOTE: {X_sm.shape}"""
            )
            print(
                f"""Shape of y before SMOTE: {y.shape}
            Shape of y after SMOTE: {y_sm.shape}"""
            )
            return X_sm, y_sm
            # returns X, y

        def fit_and_scale(X, y):
            from sklearn.preprocessing import StandardScaler

            scaler = StandardScaler()

            scaler.fit(X)
            X_transformed = scaler.transform(X)
            return X_transformed, y, scaler

        def fit(X):  # how do we handle these that don't affect directly the dataset..
            from sklearn.preprocessing import (
                StandardScaler,  # for this specific case we will record the output..
            )

            scaler = (
                StandardScaler()
            )  # ie. the scaler (as the input to another function) but that's not general..

            scaler.fit(
                X
            )  # MAJOR question is. could we identify if they fitted it over the whole dataset... let's test
            return scaler

        def scale(X, y, fitted_scaler):
            X_transformed = fitted_scaler.transform(X)
            return X_transformed, y

        df = encode_nans(self.sample_df_1)
        corr_thresh = 0.97
        df = drop_correlated(df, corr_thresh)
        null_thresh = 0.9
        df = drop_nulls(df, threshold=null_thresh)
        df = encode_categorical(df)
        na_values = {"collision_type": -1, "property_damage": -1}
        df = fill_na_by_col(df, na_values)
        output_col = "fraud_reported"
        X, y = get_samples_labels(df, output_col=output_col)
        test_size = 0.2
        random_state = 42
        X_train, X_test, y_train, self.y_test = get_test_train_splits(
            X, y, test_size=test_size, random_state=random_state
        )
        self.X_sm, self.y_sm = smote_balance(X_train, y_train, random_state=random_state)
        # deliberate test of datasnooping.
        scaler = fit(pd.concat([self.X_sm, X_test], axis=0))
        self.X_sm_scaled, _ = scale(self.X_sm, self.y_sm, scaler)
        self.X_test_scaled, _ = scale(X_test, self.y_test, scaler)

        self.complicated_transformations = [
            DatasetTransformation(encode_nans, {"df": self.sample_df_1}, {}, ["data"]),
            DatasetTransformation(
                drop_correlated, {"data": "inherit"}, {"thresh": corr_thresh}, ["df"]
            ),
            DatasetTransformation(
                drop_nulls, {"df": "inherit"}, {"threshold": null_thresh}, ["df"]
            ),
            DatasetTransformation(encode_categorical, {"df": "inherit"}, {}, ["df"]),
            DatasetTransformation(
                fill_na_by_col, {"df": "inherit"}, {"fill_values": na_values}, ["df"]
            ),
            DatasetTransformation(
                get_samples_labels, {"df": "inherit"}, {"output_col": "fraud_reported"}, ["X", "y"]
            ),
        ]

        # upload dataset here
        self.controller_1.upload_dataset_split(
            X=X,
            y=y,
            dataset_name=f"{self.sample_df_1_name} - Cleaned",
            metadata=self.sample_df_1_meta,
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
        self.controller_1.upload_dataset_split(
            X=self.X_sm_scaled,
            y=self.y_sm,
            dataset_name=f"{self.sample_df_1_name} Train - Balanced - Scaled",
            metadata=self.sample_df_1_meta,
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
            ),
            DatasetTransformation(
                scale, {"X": "inherit", "y": "inherit"}, {"scaler": scaler}, ["X", "y"]
            ),
        ]

        # upload dataset here
        self.controller_1.upload_dataset_split(
            X=self.X_test_scaled,
            y=self.y_test,
            dataset_name=f"{self.sample_df_1_name} Test - Scaled",
            metadata=self.sample_df_1_meta,
            transformations=self.complicated_transformations_test,
        )

        self.transformations = [
            encode_nans,
            (drop_correlated, [corr_thresh], {}),
            (drop_nulls, [null_thresh], {}),
            (encode_categorical, [], {}),
            (fill_na_by_col, [], {"fill_values": na_values}),
        ]

        self.sample_df_1_transformed = df

    def step_3_upload_trainingrun(self):
        # define model

        print(
            f"""% Positive class in Train = {np.round(self.y_sm.value_counts(normalize=True)[1] * 100, 2)}
            % Positive class in Test  = {np.round(self.y_test.value_counts(normalize=True)[1] * 100, 2)}"""
        )

        from sklearn.ensemble import RandomForestClassifier

        # from sklearn.metrics import accuracy_score
        # Train
        model = RandomForestClassifier(random_state=42)
        model.fit(self.X_sm_scaled, self.y_sm)
        # preds = model.predict(self.X_test_scaled)

        self.controller_1.upload_training_run(
            model,
            framework=Frameworks.SKLEARN,
            dataset=self.sample_df_1_transformed,
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
