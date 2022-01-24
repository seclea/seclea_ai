import os
import uuid
from unittest import TestCase

import numpy as np
import pandas as pd

from seclea_ai import Frameworks, SecleaAI

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
        # transformations
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

        df = encode_nans(self.sample_df_1)

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

        corr_thresh = 0.97
        df = drop_correlated(df, corr_thresh)

        def drop_nulls(df, threshold):
            cols = [x for x in df.columns if df[x].isnull().sum() / df.shape[0] > threshold]
            return df.drop(columns=cols)

        null_thresh = 0.9
        df = drop_nulls(df, threshold=null_thresh)

        def encode_categorical(df):
            from sklearn.preprocessing import LabelEncoder

            cat_cols = df.select_dtypes(include=["object"]).columns.tolist()

            for col in cat_cols:
                if col in df.columns:
                    le = LabelEncoder()
                    le.fit(list(df[col].astype(str).values))
                    df[col] = le.transform(list(df[col].astype(str).values))
            return df

        df = encode_categorical(df)

        def fill_na_by_col(df, fill_values: dict):
            return df.fillna(fill_values)

        na_values = {"collision_type": -1, "property_damage": -1}
        df = fill_na_by_col(df, na_values)

        def smote_balance(df):
            from imblearn.over_sampling import SMOTE

            X1 = df.drop("fraud_reported", axis=1)
            y1 = df.fraud_reported

            sm = SMOTE(random_state=42)

            X_sm, y_sm = sm.fit_resample(X1, y1)

            print(
                f"""Shape of X before SMOTE: {X1.shape}
            Shape of X after SMOTE: {X_sm.shape}"""
            )
            print(
                f"""Shape of y before SMOTE: {y1.shape}
            Shape of y after SMOTE: {y_sm.shape}"""
            )
            return pd.concat([X_sm, y_sm], axis=1)
            # returns DataFrame

        def smote_balance_split(X, y):
            from imblearn.over_sampling import SMOTE

            sm = SMOTE(random_state=42)

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

        def smote_balance_df_split(df, output_col):
            from imblearn.over_sampling import SMOTE

            X1 = df.drop(output_col, axis=1)
            y1 = df[output_col]

            sm = SMOTE(random_state=42)

            X_sm, y_sm = sm.fit_resample(X1, y1)

            print(
                f"""Shape of X before SMOTE: {X1.shape}
            Shape of X after SMOTE: {X_sm.shape}"""
            )
            print(
                f"""Shape of y before SMOTE: {y1.shape}
            Shape of y after SMOTE: {y_sm.shape}"""
            )
            return X_sm, y_sm
            # returns X, y

        def get_test_train_splits(df, output_col, test_size, random_state):
            from sklearn.model_selection import train_test_split

            X = df.drop(output_col, axis=1)
            y = df[output_col]

            return train_test_split(
                X, y, test_size=test_size, stratify=y, random_state=random_state
            )
            # returns X_train, X_test, y_train, y_test

        def get_test_train_splits_split(X, y, test_size, random_state):
            from sklearn.model_selection import train_test_split

            return train_test_split(
                X, y, test_size=test_size, stratify=y, random_state=random_state
            )
            # returns X_train, X_test, y_train, y_test

        self.transformations = [
            encode_nans,
            (drop_correlated, [corr_thresh], {}),
            (drop_nulls, [null_thresh], {}),
            (encode_categorical, [], {}),
            (fill_na_by_col, [], {"fill_values": na_values}),
        ]

        # could do variable substitution - assign variable name to output to use later in transformations.
        # merge prev output - None's and new - inherit

        # self.complicated_transformations = [
        #     DatasetTransformation(encode_nans, {"df": self.sample_df_1}, ["data"]),
        #     DatasetTransformation(
        #         drop_correlated, {"data": "inherit", "thresh": corr_thresh}, ["df"]
        #     ),
        #     DatasetTransformation(drop_nulls, {"df": "inherit", "threshold": null_thresh}, ["df"]),
        #     DatasetTransformation(encode_categorical, {"df": "inherit"}, ["df"]),
        #     DatasetTransformation(
        #         fill_na_by_col, {"df": "inherit", "fill_values": na_values}, ["df"]
        #     ),
        #     DatasetTransformation(
        #         get_test_train_splits,
        #         {
        #             "df": "inherit",
        #             "output_col": "fraud_reported",
        #             "test_size": 0.2,
        #             "random_state": 42,
        #         },
        #         ["X", None, "y", None],
        #     ),
        #     DatasetTransformation(
        #         smote_balance_split, {"X": "inherit", "y": "inherit"}, ["X", "y"]
        #     ),
        # ]
        #
        # self.complicated_transformations_train = [
        #     DatasetTransformation(
        #         get_test_train_splits,
        #         {
        #             "df": "inherit",
        #             "output_col": "fraud_reported",
        #             "test_size": 0.2,
        #             "random_state": 42,
        #         },
        #         ["X", None, "y", None],
        #     ),
        #     DatasetTransformation(
        #         smote_balance_split, {"X": "inherit", "y": "inherit"}, ["X", "y"]
        #     ),
        # ]
        #
        # self.complicated_transformations_test = [
        #     DatasetTransformation(
        #         get_test_train_splits,
        #         {
        #             "df": "inherit",
        #             "output_col": "fraud_reported",
        #             "test_size": 0.2,
        #             "random_state": 42,
        #         },
        #         [None, "X", None, "y"],
        #     )
        # ]

        self.sample_df_1_transformed = df

    def step_3_upload_transformed_dataset(self):
        self.sample_df_1_transformed_name = "Insurance Fraud Transformed"
        self.controller_1.upload_dataset(
            self.sample_df_1_transformed,
            self.sample_df_1_transformed_name,
            metadata=self.sample_df_1_meta,
            parent=self.sample_df_1,
            transformations=self.transformations,
        )

    def step_4_upload_trainingrun(self):
        # define model

        from sklearn.model_selection import train_test_split

        X = self.sample_df_1_transformed.drop("fraud_reported", axis=1)
        y = self.sample_df_1_transformed.fraud_reported

        # train test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        print(
            f"""% Positive class in Train = {np.round(y_train.value_counts(normalize=True)[1] * 100, 2)}
        % Positive class in Test  = {np.round(y_test.value_counts(normalize=True)[1] * 100, 2)}"""
        )

        from sklearn.ensemble import RandomForestClassifier

        # from sklearn.metrics import accuracy_score
        # Train
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)
        # preds = model.predict(X_test)

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
