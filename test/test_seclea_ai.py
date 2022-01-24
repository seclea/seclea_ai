import os
import unittest

import pandas as pd

# from unittest import mock
# from unittest.mock import mock_open, patch
#
# import responses
from seclea_ai.transformations import DatasetTransformation

# from seclea_ai import SecleaAI


base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
folder_path = os.path.join(base_dir, "")
print(folder_path)


class TestSecleaAI(unittest.TestCase):
    # @responses.activate
    # @mock.patch("seclea_ai.authentication.getpass", return_value="test_pass")
    # @mock.patch("builtins.input", autospec=True, return_value="test_user")
    # def test_init_seclea_object(self, mock_input, mock_getpass) -> None:
    #     responses.add(
    #         method=responses.POST,
    #         url="http://localhost:8010/api/token/refresh/",
    #         json={"access": "dummy_access_token"},
    #         status=200,
    #     )
    #     responses.add(
    #         method=responses.POST,
    #         url="http://localhost:8010/api/token/obtain/",
    #         json={"access": "dummy_access_token", "refresh": "dummy_refresh_token"},
    #         status=200,
    #     )
    #     responses.add(
    #         method=responses.GET,
    #         url="http://localhost:8000/collection/projects?name=test-project",
    #         json=[{"id": 1, "name": "test-project"}],
    #         status=200,
    #     )
    #     responses.add(
    #         method=responses.GET,
    #         url="http://localhost:8000/collection/models",
    #         json=[{"id": 1, "name": "GBM-1"}],
    #         status=200,
    #     )
    #     responses.add(
    #         method=responses.GET,
    #         url="http://localhost:8000/collection/datasets",
    #         json=[{"id": 1, "name": "Fraud detection"}],
    #         status=200,
    #     )
    #     with patch(
    #         "builtins.open",
    #         new=mock_open(
    #             read_data='{"refresh": "dummy_refresh_token", "username": "test_user"}\n'
    #         ),
    #     ) as mock_file:
    #         SecleaAI(
    #             project_name="test-project",
    #             organization="Onespan",
    #             platform_url="http://localhost:8000",
    #             auth_url="http://localhost:8010",
    #         )
    #     mock_file.assert_called()
    #
    # @responses.activate
    # @mock.patch("seclea_ai.authentication.getpass", return_value="test_pass")
    # @mock.patch("builtins.input", autospec=True, return_value="test_user")
    # def test_getting_project_fail(self, mock_input, mock_getpass) -> None:
    #     """Test using an Project name that does not exist and sending new project fails raises a ValueError"""
    #     responses.add(
    #         method=responses.POST,
    #         url="http://localhost:8010/api/token/refresh/",
    #         json={"access": "dummy_access_token"},
    #         status=200,
    #     )
    #     responses.add(
    #         method=responses.POST,
    #         url="http://localhost:8010/api/token/obtain/",
    #         json={"access": "dummy_access_token", "refresh": "dummy_refresh_token"},
    #         status=200,
    #     )
    #     responses.add(
    #         method=responses.GET,
    #         url="http://localhost:8000/collection/projects?name=New Project",
    #         status=400,
    #     )
    #     with self.assertRaises(ValueError):
    #         SecleaAI(
    #             project_name="New Project",
    #             organization="Onespan",
    #             platform_url="http://localhost:8000",
    #             auth_url="http://localhost:8010",
    #         )
    #
    # @responses.activate
    # @mock.patch("seclea_ai.authentication.getpass", return_value="test_pass")
    # @mock.patch("builtins.input", autospec=True, return_value="test_user")
    # def test_upload_project_fail(self, mock_input, mock_getpass) -> None:
    #     """Test using an Project name that does not exist and sending new project fails raises a ValueError"""
    #     responses.add(
    #         method=responses.POST,
    #         url="http://localhost:8010/api/token/refresh/",
    #         json={"access": "dummy_access_token"},
    #         status=200,
    #     )
    #     responses.add(
    #         method=responses.POST,
    #         url="http://localhost:8010/api/token/obtain/",
    #         json={"access": "dummy_access_token", "refresh": "dummy_refresh_token"},
    #         status=200,
    #     )
    #     responses.add(
    #         method=responses.GET,
    #         url="http://localhost:8000/collection/projects?name=New Project",
    #         status=200,
    #         json=[],
    #     )
    #     responses.add(
    #         method=responses.GET,
    #         url="http://localhost:8000/collection/models",
    #         json=[{"id": 1, "name": "GBM-1"}],
    #         status=200,
    #     )
    #     responses.add(
    #         method=responses.GET,
    #         url="http://localhost:8000/collection/datasets",
    #         json=[{"id": 1, "name": "Fraud detection"}],
    #         status=200,
    #     )
    #     responses.add(
    #         method=responses.POST,
    #         url="http://localhost:8000/collection/projects",
    #         status=403,
    #     )
    #     with self.assertRaises(ValueError):
    #         SecleaAI(
    #             project_name="New Project",
    #             organization="Onespan",
    #             platform_url="http://localhost:8000",
    #             auth_url="http://localhost:8010",
    #         )

    def test_process_transformations(self):
        dataset = pd.read_csv(f"{folder_path}/insurance_claims.csv")
        # dataset_name = "Insurance Fraud Dataset"
        # dataset_meta = {
        #     "index": None,
        #     "outcome_name": "fraud_reported",
        #     "continuous_features": [
        #         "total_claim_amount",
        #         "policy_annual_premium",
        #         "capital-gains",
        #         "capital-loss",
        #         "injury_claim",
        #         "property_claim",
        #         "vehicle_claim",
        #         "incident_hour_of_the_day",
        #     ],
        # }

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

        df = encode_nans(dataset)

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

        def get_samples_labels(df, output_col):
            X = df.drop(output_col, axis=1)
            y = df[output_col]

            return X, y

        output_col = "fraud_reported"
        X, y = get_samples_labels(df, output_col=output_col)

        def get_test_train_splits(X, y, test_size, random_state):
            from sklearn.model_selection import train_test_split

            return train_test_split(
                X, y, test_size=test_size, stratify=y, random_state=random_state
            )
            # returns X_train, X_test, y_train, y_test

        test_size = 0.2
        random_state = 42
        X_train, X_test, y_train, y_test = get_test_train_splits(
            X, y, test_size=test_size, random_state=random_state
        )

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

        X_sm, y_sm = smote_balance(X_train, y_train, random_state=random_state)

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

        # deliberate test of datasnooping.
        scaler = fit(pd.concat([X_sm, X_test], axis=0))
        X_sm_scaled, _ = scale(X_sm, y_sm, scaler)
        X_test_scaled, _ = scale(X_test, y_test, scaler)

        self.complicated_transformations = [
            DatasetTransformation(encode_nans, {"df": dataset}, {}, ["data"]),
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

        self.complicated_transformations_train = [
            DatasetTransformation(
                get_test_train_splits,
                {"X": X, "y": y},
                {
                    "test_size": 0.2,
                    "random_state": 42,  # this becomes very important if we are re-running functions for uploading different branches...
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


if __name__ == "__main__":
    unittest.main()
