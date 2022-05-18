import os
import traceback
import uuid
from unittest import TestCase

import numpy as np
import pandas as pd

from seclea_ai import SecleaAI
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
        self.password = "asdf"  # nosec
        self.username = "onespanadmin"  # nosec
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
        self.sample_df_1 = pd.read_csv(
            f"{folder_path}/insurance_claims.csv", index_col="policy_number"
        )
        self.sample_df_1_name = "Auto Insurance Fraud"
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
        self.controller_1.upload_dataset(
            self.sample_df_1, self.sample_df_1_name, self.sample_df_1_meta
        )

        self.sample_df_2 = pd.read_csv(f"{folder_path}/adult_data.csv")
        self.sample_df_2_name = "Census dataset"
        self.sample_df_2_meta = {
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
        self.controller_2.upload_dataset(
            self.sample_df_2, self.sample_df_2_name, self.sample_df_2_meta
        )

    def step_2_define_transformations(self):

        # Create a copy to isolate the original dataset
        df1 = self.sample_df_1.copy(deep=True)

        def encode_nans(df):
            # convert the special characters to nans
            return df.replace("?", np.NaN)

        df2 = encode_nans(df1)

        # Drop the the columns which are more than some proportion NaN values
        def drop_nulls(df, threshold):
            cols = [x for x in df.columns if df[x].isnull().sum() / df.shape[0] > threshold]
            return df.drop(columns=cols)

        # We choose 95% as our threshold
        null_thresh = 0.95
        df3 = drop_nulls(df2, threshold=null_thresh)

        def drop_correlated(data, thresh):
            import numpy as np

            # calculate correlations
            corr_matrix = data.corr().abs()
            # get the upper part of correlation matrix
            upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

            # columns with correlation above threshold
            redundant = [column for column in upper.columns if any(upper[column] >= thresh)]
            print(f"Columns to drop with correlation > {thresh}: {redundant}")
            new_data = data.drop(columns=redundant)
            return new_data

        # drop columns that are too closely correlated
        correlation_threshold = 0.9
        df4 = drop_correlated(df3, correlation_threshold)

        # define the updates to the metadata - empty as there are no changes
        processed_metadata = {}

        # 🔀 define the transformations - note the arguments
        cleaning_transformations = [
            DatasetTransformation(encode_nans, data_kwargs={"df": df1}, kwargs={}, outputs=["df"]),
            DatasetTransformation(
                drop_nulls,
                data_kwargs={"df": "inherit"},
                kwargs={"threshold": null_thresh},
                outputs=["data"],
            ),
            DatasetTransformation(
                drop_correlated,
                data_kwargs={"data": "inherit"},
                kwargs={"thresh": correlation_threshold},
                outputs=["df"],
            ),
        ]

        # ⬆️ upload the cleaned datasets
        self.controller_1.upload_dataset(
            dataset=df4,
            dataset_name="Auto Insurance Fraud - Cleaned",
            metadata=processed_metadata,
            transformations=cleaning_transformations,
        )

        def fill_nan_const(df, val):
            """Fill NaN values in the dataframe with a constant value"""
            return df.replace(["None", np.nan], val)

        # Fill nans in 1st dataset with -1
        const_val = -1
        df_const = fill_nan_const(df4, const_val)

        def fill_nan_mode(df, columns):
            """
            Fills nans in specified columns with the mode of that column
            Note that we want to make sure to not modify the dataset we passed in but to
            return a new copy.
            We do that by making a copy and specifying deep=True.
            """
            new_df = df.copy(deep=True)
            for col in df.columns:
                if col in columns:
                    new_df[col] = df[col].fillna(df[col].mode()[0])
            return new_df

        nan_cols = ["collision_type", "property_damage", "police_report_available"]
        df_mode = fill_nan_mode(df4, nan_cols)

        # find columns with categorical data for both dataset
        cat_cols = df_const.select_dtypes(include=["object"]).columns.tolist()

        def encode_categorical(df, cat_cols):
            from sklearn.preprocessing import LabelEncoder

            new_df = df.copy(deep=True)
            for col in cat_cols:
                if col in df.columns:
                    le = LabelEncoder()
                    le.fit(list(df[col].astype(str).values))
                    new_df[col] = le.transform(list(df[col].astype(str).values))
            return new_df

        df_const = encode_categorical(df_const, cat_cols)
        df_mode = encode_categorical(df_mode, cat_cols)

        # 🔀 define the transformations - for the constant fill dataset
        const_processed_transformations = [
            DatasetTransformation(
                fill_nan_const, data_kwargs={"df": df4}, kwargs={"val": const_val}, outputs=["df"]
            ),
            DatasetTransformation(
                encode_categorical,
                data_kwargs={"df": "inherit"},
                kwargs={"cat_cols": cat_cols},
                outputs=["df"],
            ),
        ]

        # ⬆️ upload the constant fill dataset
        self.controller_1.upload_dataset(
            dataset=df_const,
            dataset_name="Auto Insurance Fraud - Const Fill",
            metadata=processed_metadata,
            transformations=const_processed_transformations,
        )

        # 🔀 define the transformations - for the mode fill dataset
        mode_processed_transformations = [
            DatasetTransformation(
                fill_nan_mode, data_kwargs={"df": df4}, kwargs={"columns": nan_cols}, outputs=["df"]
            ),
            DatasetTransformation(
                encode_categorical,
                data_kwargs={"df": "inherit"},
                kwargs={"cat_cols": cat_cols},
                outputs=["df"],
            ),
        ]

        # ⬆️ upload the mode fill dataset
        self.controller_1.upload_dataset(
            dataset=df_mode,
            dataset_name="Auto Insurance Fraud - Mode Fill",
            metadata=processed_metadata,
            transformations=mode_processed_transformations,
        )

        def get_samples_labels(df, output_col):
            X = df.drop(output_col, axis=1)
            y = df[output_col]

            return X, y

        # split the datasets into samples and labels ready for modelling.
        self.X_const, self.y_const = get_samples_labels(df_const, "fraud_reported")
        self.X_mode, self.y_mode = get_samples_labels(df_mode, "fraud_reported")

        def get_test_train_splits(X, y, test_size, random_state):
            from sklearn.model_selection import train_test_split

            return train_test_split(
                X, y, test_size=test_size, stratify=y, random_state=random_state
            )
            # returns X_train, X_test, y_train, y_test

        # split into test and train sets
        (
            self.X_train_const,
            self.X_test_const,
            self.y_train_const,
            self.y_test_const,
        ) = get_test_train_splits(self.X_const, self.y_const, test_size=0.2, random_state=42)
        (
            self.X_train_mode,
            self.X_test_mode,
            self.y_train_mode,
            self.y_test_mode,
        ) = get_test_train_splits(self.X_mode, self.y_mode, test_size=0.2, random_state=42)

        # 🔀 define the transformations - for the constant fill training set
        const_train_transformations = [
            DatasetTransformation(
                get_test_train_splits,
                data_kwargs={"X": self.X_const, "y": self.y_const},
                kwargs={"test_size": 0.2, "random_state": 42},
                outputs=["X_train_const", None, "y_train_const", None],
                split="train",
            ),
        ]

        # ⬆️ upload the const fill training set
        self.controller_1.upload_dataset_split(
            X=self.X_train_const,
            y=self.y_train_const,
            dataset_name="Auto Insurance Fraud - Const Fill - Train",
            metadata=processed_metadata,
            transformations=const_train_transformations,
        )

        # 🔀 define the transformations - for the constant fill test set
        const_test_transformations = [
            DatasetTransformation(
                get_test_train_splits,
                data_kwargs={"X": self.X_const, "y": self.y_const},
                kwargs={"test_size": 0.2, "random_state": 42},
                outputs=[None, "X_test_const", None, "y_test_const"],
                split="test",
            ),
        ]

        # ⬆️ upload the const fill test set
        self.controller_1.upload_dataset_split(
            X=self.X_test_const,
            y=self.y_test_const,
            dataset_name="Auto Insurance Fraud - Const Fill - Test",
            metadata=processed_metadata,
            transformations=const_test_transformations,
        )

        # 🔀 define the transformations - for the mode fill training set
        mode_train_transformations = [
            DatasetTransformation(
                get_test_train_splits,
                data_kwargs={"X": self.X_mode, "y": self.y_mode},
                kwargs={"test_size": 0.2, "random_state": 42},
                outputs=["X_train_mode", None, "y_train_mode", None],
                split="train",
            ),
        ]

        # ⬆️ upload the mode fill train set
        self.controller_1.upload_dataset_split(
            X=self.X_train_mode,
            y=self.y_train_mode,
            dataset_name="Auto Insurance Fraud - Mode Fill - Train",
            metadata=processed_metadata,
            transformations=mode_train_transformations,
        )

        # 🔀 define the transformations - for the mode fill test set
        mode_test_transformations = [
            DatasetTransformation(
                get_test_train_splits,
                data_kwargs={"X": self.X_mode, "y": self.y_mode},
                kwargs={"test_size": 0.2, "random_state": 42},
                outputs=[None, "X_test_mode", None, "y_test_mode"],
                split="test",
            ),
        ]

        # ⬆️ upload the mode fill test set
        self.controller_1.upload_dataset_split(
            X=self.X_test_mode,
            y=self.y_test_mode,
            dataset_name="Auto Insurance Fraud - Mode Fill - Test",
            metadata=processed_metadata,
            transformations=mode_test_transformations,
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

        # balance the training sets - creating new training sets for comparison
        self.X_train_const_smote, self.y_train_const_smote = smote_balance(
            self.X_train_const, self.y_train_const, random_state=42
        )
        self.X_train_mode_smote, self.y_train_mode_smote = smote_balance(
            self.X_train_mode, self.y_train_mode, random_state=42
        )

        # 🔀 define the transformations - for the constant fill balanced train set
        const_smote_transformations = [
            DatasetTransformation(
                smote_balance,
                data_kwargs={"X": self.X_train_const, "y": self.y_train_const},
                kwargs={"random_state": 42},
                outputs=["X", "y"],
            ),
        ]

        # ⬆️ upload the constant fill balanced train set
        self.controller_1.upload_dataset_split(
            X=self.X_train_const_smote,
            y=self.y_train_const_smote,
            dataset_name="Auto Insurance Fraud - Const Fill - Smote Train",
            metadata=processed_metadata,
            transformations=const_smote_transformations,
        )

        # 🔀 define the transformations - for the mode fill balanced train set
        mode_smote_transformations = [
            DatasetTransformation(
                smote_balance,
                data_kwargs={"X": self.X_train_mode, "y": self.y_train_mode},
                kwargs={"random_state": 42},
                outputs=["X", "y"],
            ),
        ]

        # ⬆️ upload the mode fill balanced train set
        self.controller_1.upload_dataset_split(
            X=self.X_train_mode_smote,
            y=self.y_train_mode_smote,
            dataset_name="Auto Insurance Fraud - Mode Fill - Smote Train",
            metadata=processed_metadata,
            transformations=mode_smote_transformations,
        )

    def step_3_upload_training_runs(self):
        # define model

        from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
        from sklearn.metrics import accuracy_score
        from sklearn.model_selection import cross_val_score
        from sklearn.tree import DecisionTreeClassifier

        classifiers = {
            "RandomForestClassifier": RandomForestClassifier(),
            "DecisionTreeClassifier": DecisionTreeClassifier(),
            "GradientBoostingClassifier": GradientBoostingClassifier(),
        }

        datasets = [
            (
                "Const Fill",
                (self.X_train_const, self.X_test_const, self.y_train_const, self.y_test_const),
            ),
            (
                "Mode Fill",
                (self.X_train_mode, self.X_test_mode, self.y_train_mode, self.y_test_mode),
            ),
            (
                "Const Fill Smote",
                (
                    self.X_train_const_smote,
                    self.X_test_const,
                    self.y_train_const_smote,
                    self.y_test_const,
                ),
            ),
            (
                "Mode Fill Smote",
                (
                    self.X_train_mode_smote,
                    self.X_test_mode,
                    self.y_train_mode_smote,
                    self.y_test_mode,
                ),
            ),
        ]

        for name, (X_train, X_test, y_train, y_test) in datasets:

            for key, classifier in classifiers.items():
                # cross validate to get an idea of generalisation.
                training_score = cross_val_score(classifier, X_train, y_train, cv=5)

                # train on the full training set
                classifier.fit(X_train, y_train)

                # ⬆️ upload the fully trained model
                self.controller_1.upload_training_run_split(
                    model=classifier, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
                )

                # test accuracy
                y_preds = classifier.predict(X_test)
                test_score = accuracy_score(y_test, y_preds)
                print(
                    f"Classifier: {classifier.__class__.__name__} has a training score of {round(training_score.mean(), 3) * 100}% accuracy score on {name}"
                )
                print(
                    f"Classifier: {classifier.__class__.__name__} has a test score of {round(test_score, 3) * 100}% accuracy score on {name}"
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
                traceback.print_exc()
                self.fail(f"{step} failed ({type(e)}: {e})")
