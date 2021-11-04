from seclea_ai import SecleaAI
from seclea_ai import Frameworks

from unittest import TestCase
import pandas as pd
import numpy as np
import uuid


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
        self.password = 'asdf'
        self.username = 'onespanadmin'
        self.project_name = f'test-project{uuid.uuid4()}'
        self.portal_url = 'http://localhost:8000'
        self.auth_url = 'http://localhost:8010'
        self.controller = SecleaAI(self.project_name, self.portal_url, self.auth_url, username=self.username,
                                   password=self.password)

    def step_1_upload_dataset(self):
        self.sample_df_1 = pd.read_csv('insurance_claims.csv')
        self.sample_df_1_name = 'test_dataset_1'
        self.sample_df_1_meta = {"index": None, "outcome_name": "fraud_reported",
                                 "continuous_features": ["total_claim_amount", 'policy_annual_premium', 'capital-gains',
                                                         'capital-loss', 'injury_claim', 'property_claim',
                                                         'vehicle_claim', 'incident_hour_of_the_day', ]}
        self.controller.upload_dataset(self.sample_df_1, self.sample_df_1_name, self.sample_df_1_meta)

    def step_2_define_transformations(self):
        # transformations
        def encode_nans(df):
            # dealing with special character
            df['collision_type'] = df['collision_type'].replace('?', np.NaN)
            df['property_damage'] = df['property_damage'].replace('?', np.NaN)
            df['police_report_available'] = df['police_report_available'].replace('?',
                                                                                  "NO")  # default to no police report present if previously ?
            return df

        df = encode_nans(self.sample_df_1)

        def drop_correlated(data, thresh):
            # calculate correlations
            corr_matrix = data.corr().abs()
            # get the upper part of correlation matrix
            upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

            # columns with correlation above threshold
            redundant = [column for column in upper.columns if any(upper[column] >= thresh)]
            print(f'Columns to drop with correlation > {thresh}: {redundant}')
            data.drop(columns=redundant, inplace=True)
            return data

        corr_thresh = 0.97
        df = drop_correlated(df, corr_thresh)

        def drop_nulls(df, threshold):
            cols = [x for x in df.columns if df[x].isnull().sum() / df.shape[0] > threshold]
            return df.drop(columns=cols)

        null_thresh = 0.9
        df = drop_nulls(df, threshold=null_thresh)

        def encode_categorical(df, cat_cols):
            from sklearn.preprocessing import LabelEncoder
            for col in cat_cols:
                if col in df.columns:
                    le = LabelEncoder()
                    le.fit(list(df[col].astype(str).values))
                    df[col] = le.transform(list(df[col].astype(str).values))
            return df

        cat_cols = df.select_dtypes(include=['object']).columns.tolist()
        df = encode_categorical(df, cat_cols)

        df.fillna({"collision_type": -1, "property_damage": -1})

        self.transformations = [encode_nans, (drop_correlated, [corr_thresh]), (drop_nulls, [null_thresh]),
                                (encode_categorical, {"cat_cols": cat_cols})]

        self.sample_df_1_transformed = df

    def step_3_upload_trainingrun(self):
        # define model

        from sklearn.model_selection import train_test_split

        X = self.sample_df_1_transformed.drop('fraud_reported', axis=1)
        y = self.sample_df_1_transformed.fraud_reported

        # train test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        print(f'''% Positive class in Train = {np.round(y_train.value_counts(normalize=True)[1] * 100, 2)}
        % Positive class in Test  = {np.round(y_test.value_counts(normalize=True)[1] * 100, 2)}''')

        # %%

        ### testing without SMOTE

        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score

        # Train
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        self.controller.upload_training_run(model, model_type=model.__class__.__name__, framework=Frameworks.SKLEARN,
                                            dataset_name=self.sample_df_1_name, transformations=self.transformations)

    def _steps(self):
        for name in dir(self):  # dir() result is implicitly sorted
            if name.startswith("step"):
                yield name, getattr(self, name)

    def test_steps(self):
        for name, step in self._steps():
            try:
                step()
                print('STEP COMPLETE')
            except Exception as e:
                self.fail(F"{step} failed ({type(e)}: {e})")
