********
Tutorial
********

Introduction
########

The seclea-ai package is used for tracking ML model development. This means that most of the
functions are designed to record and upload activities or data associated with your modelling.


Usage Guide
#######

There are two main steps to using the package:

- Before you begin modelling you will need to upload the dataset that you are working with. This will currently need to be a csv, a Pandas DataFrame or a list of csv's.

- After each time you train a new model you will upload it using ``SecleaAI.upload_training_run()``.

Example Usage
########

Example of a simple Jupyter Notebook for a project using seclea_ai::

    from seclea_ai import SecleaAI
    import pandas as pd

    seclea = SecleaAI(project_name="Test Project", organization="Test Org")

::

    dataset = pd.read_csv("/content/dataset.csv")

    dataset_metadata = {
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

    seclea.upload_dataset(dataset, dataset_name="Test Dataset", metadata=dataset_metadata

    # Alternative - using a file instead of a loaded DataFrame.
    # seclea.upload_dataset("/content/dataset.csv", dataset_name="Test Dataset", metadata=dataset_metadata)

::

    def remove_correlated_features(dataframe: DataFrame, keep: List[str], threshold: float) -> DataFrame:
        """ Remove strongly correlated features """

        # Absolute value correlation matrix
        corr_matrix = dataframe[dataframe["isFraud"].notnull()].corr().abs()

        # Getting the upper triangle of correlations
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

        # Select columns with correlations above threshold
        to_drop = [column for column in upper.columns if any(upper[column] > threshold) and column not in keep]

        return dataframe.drop(columns=to_drop)


    def encode_categorical(dataframe: DataFrame) -> DataFrame:
        # encode all the categorical features into numerical (ie. [1, 4, 6] -> [0, 1, 2] or ["lights", "signs", ""]
        for col in dataframe.columns:
            if dataframe[col].dtype == "object" or dataframe[col].dtype == "object":
                lbl = LabelEncoder()
                lbl.fit(list(dataframe[col].values))
                dataframe[col] = lbl.transform(list(dataframe[col].values))
        return dataframe

::

    import xgboost as xgb

    threshold = 0.98

    keep = [
        "TransactionAmt_to_mean_card4",
        "TransactionAmt_to_std_card4",
        "D15_to_mean_card4",
        "D15_to_std_card4",
        "Hours",
    ]

    dataset_cleaned = remove_correlated_features(dataset, keep, threshold)

    dataset_cleaned = encode_categorical(dataset_cleaned)

    # load to XGBoost format
    dtrain = xgb.DMatrix(
                         data=dataset_cleaned.drop("isFraud", axis=1),
                         label=dataset_cleaned["isFraud"].copy(deep=True)
                         )

    # setup training params
    params = dict(max_depth=2, eta=1, objective="binary:logistic", nthread=4, eval_metric="auc")
    num_rounds = 5

    # train model
    booster = xgb.train(params=params, dtrain=dtrain, num_boost_round=num_rounds)

    transformations = [DatasetTransformation(
                                            remove_correlated_features,
                                            data_kwargs={"dataframe": dataset},
                                            kwargs={"keep": keep, "threshold": threshold},
                                            outputs=["dataframe"]
                                            ),
                       DatasetTransformation(
                                            encode_categorical,
                                            data_kwargs={"dataframe": inherit},
                                            kwargs={},
                                            outputs=["dataframe"]
                                            )]

    seclea.upload_dataset(
        dataset_cleaned,
        dataset_name="Test Dataset Cleaned",
        metadata=dataset_metadata,
        transformations=transformations,
    )

    # upload model state and data
    seclea.upload_training_run(
        booster,
        dataset=dataset_cleaned,
    )

