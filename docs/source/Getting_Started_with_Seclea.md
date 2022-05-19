<img src="https://github.com/seclea/seclea_ai/raw/dev/docs/media/logos/logo-light.png" width="400" alt="Seclea" />

# Getting Started

We will run through a sample project showing how to use Seclea's tools to record your data science work
and explore the results in the Seclea Platform.


## Set up the Project

Head to [platform.seclea.com](https://platform.seclea.com) and log in.

Create a new project and give it a name and description.

![](https://github.com/seclea/seclea_ai/raw/dev/docs/media/notebooks/getting_started/create-new-project.png)
![](https://github.com/seclea/seclea_ai/raw/dev/docs/media/notebooks/getting_started/create-project-name-description.png)

- Go to project settings
- Select Compliance, Risk and Performance Templates for this project.

These are optional but are needed to take advantage of Checks. If in doubt leave these empty for now and come back.

## Integrate with seclea-ai

You can get the seclea-ai package from either pip or conda-forge - whichever you prefer!


```python
!pip install seclea_ai
# !conda install seclea_ai
```

When you initialise the SecleaAI object you will be prompted to login if you haven't already done so.
Use the same Project Name you used earlier and the Organization name provided with your credentials.



```python
from seclea_ai import SecleaAI

# NOTE - use the organization name provided to you when issued credentials.
seclea = SecleaAI(project_name="Car Insurance Fraud Detection", organization='Onespan', platform_url="http://localhost:8000", auth_url="http://localhost:8010")
```

## 🗄 Handle the Data

Download the [data](https://raw.githubusercontent.com/seclea/seclea_ai/dev/docs/examples/insurance_claims.csv) for
this tutorial if you are working on this in Colab or without reference to the repo - this is an Insurance Claims dataset with a variety of features and 1000 samples.

Now we can upload the initial data to the Seclea Platform. 

This should include whatever information we know about the dataset at this point as metadata. 
There are only two keys to add in metadata for now - outcome_name and continuous_features.

You can leave out outcome_name if you haven't decided what you will be predicting yet, but you should
know or be able to find out the continuous features at this point.

You can also update these when uploading datasets
during/after pre-processing. 



```python
import numpy as np
import pandas as pd

# load the data 
data = pd.read_csv('insurance_claims.csv', index_col="policy_number")

# define the metadata for the dataset.
dataset_metadata = {"outcome_name": "fraud_reported",
                    "favourable_outcome": "N",
                    "unfavourable_outcome": "Y",
                    "continuous_features": [
                                            "total_claim_amount",
                                            'policy_annual_premium',
                                            'capital-gains',
                                            'capital-loss',
                                            'injury_claim',
                                            'property_claim',
                                            'vehicle_claim',
                                            'incident_hour_of_the_day',
                                            ]}


# ⬆️ upload the dataset - pick a meaningful name here, you'll be seeing it a lot on the platform!
seclea.upload_dataset(dataset=data, dataset_name="Auto Insurance Fraud", metadata=dataset_metadata)

```

### 🔍 Evaluating the Dataset

Head back to the platform, so we can take a look at our Dataset

Navigate to the Datasets section - under Prepare tab. See the preview and use the format check/PII check.

PII  and Format Check

Bias Check

Include screen shots.

## 🔀 Transformations

When using Seclea to record your Data Science work you will have to take care with how you
deal with transformations of the data.

We require that all transformations are encapsulated in a function, that takes the data and returns the
transformed data. There are a few things to be aware of so please see the [docs](https://docs.seclea.com) for more.


```python
# Create a copy to isolate the original dataset
df1 = data.copy(deep=True)

def encode_nans(df):
    # convert the special characters to nans
    return df.replace('?', np.NaN)

df2 = encode_nans(df1)
```

## 🧽 Data Cleaning

We will carry out some pre-processing and generate a few different datasets so that we
can see on the platform how to track these.This also means we can train our models on some
different data and see how that affects performance.


```python
## Drop the the column which are more than some proportion NaN values
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
```

## ⬆️ Upload Intermediate Dataset

Before getting to balancing the datasets we will upload them to the Seclea Platform.

- We define the metadata for the dataset - if there have been any changes since the original dataset we need to put that here, otherwise we can reuse the original metadata. In this case we have dropped some of the continuous feature columns so we will need to redefine

- We define the transformations that took place between the last state we uploaded and this dataset. This is a list of functions and arguments. See docs.seclea.com for more details of the correct formatting.




```python
from seclea_ai.transformations import DatasetTransformation

# define the updates to the metadata - empty as there are no changes
processed_metadata = {}

# 🔀 define the transformations - note the arguments
cleaning_transformations = [
            DatasetTransformation(encode_nans, data_kwargs={"df": df1}, kwargs={}, outputs=["df"]),
            DatasetTransformation(
                drop_nulls, data_kwargs={"df": "inherit"}, kwargs={"threshold": null_thresh}, outputs=["data"]
            ),
            DatasetTransformation(
                drop_correlated, data_kwargs={"data": "inherit"}, kwargs={"thresh": correlation_threshold}, outputs=["df"]
            ),
        ]

# ⬆️ upload the cleaned datasets
seclea.upload_dataset(dataset=df4,
                      dataset_name="Auto Insurance Fraud - Cleaned",
                      metadata=processed_metadata,
                      transformations=cleaning_transformations)

```


```python
def fill_nan_const(df, val):
    """Fill NaN values in the dataframe with a constant value"""
    return df.replace(['None', np.nan], val)


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


nan_cols = ['collision_type','property_damage', 'police_report_available']
df_mode = fill_nan_mode(df4, nan_cols)


# find columns with categorical data for both dataset
cat_cols = df_const.select_dtypes(include=['object']).columns.tolist()

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
    DatasetTransformation(fill_nan_const, data_kwargs={"df": df4}, kwargs={"val": const_val}, outputs=["df"]),
    DatasetTransformation(encode_categorical, data_kwargs={"df": "inherit"}, kwargs={"cat_cols":cat_cols}, outputs=["df"]),
]

# ⬆️ upload the constant fill dataset
seclea.upload_dataset(dataset=df_const, 
                      dataset_name="Auto Insurance Fraud - Const Fill", 
                      metadata=processed_metadata,
                      transformations=const_processed_transformations)

# 🔀 define the transformations - for the mode fill dataset
mode_processed_transformations = [
    DatasetTransformation(fill_nan_mode, data_kwargs={"df": df4}, kwargs={"columns": nan_cols}, outputs=["df"]),
    DatasetTransformation(encode_categorical, data_kwargs={"df": "inherit"}, kwargs={"cat_cols": cat_cols}, outputs=["df"]),
]

# ⬆️ upload the mode fill dataset
seclea.upload_dataset(dataset=df_mode,
                      dataset_name="Auto Insurance Fraud - Mode Fill",
                      metadata=processed_metadata,
                      transformations=mode_processed_transformations)

def get_samples_labels(df, output_col):
    X = df.drop(output_col, axis=1)
    y = df[output_col]

    return X, y

# split the datasets into samples and labels ready for modelling.
X_const, y_const = get_samples_labels(df_const, "fraud_reported")
X_mode, y_mode = get_samples_labels(df_mode, "fraud_reported")

def get_test_train_splits(X, y, test_size, random_state):
    from sklearn.model_selection import train_test_split

    return train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    # returns X_train, X_test, y_train, y_test

# split into test and train sets
X_train_const, X_test_const, y_train_const, y_test_const = get_test_train_splits(X_const, y_const, test_size=0.2, random_state=42)
X_train_mode, X_test_mode, y_train_mode, y_test_mode = get_test_train_splits(X_mode, y_mode, test_size=0.2, random_state=42)

# 🔀 define the transformations - for the constant fill training set
const_train_transformations = [
    DatasetTransformation(
            get_test_train_splits,
            data_kwargs={"X": X_const, "y": y_const},
            kwargs={"test_size": 0.2, "random_state": 42},
            outputs=["X_train_const", None, "y_train_const", None],
            split="train",
            ),
]

# ⬆️ upload the const fill training set
seclea.upload_dataset_split(
                        X=X_train_const,
                        y=y_train_const,
                        dataset_name="Auto Insurance Fraud - Const Fill - Train",
                        metadata=processed_metadata,
                        transformations=const_train_transformations
)

# 🔀 define the transformations - for the constant fill test set
const_test_transformations = [
    DatasetTransformation(
            get_test_train_splits,
            data_kwargs={"X": X_const, "y": y_const},
            kwargs={"test_size": 0.2, "random_state": 42},
            outputs=[None, "X_test_const", None, "y_test_const"],
            split="test"
            ),
]

# ⬆️ upload the const fill test set
seclea.upload_dataset_split(X=X_test_const,
                      y=y_test_const,
                      dataset_name="Auto Insurance Fraud - Const Fill - Test",
                      metadata=processed_metadata,
                      transformations=const_test_transformations)

# 🔀 define the transformations - for the mode fill training set
mode_train_transformations = [
    DatasetTransformation(
            get_test_train_splits,
            data_kwargs={"X": X_mode, "y": y_mode},
            kwargs={"test_size": 0.2, "random_state": 42},
            outputs=["X_train_mode", None, "y_train_mode", None],
            split="train",
            ),
]

# ⬆️ upload the mode fill train set
seclea.upload_dataset_split(X=X_train_mode,
                      y=y_train_mode,
                      dataset_name="Auto Insurance Fraud - Mode Fill - Train",
                      metadata=processed_metadata,
                      transformations=mode_train_transformations)

# 🔀 define the transformations - for the mode fill test set
mode_test_transformations = [
    DatasetTransformation(
            get_test_train_splits,
            data_kwargs={"X": X_mode, "y": y_mode},
            kwargs={"test_size": 0.2, "random_state": 42},
            outputs=[None, "X_test_mode", None, "y_test_mode"],
            split="test",
            ),
]

# ⬆️ upload the mode fill test set
seclea.upload_dataset_split(X=X_test_mode,
                      y=y_test_mode,
                      dataset_name="Auto Insurance Fraud - Mode Fill - Test",
                      metadata=processed_metadata,
                      transformations=mode_test_transformations)



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
X_train_const_smote, y_train_const_smote = smote_balance(X_train_const, y_train_const, random_state=42)
X_train_mode_smote, y_train_mode_smote = smote_balance(X_train_mode, y_train_mode, random_state=42)

# 🔀 define the transformations - for the constant fill balanced train set
const_smote_transformations = [
    DatasetTransformation(
            smote_balance,
            data_kwargs={"X": X_train_const, "y": y_train_const},
            kwargs={"random_state": 42},
            outputs=["X", "y"]
            ),
]

# ⬆️ upload the constant fill balanced train set
seclea.upload_dataset_split(X=X_train_const_smote,
                      y=y_train_const_smote,
                      dataset_name="Auto Insurance Fraud - Const Fill - Smote Train",
                      metadata=processed_metadata,
                      transformations=const_smote_transformations)

# 🔀 define the transformations - for the mode fill balanced train set
mode_smote_transformations = [
    DatasetTransformation(
            smote_balance,
            data_kwargs={"X": X_train_mode, "y": y_train_mode},
            kwargs={"random_state": 42},
            outputs=["X", "y"]
            ),
]

# ⬆️ upload the mode fill balanced train set
seclea.upload_dataset_split(X=X_train_mode_smote,
                      y=y_train_mode_smote,
                      dataset_name="Auto Insurance Fraud - Mode Fill - Smote Train",
                      metadata=processed_metadata,
                      transformations=mode_smote_transformations)
```

    Shape of X before SMOTE: (800, 35)
        Shape of X after SMOTE: (1204, 35)
    Shape of y before SMOTE: (800,)
        Shape of y after SMOTE: (1204,)
    Shape of X before SMOTE: (800, 35)
        Shape of X after SMOTE: (1204, 35)
    Shape of y before SMOTE: (800,)
        Shape of y after SMOTE: (1204,)
    Shape of X before SMOTE: (800, 35)
        Shape of X after SMOTE: (1204, 35)
    Shape of y before SMOTE: (800,)
        Shape of y after SMOTE: (1204,)
    Shape of X before SMOTE: (800, 35)
        Shape of X after SMOTE: (1204, 35)
    Shape of y before SMOTE: (800,)
        Shape of y after SMOTE: (1204,)


### 🔍 Evaluating the Transformations

Now head to platform.seclea.com again to take another look at the Datasets section. You will see that there is a lot more to look at this time.

You can see here how the transformations are used to show you the history of the data and how it arrived in its final state.

# 🛠️ Modeling

Now we get started with the modelling. We will run the same models over each of our datasets to explore how the different processing of the data has affected our results.

We will use three models from sklearn for this, DecisionTree, RandomForest and GradientBoosting Classifers. 


## 📈 Training


```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score

classifiers = {
    "RandomForestClassifier": RandomForestClassifier(),
    "DecisionTreeClassifier": DecisionTreeClassifier(),
    "GradientBoostingClassifier": GradientBoostingClassifier()
}

datasets = [
    ("Const Fill", (X_train_const, X_test_const, y_train_const, y_test_const)),
    ("Mode Fill", (X_train_mode, X_test_mode, y_train_mode, y_test_mode)),
    ("Const Fill Smote", (X_train_const_smote, X_test_const, y_train_const_smote, y_test_const)),
    ("Mode Fill Smote", (X_train_mode_smote, X_test_mode, y_train_mode_smote, y_test_mode))
    ]

for name, (X_train, X_test, y_train, y_test) in datasets:

    for key, classifier in classifiers.items():
        # cross validate to get an idea of generalisation.
        training_score = cross_val_score(classifier, X_train, y_train, cv=5)

        # train on the full training set
        classifier.fit(X_train, y_train)

        # ⬆️ upload the fully trained model
        seclea.upload_training_run_split(model=classifier, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)

        # test accuracy
        y_preds = classifier.predict(X_test)
        test_score = accuracy_score(y_test, y_preds)
        print(f"Classifier: {classifier.__class__.__name__} has a training score of {round(training_score.mean(), 3) * 100}% accuracy score on {name}")
        print(f"Classifier: {classifier.__class__.__name__} has a test score of {round(test_score, 3) * 100}% accuracy score on {name}")
```

## 🔍 Analysis

Head back to [platform.seclea.com](https://platform.seclea.com) and we can analyse our Models


