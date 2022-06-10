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

    Initial Tokens - Status: 200 - content b'{"first_name":"","last_name":"","username":"onespanadmin","email":"asdf@gmail.com","organization_permissions":{"admin":[{"name":"Onespan","is_active":true}],"data_scientist":[{"name":"Onespan","is_active":true}],"analyst":[{"name":"Onespan","is_active":true}]},"TTL_set":true,"access":"eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiJ9.eyJ0b2tlbl90eXBlIjoiYWNjZXNzIiwiZXhwIjoxNjU0NTEzMDY3LCJqdGkiOiI0OGJiMjFiYTgxMGE0NzVjYjdmZTg4YWJhOGJkMTg1NyIsInVzZXJfaWQiOiJhc2RmQGdtYWlsLmNvbSIsIl9kYXRhIjp7InVzZXIiOnsiZmlyc3RfbmFtZSI6IiIsImxhc3RfbmFtZSI6IiIsInVzZXJuYW1lIjoib25lc3BhbmFkbWluIiwiZW1haWwiOiJhc2RmQGdtYWlsLmNvbSIsIm9yZ2FuaXphdGlvbl9wZXJtaXNzaW9ucyI6eyJhZG1pbiI6W3sibmFtZSI6Ik9uZXNwYW4iLCJpc19hY3RpdmUiOnRydWV9XSwiZGF0YV9zY2llbnRpc3QiOlt7Im5hbWUiOiJPbmVzcGFuIiwiaXNfYWN0aXZlIjp0cnVlfV0sImFuYWx5c3QiOlt7Im5hbWUiOiJPbmVzcGFuIiwiaXNfYWN0aXZlIjp0cnVlfV19fX19.e2Jus0BekAuAqUKDiA4xR_hPk3ZDvj9V0MhjCNeCafjxZ2DRan7wzUxSXaUQqXLaaUCq2uT-EjRDpQOvboILMN_5giN0GbNAZ4_eN8_LcRaxHpdquP4mnjeCtkgl0paTUQkyXHIjMU0EhLADEvG831qFi0h9T0MyaUhF_l0v6NwyBAsiRXbTQirUg355ps11eSFtHl9FY-W9-COfZwmlzkT7Z1ySVCTKKmxXpx3zrgEkv8c7Eq8jXeqkwmRXf7CznH7Fpg1cJs6b2nkFlHfxjL3-6x8AKrngcjup5Jfru94IRkMeVQ0SfAjxFmKEC-Mc8uR9CbK-Jm6s6RfbrpuCtXUigNsP2wdEN4STkNdVyJYGRpuJaCA2ON6QFmIwEUuFwRx_kkFqX6ydPaAwM7fP5cYy_z30DbO85wD0JXj8RhQL_Jk8uCZTOovdJoAjfcvvvh-VuaRcgTCFl25TDIGTfTSBZRV_bfBbzo89Co60Y7eCOzA_shDcf5vuRdkefYdoame42qdWuVoAZV4Z4elTm1kag-MWWrtY6wi44SNZwVZeNgjfV8NA1JbPmdbmzJQIYvPxkR9iAtxrfsFpCnPHQ-x4t7BCed3qQyd5Y_1fRt5B-bXON71lQo0gDPvuK6hrHOBky-_YKcXu6trZzdATDT1TVI4GI-I5kNvHowut4Wc","refresh":"eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiJ9.eyJ0b2tlbl90eXBlIjoicmVmcmVzaCIsImV4cCI6MTY1NDU4NTA2NywianRpIjoiODdkOTg2OWUyODQ1NGFiYTlhMjNhMDdmODk4Mzc2MGIiLCJ1c2VyX2lkIjoiYXNkZkBnbWFpbC5jb20iLCJfZGF0YSI6eyJ1c2VyIjp7ImZpcnN0X25hbWUiOiIiLCJsYXN0X25hbWUiOiIiLCJ1c2VybmFtZSI6Im9uZXNwYW5hZG1pbiIsImVtYWlsIjoiYXNkZkBnbWFpbC5jb20iLCJvcmdhbml6YXRpb25fcGVybWlzc2lvbnMiOnsiYWRtaW4iOlt7Im5hbWUiOiJPbmVzcGFuIiwiaXNfYWN0aXZlIjp0cnVlfV0sImRhdGFfc2NpZW50aXN0IjpbeyJuYW1lIjoiT25lc3BhbiIsImlzX2FjdGl2ZSI6dHJ1ZX1dLCJhbmFseXN0IjpbeyJuYW1lIjoiT25lc3BhbiIsImlzX2FjdGl2ZSI6dHJ1ZX1dfX19fQ.XU1nRCeeuoqchVc8CPhsnDCCaRAc6v2Anhhag-pBgb1Vswd9rU_jmMmskq_dyaAlApeX_i5f22fIrul5bYOcfweIoC23sbYjEvvryMh_l4fkn6lKchQGG37gUMsBiXysVe0BOmaixIN1CYI3e0d-ZE_dKv7RICUvmUCBTSFSUv2oxkBsrnrxHgXeArcPLCNgq2qKGCCAm0vzlgTSClGZFLsCOdJs6JoLFmzg0tCETZqkjo1PXlI5_Z2fM7cgITGDhEHRFuHCdUiXcl3n1deT0pfMC7wh8l4KXDdILPe_9ibvivpz7p8d1i7Mvb1CfxSXUUMc-cIS0XkoV67L2Y32jxOT0ehOCHvnR0ouXnzK8C56SJELbXiUos6hyUuhoOYSSGHvoId5aXk1XjkJvW4-OmJWJ8H0S3SqcUMwUwl_kMnpOXb2e-k1cJff4-5PjBGbSk2u15c1WZ6NihUfhY8T3De7pvpIzaMPFLE2Vwg77mW4C-SVRM0HYaUzqPtdCLU5TAtDV_d1vx6sRniN2KCXxUgLHCAS7Htq7Bpw4XcXl9BVGWNH1kE1FbtQ7BaIL2uvC6IZkzXoSWne-Y-jDNIIcgGl62cz-m4sOtEEmmJ5Xx_kf_mLPWRUOSi5L5qSqH-f61nTVoHkhel3kE-BF5jthB51JMkaEGriKOleCgz17Yo"}' - cookies - <RequestsCookieJar[<Cookie access_token=eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiJ9.eyJ0b2tlbl90eXBlIjoiYWNjZXNzIiwiZXhwIjoxNjU0NTEzMDY3LCJqdGkiOiI0OGJiMjFiYTgxMGE0NzVjYjdmZTg4YWJhOGJkMTg1NyIsInVzZXJfaWQiOiJhc2RmQGdtYWlsLmNvbSIsIl9kYXRhIjp7InVzZXIiOnsiZmlyc3RfbmFtZSI6IiIsImxhc3RfbmFtZSI6IiIsInVzZXJuYW1lIjoib25lc3BhbmFkbWluIiwiZW1haWwiOiJhc2RmQGdtYWlsLmNvbSIsIm9yZ2FuaXphdGlvbl9wZXJtaXNzaW9ucyI6eyJhZG1pbiI6W3sibmFtZSI6Ik9uZXNwYW4iLCJpc19hY3RpdmUiOnRydWV9XSwiZGF0YV9zY2llbnRpc3QiOlt7Im5hbWUiOiJPbmVzcGFuIiwiaXNfYWN0aXZlIjp0cnVlfV0sImFuYWx5c3QiOlt7Im5hbWUiOiJPbmVzcGFuIiwiaXNfYWN0aXZlIjp0cnVlfV19fX19.e2Jus0BekAuAqUKDiA4xR_hPk3ZDvj9V0MhjCNeCafjxZ2DRan7wzUxSXaUQqXLaaUCq2uT-EjRDpQOvboILMN_5giN0GbNAZ4_eN8_LcRaxHpdquP4mnjeCtkgl0paTUQkyXHIjMU0EhLADEvG831qFi0h9T0MyaUhF_l0v6NwyBAsiRXbTQirUg355ps11eSFtHl9FY-W9-COfZwmlzkT7Z1ySVCTKKmxXpx3zrgEkv8c7Eq8jXeqkwmRXf7CznH7Fpg1cJs6b2nkFlHfxjL3-6x8AKrngcjup5Jfru94IRkMeVQ0SfAjxFmKEC-Mc8uR9CbK-Jm6s6RfbrpuCtXUigNsP2wdEN4STkNdVyJYGRpuJaCA2ON6QFmIwEUuFwRx_kkFqX6ydPaAwM7fP5cYy_z30DbO85wD0JXj8RhQL_Jk8uCZTOovdJoAjfcvvvh-VuaRcgTCFl25TDIGTfTSBZRV_bfBbzo89Co60Y7eCOzA_shDcf5vuRdkefYdoame42qdWuVoAZV4Z4elTm1kag-MWWrtY6wi44SNZwVZeNgjfV8NA1JbPmdbmzJQIYvPxkR9iAtxrfsFpCnPHQ-x4t7BCed3qQyd5Y_1fRt5B-bXON71lQo0gDPvuK6hrHOBky-_YKcXu6trZzdATDT1TVI4GI-I5kNvHowut4Wc for localhost.local/>, <Cookie refresh_token=eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiJ9.eyJ0b2tlbl90eXBlIjoicmVmcmVzaCIsImV4cCI6MTY1NDU4NTA2NywianRpIjoiODdkOTg2OWUyODQ1NGFiYTlhMjNhMDdmODk4Mzc2MGIiLCJ1c2VyX2lkIjoiYXNkZkBnbWFpbC5jb20iLCJfZGF0YSI6eyJ1c2VyIjp7ImZpcnN0X25hbWUiOiIiLCJsYXN0X25hbWUiOiIiLCJ1c2VybmFtZSI6Im9uZXNwYW5hZG1pbiIsImVtYWlsIjoiYXNkZkBnbWFpbC5jb20iLCJvcmdhbml6YXRpb25fcGVybWlzc2lvbnMiOnsiYWRtaW4iOlt7Im5hbWUiOiJPbmVzcGFuIiwiaXNfYWN0aXZlIjp0cnVlfV0sImRhdGFfc2NpZW50aXN0IjpbeyJuYW1lIjoiT25lc3BhbiIsImlzX2FjdGl2ZSI6dHJ1ZX1dLCJhbmFseXN0IjpbeyJuYW1lIjoiT25lc3BhbiIsImlzX2FjdGl2ZSI6dHJ1ZX1dfX19fQ.XU1nRCeeuoqchVc8CPhsnDCCaRAc6v2Anhhag-pBgb1Vswd9rU_jmMmskq_dyaAlApeX_i5f22fIrul5bYOcfweIoC23sbYjEvvryMh_l4fkn6lKchQGG37gUMsBiXysVe0BOmaixIN1CYI3e0d-ZE_dKv7RICUvmUCBTSFSUv2oxkBsrnrxHgXeArcPLCNgq2qKGCCAm0vzlgTSClGZFLsCOdJs6JoLFmzg0tCETZqkjo1PXlI5_Z2fM7cgITGDhEHRFuHCdUiXcl3n1deT0pfMC7wh8l4KXDdILPe_9ibvivpz7p8d1i7Mvb1CfxSXUUMc-cIS0XkoV67L2Y32jxOT0ehOCHvnR0ouXnzK8C56SJELbXiUos6hyUuhoOYSSGHvoId5aXk1XjkJvW4-OmJWJ8H0S3SqcUMwUwl_kMnpOXb2e-k1cJff4-5PjBGbSk2u15c1WZ6NihUfhY8T3De7pvpIzaMPFLE2Vwg77mW4C-SVRM0HYaUzqPtdCLU5TAtDV_d1vx6sRniN2KCXxUgLHCAS7Htq7Bpw4XcXl9BVGWNH1kE1FbtQ7BaIL2uvC6IZkzXoSWne-Y-jDNIIcgGl62cz-m4sOtEEmmJ5Xx_kf_mLPWRUOSi5L5qSqH-f61nTVoHkhel3kE-BF5jthB51JMkaEGriKOleCgz17Yo for localhost.local/>]>
    Cookies: {'refresh_token': 'eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiJ9.eyJ0b2tlbl90eXBlIjoicmVmcmVzaCIsImV4cCI6MTY1NDU3OTAyMiwianRpIjoiMmRkMGQzNjA2Y2IyNGJjYTlkNTk0OGE5OTljNTlkNzciLCJ1c2VyX2lkIjoiYXNkZkBnbWFpbC5jb20iLCJfZGF0YSI6eyJ1c2VyIjp7ImZpcnN0X25hbWUiOiIiLCJsYXN0X25hbWUiOiIiLCJ1c2VybmFtZSI6Im9uZXNwYW5hZG1pbiIsImVtYWlsIjoiYXNkZkBnbWFpbC5jb20iLCJvcmdhbml6YXRpb25fcGVybWlzc2lvbnMiOnsiYWRtaW4iOlt7Im5hbWUiOiJPbmVzcGFuIiwiaXNfYWN0aXZlIjp0cnVlfV0sImRhdGFfc2NpZW50aXN0IjpbeyJuYW1lIjoiT25lc3BhbiIsImlzX2FjdGl2ZSI6dHJ1ZX1dLCJhbmFseXN0IjpbeyJuYW1lIjoiT25lc3BhbiIsImlzX2FjdGl2ZSI6dHJ1ZX1dfX19fQ.DWpc7PJFQm1vnKgVUze-h_U9K9KjdbpdPd_nqLr_sp3IdKYA9t0yDFJ_59lCjLnFOfkvJEp054GYYJGRbUB9_Cp3RkgG8ubKOoRlanSAN7OV-hs9EW32Au_LkakmUZ-lC5b-bwp5erP22HlXAlEUzCzszAfjQ5tJvxgrIJvoW8rOGvfk6YLyR7AL4Zb0AfcCXZ44jC_S3iMji2v6MzGifblzDk7gBs34BaWRVHdRR28S9Cfs5q88WAzwNfEBKDIgpKkVo4rLwuiriSc8mT0nuFu2pU6ipoU1ewOlXj5RiB81gRaZle2IBTA5KEirUzCQMonOHGLmUIMEv7yWUDw8RrHur2qqvjBg-Bj5hzXzqyUnCzj-7bExPbgjpndrRCUKG3FCH6WX39UiShSIYmz2T9k-uNFOLCKRLZqiu_uchvPuTeidJXRNkngzju2DHK3t67JigPEpY5y81MMlYtUB6eAnmZ8qJxeCtVhgRN_sv0WEsURwpKOBjvLixubXfn7W6L0ANL9t1axTZgEhky0Q2N4LkAyZN-ddMEKZNdiEWtAdWnPre3OBgzLFKtEYl2lij5alWizZj_UQaXtRsGMU7tU5NfpJcIKsoQiVDpPnHncntHwbECi4Ws7nF_ua1ElND6AX3UyWYsDgOpbjoWGgmBTwKzWz8pVxY7xgisjjUnM', 'access_token': 'eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiJ9.eyJ0b2tlbl90eXBlIjoiYWNjZXNzIiwiZXhwIjoxNjU0NTEzMDY3LCJqdGkiOiI0OGJiMjFiYTgxMGE0NzVjYjdmZTg4YWJhOGJkMTg1NyIsInVzZXJfaWQiOiJhc2RmQGdtYWlsLmNvbSIsIl9kYXRhIjp7InVzZXIiOnsiZmlyc3RfbmFtZSI6IiIsImxhc3RfbmFtZSI6IiIsInVzZXJuYW1lIjoib25lc3BhbmFkbWluIiwiZW1haWwiOiJhc2RmQGdtYWlsLmNvbSIsIm9yZ2FuaXphdGlvbl9wZXJtaXNzaW9ucyI6eyJhZG1pbiI6W3sibmFtZSI6Ik9uZXNwYW4iLCJpc19hY3RpdmUiOnRydWV9XSwiZGF0YV9zY2llbnRpc3QiOlt7Im5hbWUiOiJPbmVzcGFuIiwiaXNfYWN0aXZlIjp0cnVlfV0sImFuYWx5c3QiOlt7Im5hbWUiOiJPbmVzcGFuIiwiaXNfYWN0aXZlIjp0cnVlfV19fX19.e2Jus0BekAuAqUKDiA4xR_hPk3ZDvj9V0MhjCNeCafjxZ2DRan7wzUxSXaUQqXLaaUCq2uT-EjRDpQOvboILMN_5giN0GbNAZ4_eN8_LcRaxHpdquP4mnjeCtkgl0paTUQkyXHIjMU0EhLADEvG831qFi0h9T0MyaUhF_l0v6NwyBAsiRXbTQirUg355ps11eSFtHl9FY-W9-COfZwmlzkT7Z1ySVCTKKmxXpx3zrgEkv8c7Eq8jXeqkwmRXf7CznH7Fpg1cJs6b2nkFlHfxjL3-6x8AKrngcjup5Jfru94IRkMeVQ0SfAjxFmKEC-Mc8uR9CbK-Jm6s6RfbrpuCtXUigNsP2wdEN4STkNdVyJYGRpuJaCA2ON6QFmIwEUuFwRx_kkFqX6ydPaAwM7fP5cYy_z30DbO85wD0JXj8RhQL_Jk8uCZTOovdJoAjfcvvvh-VuaRcgTCFl25TDIGTfTSBZRV_bfBbzo89Co60Y7eCOzA_shDcf5vuRdkefYdoame42qdWuVoAZV4Z4elTm1kag-MWWrtY6wi44SNZwVZeNgjfV8NA1JbPmdbmzJQIYvPxkR9iAtxrfsFpCnPHQ-x4t7BCed3qQyd5Y_1fRt5B-bXON71lQo0gDPvuK6hrHOBky-_YKcXu6trZzdATDT1TVI4GI-I5kNvHowut4Wc'}
    success


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
correlation_threshold = 0.95
df4 = drop_correlated(df3, correlation_threshold)
```

    Columns to drop with correlation > 0.95: ['vehicle_claim']


    /var/folders/yg/2b7814tx3js7jgnyw_cdydn80000gn/T/ipykernel_40795/338996727.py:16: DeprecationWarning: `np.bool` is a deprecated alias for the builtin `bool`. To silence this warning, use `bool` by itself. Doing this will not modify any behavior and is safe. If you specifically wanted the numpy scalar type, use `np.bool_` here.
    Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
      upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))


## ⬆️ Upload Intermediate Dataset

Before getting to balancing the datasets we will upload them to the Seclea Platform.

- We define the metadata for the dataset - if there have been any changes since the original dataset we need to put that here, otherwise we can reuse the original metadata. In this case we have dropped some of the continuous feature columns so we will need to redefine

- We define the transformations that took place between the last state we uploaded and this dataset. This is a list of functions and arguments. See docs.seclea.com for more details of the correct formatting.




```python
from seclea_ai.transformations import DatasetTransformation

# define the updates to the metadata - only changes are updated - here a continuous feature has been dropped so now
# we remove it from the list of continuous features.
processed_metadata = {"continuous_features": [
                                            "total_claim_amount",
                                            'policy_annual_premium',
                                            'capital-gains',
                                            'capital-loss',
                                            'injury_claim',
                                            'property_claim',
                                            'incident_hour_of_the_day',
                                            ]}

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

    /var/folders/yg/2b7814tx3js7jgnyw_cdydn80000gn/T/ipykernel_40795/338996727.py:16: DeprecationWarning: `np.bool` is a deprecated alias for the builtin `bool`. To silence this warning, use `bool` by itself. Doing this will not modify any behavior and is safe. If you specifically wanted the numpy scalar type, use `np.bool_` here.
    Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
      upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))


    Columns to drop with correlation > 0.95: ['vehicle_claim']



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

# Update metadata with new encoded values for the outcome column.
encoded_metadata = {"favourable_outcome": 0,
                    "unfavourable_outcome": 1,}


# 🔀 define the transformations - for the constant fill dataset
const_processed_transformations = [
    DatasetTransformation(fill_nan_const, data_kwargs={"df": df4}, kwargs={"val": const_val}, outputs=["df"]),
    DatasetTransformation(encode_categorical, data_kwargs={"df": "inherit"}, kwargs={"cat_cols":cat_cols}, outputs=["df"]),
]

# ⬆️ upload the constant fill dataset
seclea.upload_dataset(dataset=df_const, 
                      dataset_name="Auto Insurance Fraud - Const Fill", 
                      metadata=encoded_metadata,
                      transformations=const_processed_transformations)

# 🔀 define the transformations - for the mode fill dataset
mode_processed_transformations = [
    DatasetTransformation(fill_nan_mode, data_kwargs={"df": df4}, kwargs={"columns": nan_cols}, outputs=["df"]),
    DatasetTransformation(encode_categorical, data_kwargs={"df": "inherit"}, kwargs={"cat_cols": cat_cols}, outputs=["df"]),
]

# ⬆️ upload the mode fill dataset
seclea.upload_dataset(dataset=df_mode,
                      dataset_name="Auto Insurance Fraud - Mode Fill",
                      metadata=encoded_metadata,
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
                        metadata={},
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
                      metadata={},
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
                      metadata={},
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
                      metadata={},
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
                      metadata={},
                      transformations=mode_smote_transformations)
```

    Shape of X before SMOTE: (800, 36)
        Shape of X after SMOTE: (1204, 36)
    Shape of y before SMOTE: (800,)
        Shape of y after SMOTE: (1204,)
    Shape of X before SMOTE: (800, 36)
        Shape of X after SMOTE: (1204, 36)
    Shape of y before SMOTE: (800,)
        Shape of y after SMOTE: (1204,)
    Shape of X before SMOTE: (800, 36)
        Shape of X after SMOTE: (1204, 36)
    Shape of y before SMOTE: (800,)
        Shape of y after SMOTE: (1204,)
    Shape of X before SMOTE: (800, 36)
        Shape of X after SMOTE: (1204, 36)
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

    Cookies: {'refresh_token': 'eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiJ9.eyJ0b2tlbl90eXBlIjoicmVmcmVzaCIsImV4cCI6MTY1NDU4NTA2NywianRpIjoiODdkOTg2OWUyODQ1NGFiYTlhMjNhMDdmODk4Mzc2MGIiLCJ1c2VyX2lkIjoiYXNkZkBnbWFpbC5jb20iLCJfZGF0YSI6eyJ1c2VyIjp7ImZpcnN0X25hbWUiOiIiLCJsYXN0X25hbWUiOiIiLCJ1c2VybmFtZSI6Im9uZXNwYW5hZG1pbiIsImVtYWlsIjoiYXNkZkBnbWFpbC5jb20iLCJvcmdhbml6YXRpb25fcGVybWlzc2lvbnMiOnsiYWRtaW4iOlt7Im5hbWUiOiJPbmVzcGFuIiwiaXNfYWN0aXZlIjp0cnVlfV0sImRhdGFfc2NpZW50aXN0IjpbeyJuYW1lIjoiT25lc3BhbiIsImlzX2FjdGl2ZSI6dHJ1ZX1dLCJhbmFseXN0IjpbeyJuYW1lIjoiT25lc3BhbiIsImlzX2FjdGl2ZSI6dHJ1ZX1dfX19fQ.XU1nRCeeuoqchVc8CPhsnDCCaRAc6v2Anhhag-pBgb1Vswd9rU_jmMmskq_dyaAlApeX_i5f22fIrul5bYOcfweIoC23sbYjEvvryMh_l4fkn6lKchQGG37gUMsBiXysVe0BOmaixIN1CYI3e0d-ZE_dKv7RICUvmUCBTSFSUv2oxkBsrnrxHgXeArcPLCNgq2qKGCCAm0vzlgTSClGZFLsCOdJs6JoLFmzg0tCETZqkjo1PXlI5_Z2fM7cgITGDhEHRFuHCdUiXcl3n1deT0pfMC7wh8l4KXDdILPe_9ibvivpz7p8d1i7Mvb1CfxSXUUMc-cIS0XkoV67L2Y32jxOT0ehOCHvnR0ouXnzK8C56SJELbXiUos6hyUuhoOYSSGHvoId5aXk1XjkJvW4-OmJWJ8H0S3SqcUMwUwl_kMnpOXb2e-k1cJff4-5PjBGbSk2u15c1WZ6NihUfhY8T3De7pvpIzaMPFLE2Vwg77mW4C-SVRM0HYaUzqPtdCLU5TAtDV_d1vx6sRniN2KCXxUgLHCAS7Htq7Bpw4XcXl9BVGWNH1kE1FbtQ7BaIL2uvC6IZkzXoSWne-Y-jDNIIcgGl62cz-m4sOtEEmmJ5Xx_kf_mLPWRUOSi5L5qSqH-f61nTVoHkhel3kE-BF5jthB51JMkaEGriKOleCgz17Yo', 'access_token': 'eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiJ9.eyJ0b2tlbl90eXBlIjoiYWNjZXNzIiwiZXhwIjoxNjU0NTEzMDk5LCJqdGkiOiI5NGJjY2FhNjhhYTM0OGJiOTliMDQ4M2VmYmY1NWVhOSIsInVzZXJfaWQiOiJhc2RmQGdtYWlsLmNvbSIsIl9kYXRhIjp7InVzZXIiOnsiZmlyc3RfbmFtZSI6IiIsImxhc3RfbmFtZSI6IiIsInVzZXJuYW1lIjoib25lc3BhbmFkbWluIiwiZW1haWwiOiJhc2RmQGdtYWlsLmNvbSIsIm9yZ2FuaXphdGlvbl9wZXJtaXNzaW9ucyI6eyJhZG1pbiI6W3sibmFtZSI6Ik9uZXNwYW4iLCJpc19hY3RpdmUiOnRydWV9XSwiZGF0YV9zY2llbnRpc3QiOlt7Im5hbWUiOiJPbmVzcGFuIiwiaXNfYWN0aXZlIjp0cnVlfV0sImFuYWx5c3QiOlt7Im5hbWUiOiJPbmVzcGFuIiwiaXNfYWN0aXZlIjp0cnVlfV19fX19.RkdW7IHsIz4zO3h0Zuuc0GE3NgsRokq5hpmvld8_joM0c1ocTCTOas4QeB4IUHy5p8v123iDTQ50KWTfILRK5e4ePcI-CX_wYlgioelyu3hYFLkqiFhPKWxu4PkK-VyTEyAy_bLuswZ5rHOyfCsA-0WEWpWp66dUvZt1lIYXwSGE68LNE7l-UHEfAkYZ9CXHezLR66tqxNcjzDs7uTRvJ-dMCJgE3iOHaZBXl1d4EZqUm-r1itTr58nElITgzyzudVKLTnbzk9wn3nynkCFpj-ZTEDIbEvBVT3a4UiYEwxnqfGk_40NGtwaht9fUAQwO9bvckTdyDTmJ22JFCEyEbiWjUkn8ur8CcazrO5-X_rci1tWRBxMMpuEkKyDFJNJtb0pP1CByDd7hxzyB8wzfGnyHkfita7Xx58_okf86UuqfXXMb57JR96O84Fi0L1xjCuq5y1CPgp9KKB7-1Amxmp7cFGtQkeAsRXz-6xY0VrDB07zdl-RA4LYiLBKRfCTInsS3tB3MFndFkhnBB_u6mu2227QMrqba031f55VSo7UiyzO1WcChJ5R13N22hZpbzgnElZ2F76187qzlrSucNO0yv7E9VcJKFRAqyfYkxogmpxdR-F74DF9DFUTO1lZw-1BT1H9kFNSjIqTNkrwEQwiUA59VQa8JZQrjV9TWPXg'}
    Classifier: RandomForestClassifier has a training score of 77.4% accuracy score on Const Fill
    Classifier: RandomForestClassifier has a test score of 80.0% accuracy score on Const Fill
    Cookies: {'refresh_token': 'eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiJ9.eyJ0b2tlbl90eXBlIjoicmVmcmVzaCIsImV4cCI6MTY1NDU4NTA2NywianRpIjoiODdkOTg2OWUyODQ1NGFiYTlhMjNhMDdmODk4Mzc2MGIiLCJ1c2VyX2lkIjoiYXNkZkBnbWFpbC5jb20iLCJfZGF0YSI6eyJ1c2VyIjp7ImZpcnN0X25hbWUiOiIiLCJsYXN0X25hbWUiOiIiLCJ1c2VybmFtZSI6Im9uZXNwYW5hZG1pbiIsImVtYWlsIjoiYXNkZkBnbWFpbC5jb20iLCJvcmdhbml6YXRpb25fcGVybWlzc2lvbnMiOnsiYWRtaW4iOlt7Im5hbWUiOiJPbmVzcGFuIiwiaXNfYWN0aXZlIjp0cnVlfV0sImRhdGFfc2NpZW50aXN0IjpbeyJuYW1lIjoiT25lc3BhbiIsImlzX2FjdGl2ZSI6dHJ1ZX1dLCJhbmFseXN0IjpbeyJuYW1lIjoiT25lc3BhbiIsImlzX2FjdGl2ZSI6dHJ1ZX1dfX19fQ.XU1nRCeeuoqchVc8CPhsnDCCaRAc6v2Anhhag-pBgb1Vswd9rU_jmMmskq_dyaAlApeX_i5f22fIrul5bYOcfweIoC23sbYjEvvryMh_l4fkn6lKchQGG37gUMsBiXysVe0BOmaixIN1CYI3e0d-ZE_dKv7RICUvmUCBTSFSUv2oxkBsrnrxHgXeArcPLCNgq2qKGCCAm0vzlgTSClGZFLsCOdJs6JoLFmzg0tCETZqkjo1PXlI5_Z2fM7cgITGDhEHRFuHCdUiXcl3n1deT0pfMC7wh8l4KXDdILPe_9ibvivpz7p8d1i7Mvb1CfxSXUUMc-cIS0XkoV67L2Y32jxOT0ehOCHvnR0ouXnzK8C56SJELbXiUos6hyUuhoOYSSGHvoId5aXk1XjkJvW4-OmJWJ8H0S3SqcUMwUwl_kMnpOXb2e-k1cJff4-5PjBGbSk2u15c1WZ6NihUfhY8T3De7pvpIzaMPFLE2Vwg77mW4C-SVRM0HYaUzqPtdCLU5TAtDV_d1vx6sRniN2KCXxUgLHCAS7Htq7Bpw4XcXl9BVGWNH1kE1FbtQ7BaIL2uvC6IZkzXoSWne-Y-jDNIIcgGl62cz-m4sOtEEmmJ5Xx_kf_mLPWRUOSi5L5qSqH-f61nTVoHkhel3kE-BF5jthB51JMkaEGriKOleCgz17Yo', 'access_token': 'eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiJ9.eyJ0b2tlbl90eXBlIjoiYWNjZXNzIiwiZXhwIjoxNjU0NTEzMTAwLCJqdGkiOiJiN2MzYTAyMjJhMmI0MDNhOWM0M2E0NTUyMzUxZDJlMyIsInVzZXJfaWQiOiJhc2RmQGdtYWlsLmNvbSIsIl9kYXRhIjp7InVzZXIiOnsiZmlyc3RfbmFtZSI6IiIsImxhc3RfbmFtZSI6IiIsInVzZXJuYW1lIjoib25lc3BhbmFkbWluIiwiZW1haWwiOiJhc2RmQGdtYWlsLmNvbSIsIm9yZ2FuaXphdGlvbl9wZXJtaXNzaW9ucyI6eyJhZG1pbiI6W3sibmFtZSI6Ik9uZXNwYW4iLCJpc19hY3RpdmUiOnRydWV9XSwiZGF0YV9zY2llbnRpc3QiOlt7Im5hbWUiOiJPbmVzcGFuIiwiaXNfYWN0aXZlIjp0cnVlfV0sImFuYWx5c3QiOlt7Im5hbWUiOiJPbmVzcGFuIiwiaXNfYWN0aXZlIjp0cnVlfV19fX19.CVx4M48VebfJDi9c2UEFicVdSpjdU0-ztubgm0uCKT-aRmgRjo-XnUfoFaTwDP-tRe8l-H4ARfjec_fj3ZZU2UifsFM8WrgavEtWpZ61TkWRGMVO6_pG8AVAagtNgQnl66BlFW3klUtkiN0Q0T-WnvIbRfyrTVjQG4o4nyqhNDIy1Jw61jH4NkD60gALWYhq98DtL-TjicGz6-53Lz0jIIXaKwh9iEb7JfBOCtHycw9CyGTi6unxT8p95PCNZqNT0Mw8AbHaf0GIZZ7T2-QlSUzwbLHFU1_9DmZCE-yJsa_Rv6rB5EH8GkiE_gAAIh8fK4p1HN8VhbW3Ed-O7epGopvceslAzGrZEd8CEegCXn88vbbdtcbvsyoUIqKU6QGBtYpLJ9tHqwM0XLPD5jCQwFTVFZEd9iU7qTkcfutRgW3fc28k_Fmuuy__pTrtzMKJNnHU9MNre-a_99WrvRnN4sc-Fttn5BlZ54KdKufo08rGE7lv-WGhgv42LWPyt_YaUpAwRS7JV6KEQJB3GkB6C6qu7MgSwv7uwrl-GhQusbtzWWgL5BGeAuVypyz8l194ZNF8Ybfwgqe_Z7266BEQOefJ5thkLnAzl-NutJyfCYlF3_Lf1OfYLMK3h4WWXGWZYaRwvIqQNZ7p5buF0p0lTUYiceDtFV2WLxPrbZvQNfg'}
    Classifier: DecisionTreeClassifier has a training score of 78.60000000000001% accuracy score on Const Fill
    Classifier: DecisionTreeClassifier has a test score of 78.5% accuracy score on Const Fill
    Cookies: {'refresh_token': 'eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiJ9.eyJ0b2tlbl90eXBlIjoicmVmcmVzaCIsImV4cCI6MTY1NDU4NTA2NywianRpIjoiODdkOTg2OWUyODQ1NGFiYTlhMjNhMDdmODk4Mzc2MGIiLCJ1c2VyX2lkIjoiYXNkZkBnbWFpbC5jb20iLCJfZGF0YSI6eyJ1c2VyIjp7ImZpcnN0X25hbWUiOiIiLCJsYXN0X25hbWUiOiIiLCJ1c2VybmFtZSI6Im9uZXNwYW5hZG1pbiIsImVtYWlsIjoiYXNkZkBnbWFpbC5jb20iLCJvcmdhbml6YXRpb25fcGVybWlzc2lvbnMiOnsiYWRtaW4iOlt7Im5hbWUiOiJPbmVzcGFuIiwiaXNfYWN0aXZlIjp0cnVlfV0sImRhdGFfc2NpZW50aXN0IjpbeyJuYW1lIjoiT25lc3BhbiIsImlzX2FjdGl2ZSI6dHJ1ZX1dLCJhbmFseXN0IjpbeyJuYW1lIjoiT25lc3BhbiIsImlzX2FjdGl2ZSI6dHJ1ZX1dfX19fQ.XU1nRCeeuoqchVc8CPhsnDCCaRAc6v2Anhhag-pBgb1Vswd9rU_jmMmskq_dyaAlApeX_i5f22fIrul5bYOcfweIoC23sbYjEvvryMh_l4fkn6lKchQGG37gUMsBiXysVe0BOmaixIN1CYI3e0d-ZE_dKv7RICUvmUCBTSFSUv2oxkBsrnrxHgXeArcPLCNgq2qKGCCAm0vzlgTSClGZFLsCOdJs6JoLFmzg0tCETZqkjo1PXlI5_Z2fM7cgITGDhEHRFuHCdUiXcl3n1deT0pfMC7wh8l4KXDdILPe_9ibvivpz7p8d1i7Mvb1CfxSXUUMc-cIS0XkoV67L2Y32jxOT0ehOCHvnR0ouXnzK8C56SJELbXiUos6hyUuhoOYSSGHvoId5aXk1XjkJvW4-OmJWJ8H0S3SqcUMwUwl_kMnpOXb2e-k1cJff4-5PjBGbSk2u15c1WZ6NihUfhY8T3De7pvpIzaMPFLE2Vwg77mW4C-SVRM0HYaUzqPtdCLU5TAtDV_d1vx6sRniN2KCXxUgLHCAS7Htq7Bpw4XcXl9BVGWNH1kE1FbtQ7BaIL2uvC6IZkzXoSWne-Y-jDNIIcgGl62cz-m4sOtEEmmJ5Xx_kf_mLPWRUOSi5L5qSqH-f61nTVoHkhel3kE-BF5jthB51JMkaEGriKOleCgz17Yo', 'access_token': 'eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiJ9.eyJ0b2tlbl90eXBlIjoiYWNjZXNzIiwiZXhwIjoxNjU0NTEzMTAyLCJqdGkiOiJjNDYyZmM1YWQyNGQ0NDNhYTUwNGU1ZThmYjc4NzkzOSIsInVzZXJfaWQiOiJhc2RmQGdtYWlsLmNvbSIsIl9kYXRhIjp7InVzZXIiOnsiZmlyc3RfbmFtZSI6IiIsImxhc3RfbmFtZSI6IiIsInVzZXJuYW1lIjoib25lc3BhbmFkbWluIiwiZW1haWwiOiJhc2RmQGdtYWlsLmNvbSIsIm9yZ2FuaXphdGlvbl9wZXJtaXNzaW9ucyI6eyJhZG1pbiI6W3sibmFtZSI6Ik9uZXNwYW4iLCJpc19hY3RpdmUiOnRydWV9XSwiZGF0YV9zY2llbnRpc3QiOlt7Im5hbWUiOiJPbmVzcGFuIiwiaXNfYWN0aXZlIjp0cnVlfV0sImFuYWx5c3QiOlt7Im5hbWUiOiJPbmVzcGFuIiwiaXNfYWN0aXZlIjp0cnVlfV19fX19.L0n4iWCdm8uQ6Qa4zbb-F6wmxe3Hd1ELhdsgGmCDIVcLNb89SNB4-S6BJKpy0TmvVTccRHN9PKCK2QB1zH1Dm27RqjYF5VupYH_EaAqllv8qVD_kZepOXoDWpUOWQcGon03drUYJGQJV1poTHd-MUDws2kFdHO6TBIqGuDfxyD1VX1C15yWa5ucRrdsQkyHr3iqDpBEPGCzsNlDrR5NSUBr3Y5yBUqiNRB8TPA7k17133a1_IK9B5Zn5ophc78wUFx20Id1wDjSdalM03KaZU4poTIx0h1VdnsvoZNHpVIXpmBCdYpV63oWs4XRlg7jJKp45vbifqZ8VXPMAdTXQ1LyQsDx-Q5faGQGAPi0OsznspzQ5fNyD-0X9wYhn2C-T65g0qIID2mg009PGz-EqqJ93CEtd6j01xOoMPLlXCR7TP3F2OGCjr4v6KdwEGfayZhxFGpZx1NfpA86mBT5Zu2tBzfqObxsm1dUT54l9iTGW_7A7eOeOe2_YeVBmJQWYeH1LqWF_n03cCl1mO0VvfL9TDTjupnA6YZmF1DaG4Rs0tJvuey7MJm1nYKAnhrNGC-OmAV9PDXpgMVIEaifXTrO92QeVSP9IyBsupGu95Fe_JjICIaDDwvaWZUZOnQmlyE8kVI4FAhF224uaeN2E1-5whhzr92JI7qx-mB65gZw'}
    Classifier: GradientBoostingClassifier has a training score of 84.39999999999999% accuracy score on Const Fill
    Classifier: GradientBoostingClassifier has a test score of 80.0% accuracy score on Const Fill
    Cookies: {'refresh_token': 'eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiJ9.eyJ0b2tlbl90eXBlIjoicmVmcmVzaCIsImV4cCI6MTY1NDU4NTA2NywianRpIjoiODdkOTg2OWUyODQ1NGFiYTlhMjNhMDdmODk4Mzc2MGIiLCJ1c2VyX2lkIjoiYXNkZkBnbWFpbC5jb20iLCJfZGF0YSI6eyJ1c2VyIjp7ImZpcnN0X25hbWUiOiIiLCJsYXN0X25hbWUiOiIiLCJ1c2VybmFtZSI6Im9uZXNwYW5hZG1pbiIsImVtYWlsIjoiYXNkZkBnbWFpbC5jb20iLCJvcmdhbml6YXRpb25fcGVybWlzc2lvbnMiOnsiYWRtaW4iOlt7Im5hbWUiOiJPbmVzcGFuIiwiaXNfYWN0aXZlIjp0cnVlfV0sImRhdGFfc2NpZW50aXN0IjpbeyJuYW1lIjoiT25lc3BhbiIsImlzX2FjdGl2ZSI6dHJ1ZX1dLCJhbmFseXN0IjpbeyJuYW1lIjoiT25lc3BhbiIsImlzX2FjdGl2ZSI6dHJ1ZX1dfX19fQ.XU1nRCeeuoqchVc8CPhsnDCCaRAc6v2Anhhag-pBgb1Vswd9rU_jmMmskq_dyaAlApeX_i5f22fIrul5bYOcfweIoC23sbYjEvvryMh_l4fkn6lKchQGG37gUMsBiXysVe0BOmaixIN1CYI3e0d-ZE_dKv7RICUvmUCBTSFSUv2oxkBsrnrxHgXeArcPLCNgq2qKGCCAm0vzlgTSClGZFLsCOdJs6JoLFmzg0tCETZqkjo1PXlI5_Z2fM7cgITGDhEHRFuHCdUiXcl3n1deT0pfMC7wh8l4KXDdILPe_9ibvivpz7p8d1i7Mvb1CfxSXUUMc-cIS0XkoV67L2Y32jxOT0ehOCHvnR0ouXnzK8C56SJELbXiUos6hyUuhoOYSSGHvoId5aXk1XjkJvW4-OmJWJ8H0S3SqcUMwUwl_kMnpOXb2e-k1cJff4-5PjBGbSk2u15c1WZ6NihUfhY8T3De7pvpIzaMPFLE2Vwg77mW4C-SVRM0HYaUzqPtdCLU5TAtDV_d1vx6sRniN2KCXxUgLHCAS7Htq7Bpw4XcXl9BVGWNH1kE1FbtQ7BaIL2uvC6IZkzXoSWne-Y-jDNIIcgGl62cz-m4sOtEEmmJ5Xx_kf_mLPWRUOSi5L5qSqH-f61nTVoHkhel3kE-BF5jthB51JMkaEGriKOleCgz17Yo', 'access_token': 'eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiJ9.eyJ0b2tlbl90eXBlIjoiYWNjZXNzIiwiZXhwIjoxNjU0NTEzMTA0LCJqdGkiOiIwMzE2NTk0ZTQwNjk0ODI5YmI1YjY2YjI3NTVjMDMxZSIsInVzZXJfaWQiOiJhc2RmQGdtYWlsLmNvbSIsIl9kYXRhIjp7InVzZXIiOnsiZmlyc3RfbmFtZSI6IiIsImxhc3RfbmFtZSI6IiIsInVzZXJuYW1lIjoib25lc3BhbmFkbWluIiwiZW1haWwiOiJhc2RmQGdtYWlsLmNvbSIsIm9yZ2FuaXphdGlvbl9wZXJtaXNzaW9ucyI6eyJhZG1pbiI6W3sibmFtZSI6Ik9uZXNwYW4iLCJpc19hY3RpdmUiOnRydWV9XSwiZGF0YV9zY2llbnRpc3QiOlt7Im5hbWUiOiJPbmVzcGFuIiwiaXNfYWN0aXZlIjp0cnVlfV0sImFuYWx5c3QiOlt7Im5hbWUiOiJPbmVzcGFuIiwiaXNfYWN0aXZlIjp0cnVlfV19fX19.cVg2-44-0dGtYA8frfE-JJ3L7nbkXXawt7uSSZleRGlKrJJHKsozB00rgeMUwBhOn5XAJBwgl0haXnVPozqNWcKYig-v9rWfB9Z2Q4EBACDs8TiK5OKDOZt2QQxX-7ZR_FDumle6yYjqB8MsBVbtKg1JYTXdH1GkFVCzu2vzPqVnBYydQVsoxaagEtGHOdLkNy5FoscgARXjIQAFsbwXX6IxPyCjRFbebtpuqGFdTrg7hlDUKtNvK_o1B3CsqbRhncCTyCPm9koL3DxqMBD63NzeXhCmdFrLUVgGW-QkInP5ddYpHd-Ec_cnsNMZUGCR-mXNnFgCbY_WVCNlwy4_tXwB16PcYXdZRap4ykrlp-aVbG_-n1QB7bg_L-nsErpOVV0cjFFj-aRgY0aIIKZ0Q15BYLd6RKL2efWegEiR6AgTD1lsFrJLx2Zv_pubKRBhZ6UeKApd1ipX4XF0bMQ32io2EIWT-8oDWAcmwOfxMC5tE23cttjjOE-WlJaWkCywlmHJdqxgP5_LaGX-NsL07mtHh4Cbmt3z6nOg5yccKNAjuYiBb0gTlJKIJ9rcApMJTFwSq5jc2qY_pqZUAcMgKkMcDooZ61hnroGyJRHby57UaM02_iKeCmKbV8FcsXDTBM0dM_hYzK_VSXoZC3RQFhda7-tJZfxKro4yTFiwwGU'}
    Classifier: RandomForestClassifier has a training score of 77.2% accuracy score on Mode Fill
    Classifier: RandomForestClassifier has a test score of 80.0% accuracy score on Mode Fill
    Cookies: {'refresh_token': 'eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiJ9.eyJ0b2tlbl90eXBlIjoicmVmcmVzaCIsImV4cCI6MTY1NDU4NTA2NywianRpIjoiODdkOTg2OWUyODQ1NGFiYTlhMjNhMDdmODk4Mzc2MGIiLCJ1c2VyX2lkIjoiYXNkZkBnbWFpbC5jb20iLCJfZGF0YSI6eyJ1c2VyIjp7ImZpcnN0X25hbWUiOiIiLCJsYXN0X25hbWUiOiIiLCJ1c2VybmFtZSI6Im9uZXNwYW5hZG1pbiIsImVtYWlsIjoiYXNkZkBnbWFpbC5jb20iLCJvcmdhbml6YXRpb25fcGVybWlzc2lvbnMiOnsiYWRtaW4iOlt7Im5hbWUiOiJPbmVzcGFuIiwiaXNfYWN0aXZlIjp0cnVlfV0sImRhdGFfc2NpZW50aXN0IjpbeyJuYW1lIjoiT25lc3BhbiIsImlzX2FjdGl2ZSI6dHJ1ZX1dLCJhbmFseXN0IjpbeyJuYW1lIjoiT25lc3BhbiIsImlzX2FjdGl2ZSI6dHJ1ZX1dfX19fQ.XU1nRCeeuoqchVc8CPhsnDCCaRAc6v2Anhhag-pBgb1Vswd9rU_jmMmskq_dyaAlApeX_i5f22fIrul5bYOcfweIoC23sbYjEvvryMh_l4fkn6lKchQGG37gUMsBiXysVe0BOmaixIN1CYI3e0d-ZE_dKv7RICUvmUCBTSFSUv2oxkBsrnrxHgXeArcPLCNgq2qKGCCAm0vzlgTSClGZFLsCOdJs6JoLFmzg0tCETZqkjo1PXlI5_Z2fM7cgITGDhEHRFuHCdUiXcl3n1deT0pfMC7wh8l4KXDdILPe_9ibvivpz7p8d1i7Mvb1CfxSXUUMc-cIS0XkoV67L2Y32jxOT0ehOCHvnR0ouXnzK8C56SJELbXiUos6hyUuhoOYSSGHvoId5aXk1XjkJvW4-OmJWJ8H0S3SqcUMwUwl_kMnpOXb2e-k1cJff4-5PjBGbSk2u15c1WZ6NihUfhY8T3De7pvpIzaMPFLE2Vwg77mW4C-SVRM0HYaUzqPtdCLU5TAtDV_d1vx6sRniN2KCXxUgLHCAS7Htq7Bpw4XcXl9BVGWNH1kE1FbtQ7BaIL2uvC6IZkzXoSWne-Y-jDNIIcgGl62cz-m4sOtEEmmJ5Xx_kf_mLPWRUOSi5L5qSqH-f61nTVoHkhel3kE-BF5jthB51JMkaEGriKOleCgz17Yo', 'access_token': 'eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiJ9.eyJ0b2tlbl90eXBlIjoiYWNjZXNzIiwiZXhwIjoxNjU0NTEzMTA1LCJqdGkiOiJjNDQ4NjFmMzAyYTk0MGI2YmVhYzBlOTEzNzA4ZWM1NiIsInVzZXJfaWQiOiJhc2RmQGdtYWlsLmNvbSIsIl9kYXRhIjp7InVzZXIiOnsiZmlyc3RfbmFtZSI6IiIsImxhc3RfbmFtZSI6IiIsInVzZXJuYW1lIjoib25lc3BhbmFkbWluIiwiZW1haWwiOiJhc2RmQGdtYWlsLmNvbSIsIm9yZ2FuaXphdGlvbl9wZXJtaXNzaW9ucyI6eyJhZG1pbiI6W3sibmFtZSI6Ik9uZXNwYW4iLCJpc19hY3RpdmUiOnRydWV9XSwiZGF0YV9zY2llbnRpc3QiOlt7Im5hbWUiOiJPbmVzcGFuIiwiaXNfYWN0aXZlIjp0cnVlfV0sImFuYWx5c3QiOlt7Im5hbWUiOiJPbmVzcGFuIiwiaXNfYWN0aXZlIjp0cnVlfV19fX19.MUK77Youqo2kKOqIyd1va2-A2TMTxyRWaTy-0flwZowJ5dUlKHHQfKH-7VfV4eFWWcf-Qxwbpomw6bI_EVyv8Jdlw9M5WFfRPvdTknYxlcdxgDzqKHgZHoWyeSE6NA6bmDqMvGMSC6OTi2alir3HMMOG8SzaXfGDJiof5wJYhd4RnVLMG2uSo63hIurw_S4Z1cGvaDpz2o1iYkVUBkSdYvJt7rLgK9k836g2HrS-PYwSN300TT_2iXBRExahU-IIFzUOwti8o6NG5nhyQCIhnGac-wa1r6wp9DS9YvfpQuFtLyYMTumn_Huyp1eX-koJgfBouDA-pW2zVcfVqWzugi9y_4Wjrh7Cmo__daiOjb8ARaNYzt6-CRtC_wOI8fe5glxQyMqeZzXKosZHJlK_GFkbm4R9pasKC3TR-hQTSsRNm61dsRyHIIKEw3IotjZyThSVBC40jlG2Xab82ltxBuN71a0M6HL_olwpWngiRJRK0plUD2idDuMFaHO2CoFby9YdCd9RS0XxwNO6OGds-Rz3NerTIYHyAwVPGI2RZhzSyntncooOguePOIsVynMke916yxCEGlpjSjHm1pRgdtMC53GeWg9ErkB95UIWDbRAu_PAsgk_pHV6UettnaaGwd_zbpCR5ZOeN66IvZPKlu6uxgYJZ2erMaG0uFDUkUE'}
    Classifier: DecisionTreeClassifier has a training score of 79.60000000000001% accuracy score on Mode Fill
    Classifier: DecisionTreeClassifier has a test score of 78.5% accuracy score on Mode Fill
    Cookies: {'refresh_token': 'eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiJ9.eyJ0b2tlbl90eXBlIjoicmVmcmVzaCIsImV4cCI6MTY1NDU4NTA2NywianRpIjoiODdkOTg2OWUyODQ1NGFiYTlhMjNhMDdmODk4Mzc2MGIiLCJ1c2VyX2lkIjoiYXNkZkBnbWFpbC5jb20iLCJfZGF0YSI6eyJ1c2VyIjp7ImZpcnN0X25hbWUiOiIiLCJsYXN0X25hbWUiOiIiLCJ1c2VybmFtZSI6Im9uZXNwYW5hZG1pbiIsImVtYWlsIjoiYXNkZkBnbWFpbC5jb20iLCJvcmdhbml6YXRpb25fcGVybWlzc2lvbnMiOnsiYWRtaW4iOlt7Im5hbWUiOiJPbmVzcGFuIiwiaXNfYWN0aXZlIjp0cnVlfV0sImRhdGFfc2NpZW50aXN0IjpbeyJuYW1lIjoiT25lc3BhbiIsImlzX2FjdGl2ZSI6dHJ1ZX1dLCJhbmFseXN0IjpbeyJuYW1lIjoiT25lc3BhbiIsImlzX2FjdGl2ZSI6dHJ1ZX1dfX19fQ.XU1nRCeeuoqchVc8CPhsnDCCaRAc6v2Anhhag-pBgb1Vswd9rU_jmMmskq_dyaAlApeX_i5f22fIrul5bYOcfweIoC23sbYjEvvryMh_l4fkn6lKchQGG37gUMsBiXysVe0BOmaixIN1CYI3e0d-ZE_dKv7RICUvmUCBTSFSUv2oxkBsrnrxHgXeArcPLCNgq2qKGCCAm0vzlgTSClGZFLsCOdJs6JoLFmzg0tCETZqkjo1PXlI5_Z2fM7cgITGDhEHRFuHCdUiXcl3n1deT0pfMC7wh8l4KXDdILPe_9ibvivpz7p8d1i7Mvb1CfxSXUUMc-cIS0XkoV67L2Y32jxOT0ehOCHvnR0ouXnzK8C56SJELbXiUos6hyUuhoOYSSGHvoId5aXk1XjkJvW4-OmJWJ8H0S3SqcUMwUwl_kMnpOXb2e-k1cJff4-5PjBGbSk2u15c1WZ6NihUfhY8T3De7pvpIzaMPFLE2Vwg77mW4C-SVRM0HYaUzqPtdCLU5TAtDV_d1vx6sRniN2KCXxUgLHCAS7Htq7Bpw4XcXl9BVGWNH1kE1FbtQ7BaIL2uvC6IZkzXoSWne-Y-jDNIIcgGl62cz-m4sOtEEmmJ5Xx_kf_mLPWRUOSi5L5qSqH-f61nTVoHkhel3kE-BF5jthB51JMkaEGriKOleCgz17Yo', 'access_token': 'eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiJ9.eyJ0b2tlbl90eXBlIjoiYWNjZXNzIiwiZXhwIjoxNjU0NTEzMTA3LCJqdGkiOiJkZGM4NjhkN2MzOWI0YmMzODg4NzY0ZGVkNWYxZjQzMiIsInVzZXJfaWQiOiJhc2RmQGdtYWlsLmNvbSIsIl9kYXRhIjp7InVzZXIiOnsiZmlyc3RfbmFtZSI6IiIsImxhc3RfbmFtZSI6IiIsInVzZXJuYW1lIjoib25lc3BhbmFkbWluIiwiZW1haWwiOiJhc2RmQGdtYWlsLmNvbSIsIm9yZ2FuaXphdGlvbl9wZXJtaXNzaW9ucyI6eyJhZG1pbiI6W3sibmFtZSI6Ik9uZXNwYW4iLCJpc19hY3RpdmUiOnRydWV9XSwiZGF0YV9zY2llbnRpc3QiOlt7Im5hbWUiOiJPbmVzcGFuIiwiaXNfYWN0aXZlIjp0cnVlfV0sImFuYWx5c3QiOlt7Im5hbWUiOiJPbmVzcGFuIiwiaXNfYWN0aXZlIjp0cnVlfV19fX19.S2VDl2zEMGi-g47eyWpvppYbvZCmEC6FqcWCu9kCtvzgwYOT5HFfGBqCpj1ztTr_aOKaHkIaP89UJcoG4o4F73fk1eN8lRR_M336THyHTXAtTOXoETYAn17iW4Wvkaes_TsSlQ1yzM2doyIxH93tsn8BKXzv5EeBYaTKxDhdMALSfWqrNXZrJF9ffvOrDHETWTqu-ne1nKvF8gEnaIHjYy1qC9nGJuqmXde_kdTqKphfWbYQhWEiWPGDnM_HtJOG8NOyoB7KsjZkZhmL0pHFjEb9kqKYDGBtO9VVzIvfn-3kHiyWyMBixa612uOGtT1Kz4_qG2dNCQ2BYex6vuLk02dRUTlQjciHorTl8KkpwU-jHhANm9jicPRR1gRfvYc6sCCsas3_sWp3q4Gjeqhw6DklW5BBZLrH6MzeWEyU3L17YLNnhCSnGWCjU9i8RQhOh5XglYjD-lAG2hM9pwaBlv8R5J1ako-oND0hlfy-ZLHKp2E-y0OpqU-_po-v1JgiwGFYTutD1P1Nult2LiZjyH36B5YgsYl7i-gQamAW_oxz82KsMGFY1eG73OSDDXJydTJc9_l-iS76yV61cH-NmAxzMfOmwc5cMIBT2NBCAjClu2kCgzX28XP-ziZfXgD2jVienovL8OmZi2XuUrWogtoIWwpsbuMET6Yf8qkqNh0'}
    Classifier: GradientBoostingClassifier has a training score of 83.8% accuracy score on Mode Fill
    Classifier: GradientBoostingClassifier has a test score of 82.0% accuracy score on Mode Fill
    Cookies: {'refresh_token': 'eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiJ9.eyJ0b2tlbl90eXBlIjoicmVmcmVzaCIsImV4cCI6MTY1NDU4NTA2NywianRpIjoiODdkOTg2OWUyODQ1NGFiYTlhMjNhMDdmODk4Mzc2MGIiLCJ1c2VyX2lkIjoiYXNkZkBnbWFpbC5jb20iLCJfZGF0YSI6eyJ1c2VyIjp7ImZpcnN0X25hbWUiOiIiLCJsYXN0X25hbWUiOiIiLCJ1c2VybmFtZSI6Im9uZXNwYW5hZG1pbiIsImVtYWlsIjoiYXNkZkBnbWFpbC5jb20iLCJvcmdhbml6YXRpb25fcGVybWlzc2lvbnMiOnsiYWRtaW4iOlt7Im5hbWUiOiJPbmVzcGFuIiwiaXNfYWN0aXZlIjp0cnVlfV0sImRhdGFfc2NpZW50aXN0IjpbeyJuYW1lIjoiT25lc3BhbiIsImlzX2FjdGl2ZSI6dHJ1ZX1dLCJhbmFseXN0IjpbeyJuYW1lIjoiT25lc3BhbiIsImlzX2FjdGl2ZSI6dHJ1ZX1dfX19fQ.XU1nRCeeuoqchVc8CPhsnDCCaRAc6v2Anhhag-pBgb1Vswd9rU_jmMmskq_dyaAlApeX_i5f22fIrul5bYOcfweIoC23sbYjEvvryMh_l4fkn6lKchQGG37gUMsBiXysVe0BOmaixIN1CYI3e0d-ZE_dKv7RICUvmUCBTSFSUv2oxkBsrnrxHgXeArcPLCNgq2qKGCCAm0vzlgTSClGZFLsCOdJs6JoLFmzg0tCETZqkjo1PXlI5_Z2fM7cgITGDhEHRFuHCdUiXcl3n1deT0pfMC7wh8l4KXDdILPe_9ibvivpz7p8d1i7Mvb1CfxSXUUMc-cIS0XkoV67L2Y32jxOT0ehOCHvnR0ouXnzK8C56SJELbXiUos6hyUuhoOYSSGHvoId5aXk1XjkJvW4-OmJWJ8H0S3SqcUMwUwl_kMnpOXb2e-k1cJff4-5PjBGbSk2u15c1WZ6NihUfhY8T3De7pvpIzaMPFLE2Vwg77mW4C-SVRM0HYaUzqPtdCLU5TAtDV_d1vx6sRniN2KCXxUgLHCAS7Htq7Bpw4XcXl9BVGWNH1kE1FbtQ7BaIL2uvC6IZkzXoSWne-Y-jDNIIcgGl62cz-m4sOtEEmmJ5Xx_kf_mLPWRUOSi5L5qSqH-f61nTVoHkhel3kE-BF5jthB51JMkaEGriKOleCgz17Yo', 'access_token': 'eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiJ9.eyJ0b2tlbl90eXBlIjoiYWNjZXNzIiwiZXhwIjoxNjU0NTEzMTA5LCJqdGkiOiJhZTZkYmY1ZGIxNGM0ODNmYWE3OWJlNzgwOTkyYzQ1MyIsInVzZXJfaWQiOiJhc2RmQGdtYWlsLmNvbSIsIl9kYXRhIjp7InVzZXIiOnsiZmlyc3RfbmFtZSI6IiIsImxhc3RfbmFtZSI6IiIsInVzZXJuYW1lIjoib25lc3BhbmFkbWluIiwiZW1haWwiOiJhc2RmQGdtYWlsLmNvbSIsIm9yZ2FuaXphdGlvbl9wZXJtaXNzaW9ucyI6eyJhZG1pbiI6W3sibmFtZSI6Ik9uZXNwYW4iLCJpc19hY3RpdmUiOnRydWV9XSwiZGF0YV9zY2llbnRpc3QiOlt7Im5hbWUiOiJPbmVzcGFuIiwiaXNfYWN0aXZlIjp0cnVlfV0sImFuYWx5c3QiOlt7Im5hbWUiOiJPbmVzcGFuIiwiaXNfYWN0aXZlIjp0cnVlfV19fX19.n96Ms9_dt6Kp9KFtQinpdqQgc2obewYlg931iUwWkfSzyYb-Wmv7-y1GHDmrrwNbMfq_6mN1zntzWUoNkQriJXUHzLm7nk38K-ROn0fYh2mdkfeRM1P4Nqzylec_lh4Vxx_8kXTJeBAavOlJPJHmyAjvdn0oiuA-Vljm__tdEAC1CidGHC-LrXZFs-xMJKvPuIiusLnDoHXFNC75BMjgDnYyJ8-0DlwjYXzylLi5IZXB0dA4sgIPsXa6xrKdtvW1Pfcq5JIKkTiq2cQmIs5ANRIeWSV41-R7NyKesbUoNL9yarIFrJcC9ipZSMM37bOWYU5floe-rsI-phFlaVGv1DPOA6vGOZ4CBIcmSIkrAmEyO7FbUY5YSw8wq4D5vGGbTiplKYLfRnwT27E6mPdn6dZCk_HqFwXDouMFbyIC763Y4d6Y3o55nixXpOYRxU7Fg_iKed0sYGttkmu3EyU_1xaS_FYWcN5FHgc3xWHck53OGjY3PMfjVOcuqfDF_2Hs4suGgGfc0kU-QzNkbnlZ2os-b6JNEe08qtJpFuCs7FXzHePN_tVz90JbGPNEzBNxeEljEVTSQvjaDVAWevB5CjGefhdcyXTBEEuv7w7L-_GGKc43sc8pMZZ2tRxz5A3uDXPwG75ALFcA0ugL77LQ-sSXAS9gI2mBOjeMp1SxKc4'}
    Classifier: RandomForestClassifier has a training score of 85.8% accuracy score on Const Fill Smote
    Classifier: RandomForestClassifier has a test score of 80.0% accuracy score on Const Fill Smote
    Cookies: {'refresh_token': 'eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiJ9.eyJ0b2tlbl90eXBlIjoicmVmcmVzaCIsImV4cCI6MTY1NDU4NTA2NywianRpIjoiODdkOTg2OWUyODQ1NGFiYTlhMjNhMDdmODk4Mzc2MGIiLCJ1c2VyX2lkIjoiYXNkZkBnbWFpbC5jb20iLCJfZGF0YSI6eyJ1c2VyIjp7ImZpcnN0X25hbWUiOiIiLCJsYXN0X25hbWUiOiIiLCJ1c2VybmFtZSI6Im9uZXNwYW5hZG1pbiIsImVtYWlsIjoiYXNkZkBnbWFpbC5jb20iLCJvcmdhbml6YXRpb25fcGVybWlzc2lvbnMiOnsiYWRtaW4iOlt7Im5hbWUiOiJPbmVzcGFuIiwiaXNfYWN0aXZlIjp0cnVlfV0sImRhdGFfc2NpZW50aXN0IjpbeyJuYW1lIjoiT25lc3BhbiIsImlzX2FjdGl2ZSI6dHJ1ZX1dLCJhbmFseXN0IjpbeyJuYW1lIjoiT25lc3BhbiIsImlzX2FjdGl2ZSI6dHJ1ZX1dfX19fQ.XU1nRCeeuoqchVc8CPhsnDCCaRAc6v2Anhhag-pBgb1Vswd9rU_jmMmskq_dyaAlApeX_i5f22fIrul5bYOcfweIoC23sbYjEvvryMh_l4fkn6lKchQGG37gUMsBiXysVe0BOmaixIN1CYI3e0d-ZE_dKv7RICUvmUCBTSFSUv2oxkBsrnrxHgXeArcPLCNgq2qKGCCAm0vzlgTSClGZFLsCOdJs6JoLFmzg0tCETZqkjo1PXlI5_Z2fM7cgITGDhEHRFuHCdUiXcl3n1deT0pfMC7wh8l4KXDdILPe_9ibvivpz7p8d1i7Mvb1CfxSXUUMc-cIS0XkoV67L2Y32jxOT0ehOCHvnR0ouXnzK8C56SJELbXiUos6hyUuhoOYSSGHvoId5aXk1XjkJvW4-OmJWJ8H0S3SqcUMwUwl_kMnpOXb2e-k1cJff4-5PjBGbSk2u15c1WZ6NihUfhY8T3De7pvpIzaMPFLE2Vwg77mW4C-SVRM0HYaUzqPtdCLU5TAtDV_d1vx6sRniN2KCXxUgLHCAS7Htq7Bpw4XcXl9BVGWNH1kE1FbtQ7BaIL2uvC6IZkzXoSWne-Y-jDNIIcgGl62cz-m4sOtEEmmJ5Xx_kf_mLPWRUOSi5L5qSqH-f61nTVoHkhel3kE-BF5jthB51JMkaEGriKOleCgz17Yo', 'access_token': 'eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiJ9.eyJ0b2tlbl90eXBlIjoiYWNjZXNzIiwiZXhwIjoxNjU0NTEzMTEwLCJqdGkiOiI4M2Q3YmI4MDgyYzA0ZTA4YTRiNTBhOGExNGQ0MjZjMiIsInVzZXJfaWQiOiJhc2RmQGdtYWlsLmNvbSIsIl9kYXRhIjp7InVzZXIiOnsiZmlyc3RfbmFtZSI6IiIsImxhc3RfbmFtZSI6IiIsInVzZXJuYW1lIjoib25lc3BhbmFkbWluIiwiZW1haWwiOiJhc2RmQGdtYWlsLmNvbSIsIm9yZ2FuaXphdGlvbl9wZXJtaXNzaW9ucyI6eyJhZG1pbiI6W3sibmFtZSI6Ik9uZXNwYW4iLCJpc19hY3RpdmUiOnRydWV9XSwiZGF0YV9zY2llbnRpc3QiOlt7Im5hbWUiOiJPbmVzcGFuIiwiaXNfYWN0aXZlIjp0cnVlfV0sImFuYWx5c3QiOlt7Im5hbWUiOiJPbmVzcGFuIiwiaXNfYWN0aXZlIjp0cnVlfV19fX19.Ihnx9nNwWs0rKZbdN2Y7ybPF9Vg1r9FIWjorYXwhZI4ZJ6QvucSFBDD8YcroFBLH3uoGBqDEeJ2sxeNCjfBvvlbSkBoMtdYISYaNtihHG3B_uSZZ2V2bS20JOPTVNiHTGAF-IVqXsVPTqwklv-8MudXdXL6oygrZ4jftFtj5bKnq51pO5KwJU3jhf9XipnOYmDSXr0xjAtVnoPEYLiGRsqnUpOwhquxoPr3Rj9Mmy9_hOkFyBNqxglLLeM-wqm5v1kNjiXTj9s2CRRyjOu6IEK60aWluVe4KRaTE-v_9BKcZhCwNfblaffvRidimEkJlTMai1AwL6AytdEQfG1Yh8tMTK2QF8digBJAIJbZ6E20ZLhhcjtu-7sNrPT-HTUOKkYSWRLEyPScQFE_kiram2kD-Swn40Qc67u7_nOhTllDR0yypwaN1FEebBH22RefetS7OZxCyeGRKeBTtKTQbJGBdXX6mqZU1dpNc3Vh1guBUEErqRvoUJ6EC_qear15VB-2bXRT3KWn4TwlyqUsSsMb6QTUAEXzyVJpuJfKi7G9_VLVIsIBfZgapkrw4gW4WbMc_svwF4hjEg1MPtwr4RAGomELz6JrbA-Uve7yJ0RoT9pASBn8XoFYNBLNRk5mnpeBc64DRIh2jskeLTT_37FhLIvgl90c4MfiMk4oPO5w'}
    Classifier: DecisionTreeClassifier has a training score of 80.80000000000001% accuracy score on Const Fill Smote
    Classifier: DecisionTreeClassifier has a test score of 74.0% accuracy score on Const Fill Smote
    Cookies: {'refresh_token': 'eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiJ9.eyJ0b2tlbl90eXBlIjoicmVmcmVzaCIsImV4cCI6MTY1NDU4NTA2NywianRpIjoiODdkOTg2OWUyODQ1NGFiYTlhMjNhMDdmODk4Mzc2MGIiLCJ1c2VyX2lkIjoiYXNkZkBnbWFpbC5jb20iLCJfZGF0YSI6eyJ1c2VyIjp7ImZpcnN0X25hbWUiOiIiLCJsYXN0X25hbWUiOiIiLCJ1c2VybmFtZSI6Im9uZXNwYW5hZG1pbiIsImVtYWlsIjoiYXNkZkBnbWFpbC5jb20iLCJvcmdhbml6YXRpb25fcGVybWlzc2lvbnMiOnsiYWRtaW4iOlt7Im5hbWUiOiJPbmVzcGFuIiwiaXNfYWN0aXZlIjp0cnVlfV0sImRhdGFfc2NpZW50aXN0IjpbeyJuYW1lIjoiT25lc3BhbiIsImlzX2FjdGl2ZSI6dHJ1ZX1dLCJhbmFseXN0IjpbeyJuYW1lIjoiT25lc3BhbiIsImlzX2FjdGl2ZSI6dHJ1ZX1dfX19fQ.XU1nRCeeuoqchVc8CPhsnDCCaRAc6v2Anhhag-pBgb1Vswd9rU_jmMmskq_dyaAlApeX_i5f22fIrul5bYOcfweIoC23sbYjEvvryMh_l4fkn6lKchQGG37gUMsBiXysVe0BOmaixIN1CYI3e0d-ZE_dKv7RICUvmUCBTSFSUv2oxkBsrnrxHgXeArcPLCNgq2qKGCCAm0vzlgTSClGZFLsCOdJs6JoLFmzg0tCETZqkjo1PXlI5_Z2fM7cgITGDhEHRFuHCdUiXcl3n1deT0pfMC7wh8l4KXDdILPe_9ibvivpz7p8d1i7Mvb1CfxSXUUMc-cIS0XkoV67L2Y32jxOT0ehOCHvnR0ouXnzK8C56SJELbXiUos6hyUuhoOYSSGHvoId5aXk1XjkJvW4-OmJWJ8H0S3SqcUMwUwl_kMnpOXb2e-k1cJff4-5PjBGbSk2u15c1WZ6NihUfhY8T3De7pvpIzaMPFLE2Vwg77mW4C-SVRM0HYaUzqPtdCLU5TAtDV_d1vx6sRniN2KCXxUgLHCAS7Htq7Bpw4XcXl9BVGWNH1kE1FbtQ7BaIL2uvC6IZkzXoSWne-Y-jDNIIcgGl62cz-m4sOtEEmmJ5Xx_kf_mLPWRUOSi5L5qSqH-f61nTVoHkhel3kE-BF5jthB51JMkaEGriKOleCgz17Yo', 'access_token': 'eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiJ9.eyJ0b2tlbl90eXBlIjoiYWNjZXNzIiwiZXhwIjoxNjU0NTEzMTE0LCJqdGkiOiI3NTY0OTM5ODBlMGQ0YzU1OTJhMDlhNzMwOTQwOTk2OSIsInVzZXJfaWQiOiJhc2RmQGdtYWlsLmNvbSIsIl9kYXRhIjp7InVzZXIiOnsiZmlyc3RfbmFtZSI6IiIsImxhc3RfbmFtZSI6IiIsInVzZXJuYW1lIjoib25lc3BhbmFkbWluIiwiZW1haWwiOiJhc2RmQGdtYWlsLmNvbSIsIm9yZ2FuaXphdGlvbl9wZXJtaXNzaW9ucyI6eyJhZG1pbiI6W3sibmFtZSI6Ik9uZXNwYW4iLCJpc19hY3RpdmUiOnRydWV9XSwiZGF0YV9zY2llbnRpc3QiOlt7Im5hbWUiOiJPbmVzcGFuIiwiaXNfYWN0aXZlIjp0cnVlfV0sImFuYWx5c3QiOlt7Im5hbWUiOiJPbmVzcGFuIiwiaXNfYWN0aXZlIjp0cnVlfV19fX19.JlDjUkbQF6f9W9rjnXBTXmsbXNDno4iT0F6G7TD1HSPKXEm4MUKkX9U8hF4rF--5_dkB15hPzH8mX-oRxKAEr6rXEKX_n7YW9eoTh-N8D6IcNnfAalCe3w7uKqpuukXsRKXqoIDYH7d3T15SQP7v37YVskpxbqqrf5W27_Zyrx-yH1bVWvp7UpT2nvE_TZehAT27wDkX97SVE8D3wMGbkolfBDFJwt2SZVViZUkCXO7UgKUfdKQVLoxz5IWVVUjMgGOkXgP5HYsTupQQkXqG9U5K5VQ13fZrr6FkzH3v3ZQCwoRdGFHz-AL6lKEGhAet2D4xk7KQltuTMKbYdUBZW8XDb2_2UCGcgbhPW38grkspfQHWONqZjyXxkmffXjpKUeFQLdtnoPPFmvLTplJHdQISHGQbrp4m2hXTT2-bjXSkUU66s5iZV-_YyBCqTZJEpB17Z9Lh_bEUkU3WU-SX5bBhIZDbaDQFg43wTm2KpnNpiwgJcToRhQWdV1Hf9_hOEjX72auxGQ69ZgD-yWWP9zBp7OkpBfMLQArH_nOC8ec4N2qPMbsnJjIVzjD_X8ZuPSnXP6lQsFlb2hNW5Ga5mMpNiYPwrAVe3SSsf0B4qJrqgLYX7Kx_JKw_FCx23dcwrUhZvEU3TRqGDgdaOWiwu4ZgE5XTFsjLmhecMWJSfak'}
    Classifier: GradientBoostingClassifier has a training score of 87.4% accuracy score on Const Fill Smote
    Classifier: GradientBoostingClassifier has a test score of 80.0% accuracy score on Const Fill Smote
    Cookies: {'refresh_token': 'eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiJ9.eyJ0b2tlbl90eXBlIjoicmVmcmVzaCIsImV4cCI6MTY1NDU4NTA2NywianRpIjoiODdkOTg2OWUyODQ1NGFiYTlhMjNhMDdmODk4Mzc2MGIiLCJ1c2VyX2lkIjoiYXNkZkBnbWFpbC5jb20iLCJfZGF0YSI6eyJ1c2VyIjp7ImZpcnN0X25hbWUiOiIiLCJsYXN0X25hbWUiOiIiLCJ1c2VybmFtZSI6Im9uZXNwYW5hZG1pbiIsImVtYWlsIjoiYXNkZkBnbWFpbC5jb20iLCJvcmdhbml6YXRpb25fcGVybWlzc2lvbnMiOnsiYWRtaW4iOlt7Im5hbWUiOiJPbmVzcGFuIiwiaXNfYWN0aXZlIjp0cnVlfV0sImRhdGFfc2NpZW50aXN0IjpbeyJuYW1lIjoiT25lc3BhbiIsImlzX2FjdGl2ZSI6dHJ1ZX1dLCJhbmFseXN0IjpbeyJuYW1lIjoiT25lc3BhbiIsImlzX2FjdGl2ZSI6dHJ1ZX1dfX19fQ.XU1nRCeeuoqchVc8CPhsnDCCaRAc6v2Anhhag-pBgb1Vswd9rU_jmMmskq_dyaAlApeX_i5f22fIrul5bYOcfweIoC23sbYjEvvryMh_l4fkn6lKchQGG37gUMsBiXysVe0BOmaixIN1CYI3e0d-ZE_dKv7RICUvmUCBTSFSUv2oxkBsrnrxHgXeArcPLCNgq2qKGCCAm0vzlgTSClGZFLsCOdJs6JoLFmzg0tCETZqkjo1PXlI5_Z2fM7cgITGDhEHRFuHCdUiXcl3n1deT0pfMC7wh8l4KXDdILPe_9ibvivpz7p8d1i7Mvb1CfxSXUUMc-cIS0XkoV67L2Y32jxOT0ehOCHvnR0ouXnzK8C56SJELbXiUos6hyUuhoOYSSGHvoId5aXk1XjkJvW4-OmJWJ8H0S3SqcUMwUwl_kMnpOXb2e-k1cJff4-5PjBGbSk2u15c1WZ6NihUfhY8T3De7pvpIzaMPFLE2Vwg77mW4C-SVRM0HYaUzqPtdCLU5TAtDV_d1vx6sRniN2KCXxUgLHCAS7Htq7Bpw4XcXl9BVGWNH1kE1FbtQ7BaIL2uvC6IZkzXoSWne-Y-jDNIIcgGl62cz-m4sOtEEmmJ5Xx_kf_mLPWRUOSi5L5qSqH-f61nTVoHkhel3kE-BF5jthB51JMkaEGriKOleCgz17Yo', 'access_token': 'eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiJ9.eyJ0b2tlbl90eXBlIjoiYWNjZXNzIiwiZXhwIjoxNjU0NTEzMTE2LCJqdGkiOiI5ZjRmODRkMzVhNTc0NDE5OGVkNzA4ZTRhMGYzNTkwNiIsInVzZXJfaWQiOiJhc2RmQGdtYWlsLmNvbSIsIl9kYXRhIjp7InVzZXIiOnsiZmlyc3RfbmFtZSI6IiIsImxhc3RfbmFtZSI6IiIsInVzZXJuYW1lIjoib25lc3BhbmFkbWluIiwiZW1haWwiOiJhc2RmQGdtYWlsLmNvbSIsIm9yZ2FuaXphdGlvbl9wZXJtaXNzaW9ucyI6eyJhZG1pbiI6W3sibmFtZSI6Ik9uZXNwYW4iLCJpc19hY3RpdmUiOnRydWV9XSwiZGF0YV9zY2llbnRpc3QiOlt7Im5hbWUiOiJPbmVzcGFuIiwiaXNfYWN0aXZlIjp0cnVlfV0sImFuYWx5c3QiOlt7Im5hbWUiOiJPbmVzcGFuIiwiaXNfYWN0aXZlIjp0cnVlfV19fX19.pg_RU3Hs7GOi3lz1CA3f2QEsOnr4RaOler_mtvxpy5rCNjZf7XHbe_xCbBvlqzGP5p_mZjn2TOuIaQ7YMc8gHeH8hYl-_ZjgLiVfv9Ob77lIwZ-NDX6dZtq26VetL5sUJ-iDEWOpb1XuYP4YD3isJNAx1wKmGRkyrsVS1i2wiWondxy_jsZew_gVsuGMwg59YWnE2W96WFlJ8EJot4ZS30Q5D2LAbnGienPdOvav2Pjda3S4DJgZ8bvMq-rjx6bZ-uiIJCx7eAyrOhD_ncxcvB5r212bWBlPHAWYk_Sr7hRqPHcNjgcO7UHfKYkv6yUJOSJNWLq-oUUke2FVHA59EhhKIKmkCa6Q84NGtBF80VwtLJ67Lg686YMdVdugSVs_Q4fTEu9d2RkKEBJNFy_Jf0gJa3qYiehXaNn1q4NK1ni-ZL2v32IjOadkjn8HdMn4Eyij8RL4ltBTj7Oe74FgGNC5JCZ-1oSyMCL57i05Fmx9FDNgEZl6HKXuW-zI_JbwxQLa-C1mWae4U32mjo4D_CsW5RAZpvkoqZGF1UWiJqetQ1SbfvVRktUpJGqK-l5mHcmTOQhiMAG4-_ir3VutzigqhDDY-tPQX7rItuQsgnARqyh7fjsNDOHHlGFdiA8BSyunLWXfZlReT0CzxTDS3QxYVssNQBm9QoyB0v1O1jA'}
    Classifier: RandomForestClassifier has a training score of 86.1% accuracy score on Mode Fill Smote
    Classifier: RandomForestClassifier has a test score of 81.5% accuracy score on Mode Fill Smote
    Cookies: {'refresh_token': 'eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiJ9.eyJ0b2tlbl90eXBlIjoicmVmcmVzaCIsImV4cCI6MTY1NDU4NTA2NywianRpIjoiODdkOTg2OWUyODQ1NGFiYTlhMjNhMDdmODk4Mzc2MGIiLCJ1c2VyX2lkIjoiYXNkZkBnbWFpbC5jb20iLCJfZGF0YSI6eyJ1c2VyIjp7ImZpcnN0X25hbWUiOiIiLCJsYXN0X25hbWUiOiIiLCJ1c2VybmFtZSI6Im9uZXNwYW5hZG1pbiIsImVtYWlsIjoiYXNkZkBnbWFpbC5jb20iLCJvcmdhbml6YXRpb25fcGVybWlzc2lvbnMiOnsiYWRtaW4iOlt7Im5hbWUiOiJPbmVzcGFuIiwiaXNfYWN0aXZlIjp0cnVlfV0sImRhdGFfc2NpZW50aXN0IjpbeyJuYW1lIjoiT25lc3BhbiIsImlzX2FjdGl2ZSI6dHJ1ZX1dLCJhbmFseXN0IjpbeyJuYW1lIjoiT25lc3BhbiIsImlzX2FjdGl2ZSI6dHJ1ZX1dfX19fQ.XU1nRCeeuoqchVc8CPhsnDCCaRAc6v2Anhhag-pBgb1Vswd9rU_jmMmskq_dyaAlApeX_i5f22fIrul5bYOcfweIoC23sbYjEvvryMh_l4fkn6lKchQGG37gUMsBiXysVe0BOmaixIN1CYI3e0d-ZE_dKv7RICUvmUCBTSFSUv2oxkBsrnrxHgXeArcPLCNgq2qKGCCAm0vzlgTSClGZFLsCOdJs6JoLFmzg0tCETZqkjo1PXlI5_Z2fM7cgITGDhEHRFuHCdUiXcl3n1deT0pfMC7wh8l4KXDdILPe_9ibvivpz7p8d1i7Mvb1CfxSXUUMc-cIS0XkoV67L2Y32jxOT0ehOCHvnR0ouXnzK8C56SJELbXiUos6hyUuhoOYSSGHvoId5aXk1XjkJvW4-OmJWJ8H0S3SqcUMwUwl_kMnpOXb2e-k1cJff4-5PjBGbSk2u15c1WZ6NihUfhY8T3De7pvpIzaMPFLE2Vwg77mW4C-SVRM0HYaUzqPtdCLU5TAtDV_d1vx6sRniN2KCXxUgLHCAS7Htq7Bpw4XcXl9BVGWNH1kE1FbtQ7BaIL2uvC6IZkzXoSWne-Y-jDNIIcgGl62cz-m4sOtEEmmJ5Xx_kf_mLPWRUOSi5L5qSqH-f61nTVoHkhel3kE-BF5jthB51JMkaEGriKOleCgz17Yo', 'access_token': 'eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiJ9.eyJ0b2tlbl90eXBlIjoiYWNjZXNzIiwiZXhwIjoxNjU0NTEzMTE2LCJqdGkiOiJkYmMxN2Y1ZjgwZjU0YTI5OGIwMTM0YWQxYWFhMGJjMiIsInVzZXJfaWQiOiJhc2RmQGdtYWlsLmNvbSIsIl9kYXRhIjp7InVzZXIiOnsiZmlyc3RfbmFtZSI6IiIsImxhc3RfbmFtZSI6IiIsInVzZXJuYW1lIjoib25lc3BhbmFkbWluIiwiZW1haWwiOiJhc2RmQGdtYWlsLmNvbSIsIm9yZ2FuaXphdGlvbl9wZXJtaXNzaW9ucyI6eyJhZG1pbiI6W3sibmFtZSI6Ik9uZXNwYW4iLCJpc19hY3RpdmUiOnRydWV9XSwiZGF0YV9zY2llbnRpc3QiOlt7Im5hbWUiOiJPbmVzcGFuIiwiaXNfYWN0aXZlIjp0cnVlfV0sImFuYWx5c3QiOlt7Im5hbWUiOiJPbmVzcGFuIiwiaXNfYWN0aXZlIjp0cnVlfV19fX19.q7NNnTBjNV9cHEWmAQsDMVnIVRRZKrJRh3X3N241y_B4SHcV2mq03NeGinRdrCestaBKxUifxp__XLkS54hcDYUJAD_EjHa3A8812PMQmhsEQdJvLTv-T6FWrFtR7OQQqAXeWlYkvHq0bRGVPIxh5OVL2hmMpXzuSmjA2xnbA1HC1yeEeH5liE3vcV9JT0l1RTFi6hR8t_sOYW8PXVVA8VlGTmKu_ztww2n-am5BbSG_9mL4V8Z-TvX3hrxoXS8bVsPZz5iE5yE92GVL9OAHUKcfcyQvuJlg2vl4JYXlwrcFYBirYR7our5h9FZRdsyNQrwxCI9FzgIUFmExcEbEMRps8uY7lxxhR0hhLuh9MlQApGKe-E5yyJ1eHP07xpN-lO7WCLcxfuxVOu11Jr1A6jLAPTtcnNvSvRz24Mj05j7KmtplJ9awm9SqoA6AB5P9fz0uNW8nY-4XelwW7xTC6IedhaJ312f7zuaBIHHidDifoFGCW8vZi124MdgriRCN7No3gXZ3w9taKPZxKTa5Jk_-0HMMtxaJN47e2RcOp4MkpNGV_seZI_6rJ0ngQa39soEDwJHRbCWD5-ef57ESo6o06xrvdtjlDS5wg-GG_xVkEI1rGBNA0x5WAWN_IknTxWSB_CAF-I1LTauAkLh8i0cZT0eRqsxH8W5WXFgPCfE'}
    Classifier: DecisionTreeClassifier has a training score of 79.9% accuracy score on Mode Fill Smote
    Classifier: DecisionTreeClassifier has a test score of 74.5% accuracy score on Mode Fill Smote
    Cookies: {'refresh_token': 'eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiJ9.eyJ0b2tlbl90eXBlIjoicmVmcmVzaCIsImV4cCI6MTY1NDU4NTA2NywianRpIjoiODdkOTg2OWUyODQ1NGFiYTlhMjNhMDdmODk4Mzc2MGIiLCJ1c2VyX2lkIjoiYXNkZkBnbWFpbC5jb20iLCJfZGF0YSI6eyJ1c2VyIjp7ImZpcnN0X25hbWUiOiIiLCJsYXN0X25hbWUiOiIiLCJ1c2VybmFtZSI6Im9uZXNwYW5hZG1pbiIsImVtYWlsIjoiYXNkZkBnbWFpbC5jb20iLCJvcmdhbml6YXRpb25fcGVybWlzc2lvbnMiOnsiYWRtaW4iOlt7Im5hbWUiOiJPbmVzcGFuIiwiaXNfYWN0aXZlIjp0cnVlfV0sImRhdGFfc2NpZW50aXN0IjpbeyJuYW1lIjoiT25lc3BhbiIsImlzX2FjdGl2ZSI6dHJ1ZX1dLCJhbmFseXN0IjpbeyJuYW1lIjoiT25lc3BhbiIsImlzX2FjdGl2ZSI6dHJ1ZX1dfX19fQ.XU1nRCeeuoqchVc8CPhsnDCCaRAc6v2Anhhag-pBgb1Vswd9rU_jmMmskq_dyaAlApeX_i5f22fIrul5bYOcfweIoC23sbYjEvvryMh_l4fkn6lKchQGG37gUMsBiXysVe0BOmaixIN1CYI3e0d-ZE_dKv7RICUvmUCBTSFSUv2oxkBsrnrxHgXeArcPLCNgq2qKGCCAm0vzlgTSClGZFLsCOdJs6JoLFmzg0tCETZqkjo1PXlI5_Z2fM7cgITGDhEHRFuHCdUiXcl3n1deT0pfMC7wh8l4KXDdILPe_9ibvivpz7p8d1i7Mvb1CfxSXUUMc-cIS0XkoV67L2Y32jxOT0ehOCHvnR0ouXnzK8C56SJELbXiUos6hyUuhoOYSSGHvoId5aXk1XjkJvW4-OmJWJ8H0S3SqcUMwUwl_kMnpOXb2e-k1cJff4-5PjBGbSk2u15c1WZ6NihUfhY8T3De7pvpIzaMPFLE2Vwg77mW4C-SVRM0HYaUzqPtdCLU5TAtDV_d1vx6sRniN2KCXxUgLHCAS7Htq7Bpw4XcXl9BVGWNH1kE1FbtQ7BaIL2uvC6IZkzXoSWne-Y-jDNIIcgGl62cz-m4sOtEEmmJ5Xx_kf_mLPWRUOSi5L5qSqH-f61nTVoHkhel3kE-BF5jthB51JMkaEGriKOleCgz17Yo', 'access_token': 'eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiJ9.eyJ0b2tlbl90eXBlIjoiYWNjZXNzIiwiZXhwIjoxNjU0NTEzMTIxLCJqdGkiOiIwYjc0NjE4YmMwMTE0Y2Q5ODM3MDI5MmUzNWJjZmNhZCIsInVzZXJfaWQiOiJhc2RmQGdtYWlsLmNvbSIsIl9kYXRhIjp7InVzZXIiOnsiZmlyc3RfbmFtZSI6IiIsImxhc3RfbmFtZSI6IiIsInVzZXJuYW1lIjoib25lc3BhbmFkbWluIiwiZW1haWwiOiJhc2RmQGdtYWlsLmNvbSIsIm9yZ2FuaXphdGlvbl9wZXJtaXNzaW9ucyI6eyJhZG1pbiI6W3sibmFtZSI6Ik9uZXNwYW4iLCJpc19hY3RpdmUiOnRydWV9XSwiZGF0YV9zY2llbnRpc3QiOlt7Im5hbWUiOiJPbmVzcGFuIiwiaXNfYWN0aXZlIjp0cnVlfV0sImFuYWx5c3QiOlt7Im5hbWUiOiJPbmVzcGFuIiwiaXNfYWN0aXZlIjp0cnVlfV19fX19.gikd5T_CVcJhQ-6sx10fYrXmFILk6VZZOMp0OITzG0RxdtQTHLTTXvEMEp1bNxnlvTIwB7hgcgAb0nHcYck5IKHUOuewgCBjbjp4BN3E4ZyadDLSY0L2_01jka-tQcstLb6RCvsvpLe2JvGPDrc-mPQLO4PFL8EIF9pbJgEkIODHQZJB-ubb-zChpqp8X9-X-YZmCRx3WC9x9eLyjr_tjtOMWK5BSUZnyFIOAcwYnpvelCygm5sj6NtwI34fmfcKoP7wXQENdcU87lntTnCN9ZpJ1NGegC9NTwF5Q6FuA-vu5rygXCjJ6aeVr9n5dDnRh-Xd4trEEHnRlUVhwS3i1jEAvYDfPJOna_w_yaCTSuTH6l9ucsoBj79uBJ3eeChkmpvbpamhGCVBueygnLMEBgHlcv61Jhc7-GVMr_jTbl_70JXeUquUB3khFmkQO8E6xR5CeKxf8A7TC1gDOUR8pdEUMNGLrLI0pnwk3I3eYh7AIijGfHyNg2d42FI-VEpzQ9TJVrci7L97JFgrVKWtCC1FiNcaoWUyTr2zZ2cWVJKPPskJfzZHSfOODr9fmoSsvsTEwI1czTkq8cJktEIYFg8H0ETuOsY_ZfVgfXpGsSime2IL-28IMp3xhHnXS-v6fmgtWSJba1IS1K87jUY81uMgVbZB9cGvws-5CfjpNBQ'}
    Classifier: GradientBoostingClassifier has a training score of 86.7% accuracy score on Mode Fill Smote
    Classifier: GradientBoostingClassifier has a test score of 81.0% accuracy score on Mode Fill Smote


## 🔍 Analysis

Head back to [platform.seclea.com](https://platform.seclea.com) and we can analyse our Models
