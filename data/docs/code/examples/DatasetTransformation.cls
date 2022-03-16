
      >>> import pandas as pd
      >>>
      >>> def drop_correlated(data, thresh):
      >>>     import numpy as np
      >>>
      >>>     # calculate correlations
      >>>     corr_matrix = data.corr().abs()
      >>>     # get the upper part of correlation matrix
      >>>     upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
      >>>
      >>>     # columns with correlation above threshold
      >>>     redundant = [column for column in upper.columns if any(upper[column] >= thresh)]
      >>>     print(f"Columns to drop with correlation > {thresh}: {redundant}")
      >>>     data.drop(columns=redundant, inplace=True)
      >>>     return data
      >>>
      >>> def drop_nulls(df, threshold):
      >>>     cols = [x for x in df.columns if df[x].isnull().sum() / df.shape[0] > threshold]
      >>>     return df.drop(columns=cols)
      >>>
      >>> sample_df = pd.read_csv("some_data.csv")
      >>> corr_threshold = 0.97
      >>> sample_df1 = drop_correlated(data=sample_df, thresh=corr_threshold)
      >>>
      >>> null_threshold = 0.9
      >>> sample_df1 = drop_nulls(df=sample_df1, threshold=null_threshold)
      >>>
      >>> transformations = [
      >>>     DatasetTransformation(
      >>>         drop_correlated, {"data": sample_df}, {"thresh": corr_threshold}, ["df"]
      >>>     ),
      >>>     DatasetTransformation(
      >>>         drop_nulls, {"df": "inherit"}, {"threshold": null_threshold}, ["df"]
      >>>     ),
